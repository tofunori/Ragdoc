#!/usr/bin/env python3
"""
Indexation incrémentale intelligente avec détection des modifications.

Fonctionnalités:
- Ajoute les nouveaux documents
- Détecte et réindexe les documents modifiés (hash MD5)
- Skip les documents inchangés (économie API)
- Évite toute duplication
- Utilise le modèle contextualisé configuré dans RAGDOC_EMBEDDING_MODEL

Usage:
    python index_incremental.py                  # Indexation normale
    python index_incremental.py --force          # Forcer réindexation complète
    python index_incremental.py --delete-missing # Supprimer docs absents
"""

import os
import sys
import hashlib
import argparse
import warnings
import logging
import math
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, TextIO, Tuple

# Fix encoding pour Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# Imports conditionnels pour le verrou de fichier
if sys.platform != "win32":
    import fcntl  # Pour le verrou de fichier (Unix)
else:
    fcntl = None  # Pas disponible sur Windows

from dotenv import load_dotenv
import chromadb
import voyageai
from chonkie import TokenChunker

# Charger configuration
load_dotenv()

# Importer la configuration
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import (
    MARKDOWN_DIR, CHROMA_DB_PATH, COLLECTION_NAME, COLLECTION_CONTEXTUALIZED_METADATA,
    CHONKIE_TOKENIZER, USE_CONTENT_HASH, TRACK_INDEXED_DATE, LIBRARY_PATH,
    EMBEDDING_MODEL, LEXICAL_INDEX_PATH,
)
from src.library import Library, read_sidecar, document_metadata, locate_chunks
from src.index_safety import replace_document, bump_revision, update_collection_state, IndexRepairRequired
from src.chroma_reads import read_collection
from src.chroma_connection import open_chroma_client
from src.lexical_index import PersistentLexicalIndex

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")

class HybridModelProcessor:
    """Classe de gestion des embeddings contextualisés."""

    def __init__(self, api_key: str):
        self.client = voyageai.Client(api_key=api_key)
        self.model_name = EMBEDDING_MODEL

    def choose_strategy(self, num_chunks: int) -> dict:
        """Stratégie contextualisée unique pour tout le monde."""
        return {
            "model": self.model_name,
            "method": "contextualized",
            "reason": f"Embeddings contextualisés ({self.model_name})"
        }

    def process_contextualized(self, chunk_texts: List[str]) -> List[List[float]]:
        """Traitement contextualisé avec le modèle Voyage configuré."""
        try:
            result = self.client.contextualized_embed(
                inputs=[chunk_texts],
                model=self.model_name,
                input_type="document"
            )
            return result.results[0].embeddings
        except Exception as e:
            print(f"         ERREUR CRITIQUE API Voyage: {e}")
            return []

    def process_with_strategy(self, chunk_texts: List[str]) -> Tuple[str, List[List[float]], dict]:
        """Point d'entrée principal"""
        strategy = self.choose_strategy(len(chunk_texts))
        # Note: Le batching est géré en amont dans process_embeddings_with_limit_check
        embeddings = self.process_contextualized(chunk_texts)
        return strategy["model"], embeddings, strategy


def compute_doc_hash(content: str) -> str:
    """Calcule hash MD5 du contenu pour détection de modifications."""
    return hashlib.md5(content.encode('utf-8')).hexdigest()


LockHandle = Optional[TextIO]


def acquire_lock(lock_file: Path) -> LockHandle:
    """Acquérir un verrou de fichier pour éviter les accès concurrents."""
    try:
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        
        if sys.platform == "win32":
            # Windows - utiliser création exclusive de fichier
            try:
                # Essayer de créer le fichier en mode exclusif
                lock_handle = open(lock_file, 'x')
                lock_handle.write(str(os.getpid()))
                lock_handle.flush()
                return lock_handle
            except FileExistsError:
                return None
        else:
            # Unix/Linux
            lock_handle = open(lock_file, 'w')
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            lock_handle.write(str(os.getpid()))
            lock_handle.flush()
            return lock_handle
    except (IOError, OSError):
        return None


def release_lock(lock_file: Path, lock_handle: LockHandle):
    """Libérer le verrou de fichier."""
    try:
        if lock_handle and not lock_handle.closed:
            lock_handle.close()
    except Exception:
        pass

    try:
        if sys.platform == "win32" and lock_file.exists():
            lock_file.unlink()
    except Exception:
        pass


def process_embeddings_with_limit_check(voyage_client, chunk_texts, model, chunk_objects):
    """
    Traitement des embeddings avec découpage intelligent pour les très gros documents.
    Garantit que l'on reste sous la limite de 32k tokens du modèle contextualisé.
    """
    
    # Estimer les tokens totaux
    total_tokens = sum(chunk.token_count for chunk in chunk_objects)
    
    # Limite de sécurité (32k théorique, on prend une marge)
    SAFE_TOKEN_LIMIT = 30000
    
    if total_tokens < SAFE_TOKEN_LIMIT:
        # Cas standard : tout passe d'un coup
        print(f"      Traitement standard ({total_tokens:,} tokens)")
        try:
            result = voyage_client.contextualized_embed(
                inputs=[chunk_texts],
                model=model,
                input_type="document"
            )
            return result.results[0].embeddings
        except Exception as e:
            raise RuntimeError("Voyage embedding request failed") from e
    else:
        print(f"      ⚠️ GROS DOCUMENT ({len(chunk_texts)} chunks) -> Découpage en sections")
        all_embeddings = []
        
        # 25 x 1024 tokens stays below the 30k safety budget while avoiding
        # excessive round trips for books and long reports.
        BATCH_SIZE = 25
        
        current_idx = 0
        while current_idx < len(chunk_texts):
            end_idx = min(current_idx + BATCH_SIZE, len(chunk_texts))
            
            # On prend une section + contexte
            batch_texts = chunk_texts[current_idx:end_idx]
            
            print(f"         Section {current_idx}-{end_idx} ({len(batch_texts)} chunks)...")
            
            try:
                # Appel API avec timeout augmenté
                result = voyage_client.contextualized_embed(
                    inputs=[batch_texts],
                    model=model,
                    input_type="document"
                )
                
                # Récupérer les embeddings
                batch_embeddings = result.results[0].embeddings
                
                # Si on a du chevauchement, il faut gérer les doublons ou juste avancer
                # Ici stratégie simple : on concatène tout (les embeddings seront contextuels à leur section)
                all_embeddings.extend(batch_embeddings)
                
            except Exception as e:
                raise RuntimeError(f"Voyage embedding batch {current_idx}-{end_idx} failed") from e
            
            # Avancer
            current_idx = end_idx
            
        return all_embeddings


def index_incremental(force_reindex: bool = False,
                      delete_missing: bool = False,
                      sources: Optional[List[str]] = None) -> dict:
    """Indexation incrémentale simplifiée."""

    if not VOYAGE_API_KEY:
        raise RuntimeError("VOYAGE_API_KEY is required for indexing")
    library = Library(LIBRARY_PATH)
    # Vérifier qu'aucun autre processus d'indexation n'est en cours
    lock_file = CHROMA_DB_PATH.parent / ".indexing.lock"
    lock_handle = acquire_lock(lock_file)
    if not lock_handle:
        print("\n[ERREUR] Un processus d'indexation est deja en cours!")
        sys.exit(1)

    try:
        print("\n" + "=" * 70)
        print(f"INDEXATION RAGDOC - {EMBEDDING_MODEL}")
        print("=" * 70)

        # Initialiser Voyage
        print("\n[1/5] Connexion a Voyage AI...")
        voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)
        print("   OK Voyage AI connecte")

        # Connecter Chroma
        print("\n[2/5] Connexion à Chroma...")
        client, chroma_connection_mode = open_chroma_client(chromadb, CHROMA_DB_PATH)
        if chroma_connection_mode == "persistent-forced":
            print(f"   [INFO] Mode local force (PersistentClient): {CHROMA_DB_PATH}")
        elif chroma_connection_mode == "http":
            print("   [OK] Connecte au serveur ChromaDB (localhost:8000)")
        else:
            print("   [INFO] Mode local (PersistentClient)")

        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata=COLLECTION_CONTEXTUALIZED_METADATA
        )
        collection_model = (collection.metadata or {}).get("embedding_model")
        if collection_model != EMBEDDING_MODEL:
            raise RuntimeError(
                f"Embedding model mismatch: collection={collection_model!r}, "
                f"configured={EMBEDDING_MODEL!r}. Refusing mixed-model writes."
            )
        initial_revision = (collection.metadata or {}).get("ragdoc_revision")
        repairing = ((collection.metadata or {}).get("ragdoc_write_state", "ready") != "ready"
                     or (collection.metadata or {}).get("ragdoc_repairing", False))
        if repairing and sources:
            raise RuntimeError("Targeted --source indexing is disabled while a global index repair is required")
        if repairing and not force_reindex:
            raise RuntimeError("Interrupted index write detected. Inspect ingestion status and repair with --force.")
        if repairing:
            update_collection_state(collection, ragdoc_repairing=True)
        print(f"   OK Collection '{COLLECTION_NAME}' chargee")

        # Scanner les documents existants
        print("\n[3/5] Analyse des documents existants...")
        existing_docs = read_collection(collection, include=["metadatas"])
        indexed_map: Dict[str, Dict] = {}

        for i, metadata in enumerate(existing_docs['metadatas']):
            source = metadata.get('source')
            doc_hash = metadata.get('doc_hash')
            chunk_id = existing_docs['ids'][i]

            if source not in indexed_map:
                indexed_map[source] = {'hash': doc_hash, 'chunk_ids': [], 'metadata': metadata}
            indexed_map[source]['chunk_ids'].append(chunk_id)

        print(f"   OK {len(indexed_map)} documents indexes trouves")

        # Scanner les fichiers markdown
        print("\n[4/5] Scan du repertoire markdown...")
        all_markdown_files = sorted(list(MARKDOWN_DIR.glob("*.md")))
        markdown_files = all_markdown_files
        if sources:
            requested = set(sources)
            available = {path.name for path in all_markdown_files}
            missing = requested - available
            if missing:
                raise RuntimeError(f"Requested Markdown files missing: {', '.join(sorted(missing))}")
            markdown_files = [path for path in all_markdown_files if path.name in requested]
            print(f"   OK {len(markdown_files)} fichiers cibles sur {len(all_markdown_files)}")
        else:
            print(f"   OK {len(markdown_files)} fichiers markdown trouves")
        if not MARKDOWN_DIR.is_dir():
            raise RuntimeError("Markdown directory missing; refusing index cleanup")
        if repairing and set(indexed_map) - {p.name for p in all_markdown_files} and not delete_missing:
            raise RuntimeError("Repair requires missing source files to be restored or explicit --delete-missing")

        # Identifier les documents manquants (optionnel)
        changed_sources: set[str] = set()
        if delete_missing:
            if sources:
                raise RuntimeError("--delete-missing cannot be combined with --source")
            current_sources = {f.name for f in all_markdown_files}
            missing_sources = set(indexed_map.keys()) - current_sources
            if missing_sources:
                print(f"\n   ATTENTION {len(missing_sources)} document(s) supprime(s) detecte(s)")
                for source in missing_sources:
                    chunk_ids = indexed_map[source]['chunk_ids']
                    bump_revision(collection, "writing")
                    collection.delete(ids=chunk_ids)
                    bump_revision(collection)
                    changed_sources.add(source)
                    library.record(source, "removed")
                    print(f"      - Supprime: {source}")
                del indexed_map
                # Recharger map
                existing_docs = read_collection(collection, include=["metadatas"])
                indexed_map = {}
                for i, metadata in enumerate(existing_docs['metadatas']):
                    source = metadata.get('source')
                    if source not in indexed_map:
                        indexed_map[source] = {'hash': metadata.get('doc_hash'), 'chunk_ids': [], 'metadata': metadata}
                    indexed_map[source]['chunk_ids'].append(existing_docs['ids'][i])

        # Indexation incrémentale
        print("\n[5/5] Indexation incrémentale...\n")

        stats = {
            'new': 0,
            'modified': 0,
            'unchanged': 0,
            'errors': 0,
            'total_chunks': 0
        }

        # Configuration unifiée; stable across the Context 4 migration.
        # 1024 tokens = précision chirurgicale
        # Le modèle gère le contexte global, donc pas besoin de gros chunks
        CHUNK_SIZE_TOKENS = 1024
        CHUNK_OVERLAP_TOKENS = 180

        for i, md_file in enumerate(markdown_files, 1):
            try:
                content = md_file.read_text(encoding='utf-8')
                current_hash = compute_doc_hash(content)
                sidecar = read_sidecar(md_file)
                if sidecar.get('completeness') == 'partial':
                    raise ValueError("Partial/test PDF conversion must not replace a full indexed article")
                provenance_meta = document_metadata(md_file, content, sidecar)
                
                # Déterminer le statut du document
                status = "NEW"
                if md_file.name in indexed_map:
                    old_meta = indexed_map[md_file.name].get('metadata', {})
                    current_scientific_metadata = (
                        old_meta.get('metadata_sha256') == provenance_meta['metadata_sha256']
                        and old_meta.get('pipeline') == 'scientific_v3'
                    )
                    legacy_without_sidecar = not sidecar and old_meta.get('pipeline') != 'scientific_v3'
                    if (not force_reindex and indexed_map[md_file.name]['hash'] == current_hash
                            and (current_scientific_metadata or legacy_without_sidecar)):
                        stats['unchanged'] += 1
                        print(f"   [{i:3d}/{len(markdown_files)}] SKIP  {md_file.name}")
                        continue
                    else:
                        status = "MODIFIED"

                library.snapshot(content)
                library.record(md_file.name, "preparing", provenance_meta['canonical_sha256'])

                print(f"      Pipeline Chonkie déterministe (1024 tokens)...")

                # Voyage contextualizes each group remotely. A second local embedding
                # model added substantial CPU cost and made chunk boundaries non-reproducible.
                token_chunker = TokenChunker(
                    tokenizer=CHONKIE_TOKENIZER,
                    chunk_size=CHUNK_SIZE_TOKENS,
                    chunk_overlap=CHUNK_OVERLAP_TOKENS
                )
                chunks = token_chunker.chunk(content)
                
                # Extraire textes
                chunk_texts = [chunk.text for chunk in chunks]
                
                if not chunk_texts:
                    raise ValueError("No chunks generated; previous index preserved")

                # Étape 4: embeddings contextualisés avec gestion des gros documents
                embeddings = process_embeddings_with_limit_check(
                    voyage_client, 
                    chunk_texts, 
                    EMBEDDING_MODEL,
                    chunks
                )

                if not embeddings or len(embeddings) != len(chunk_texts):
                    raise ValueError(f"Embedding count mismatch: {len(embeddings)} vs {len(chunk_texts)}")
                dimension = len(embeddings[0])
                if not dimension or any(len(v) != dimension or not all(math.isfinite(x) for x in v) for v in embeddings):
                    raise ValueError("Invalid embedding vectors; previous index preserved")
                locations = locate_chunks(content, chunk_texts, sidecar)

                # Ajouter à Chroma
                chunk_ids = []
                chunk_documents = []
                chunk_embeddings = []
                chunk_metadatas = []

                for j, chunk_text in enumerate(chunk_texts):
                    # IDs include exact chunk text and source revision; old citations cannot silently change.
                    text_hash = hashlib.sha256(chunk_text.encode('utf-8')).hexdigest()[:16]
                    chunk_id = f"{md_file.stem}_chunk_{j}_{provenance_meta['canonical_sha256'][:16]}_{text_hash}"

                    metadata = {
                        "source": md_file.name,
                        "source_file": str(md_file),
                        "title": md_file.stem,
                        "chunk_index": j,
                        "total_chunks": len(chunks),
                        "model": EMBEDDING_MODEL,
                        "chunking_strategy": "contextualized_token_1024_overlap_180",
                        "pipeline": "scientific_v3",
                        **provenance_meta,
                        **locations[j],
                    }

                    if USE_CONTENT_HASH:
                        metadata["doc_hash"] = current_hash

                    if TRACK_INDEXED_DATE:
                        metadata["indexed_date"] = datetime.now().isoformat()

                    chunk_ids.append(chunk_id)
                    chunk_documents.append(chunk_text)
                    chunk_embeddings.append(embeddings[j])
                    chunk_metadatas.append(metadata)

                replace_document(collection, md_file.name, {
                    "ids": chunk_ids, "documents": chunk_documents,
                    "embeddings": chunk_embeddings, "metadatas": chunk_metadatas,
                })
                changed_sources.add(md_file.name)
                library.record(md_file.name, "ready", provenance_meta['canonical_sha256'])

                stats[status.lower()] += 1
                stats['total_chunks'] += len(chunks)

                icon = "MOD" if status == "MODIFIED" else "NEW"
                print(f"   [{i:3d}/{len(markdown_files)}] {icon} {status.ljust(8)} {md_file.name[:40]:40} ({len(chunks):3d} chunks)")

            except Exception as e:
                stats['errors'] += 1
                library.record(md_file.name, "failed", error=str(e))
                print(f"   [{i:3d}/{len(markdown_files)}] ERREUR {md_file.name}: {str(e)}")
                if isinstance(e, IndexRepairRequired):
                    raise

        # Résumé
        print("\n" + "=" * 70)
        print("RÉSUMÉ DE L'INDEXATION:")
        print(f"   Nouveaux documents:      {stats['new']:3d}")
        print(f"   Documents modifiés:      {stats['modified']:3d}")
        print(f"   Documents inchangés:     {stats['unchanged']:3d}")
        print(f"   Erreurs:                 {stats['errors']:3d}")
        print(f"   Chunks ajoutés/modifiés: {stats['total_chunks']:3d}")
        print("=" * 70 + "\n")
        if repairing and not stats['errors']:
            update_collection_state(collection, ragdoc_repairing=False)
            bump_revision(collection)
        if changed_sources:
            lexical_status = PersistentLexicalIndex(LEXICAL_INDEX_PATH).sync_sources(
                collection, changed_sources, initial_revision
            )
            print(
                f"   Index lexical: {lexical_status['chunks']} passages, "
                f"révision {str(lexical_status['revision'])[:12]}"
            )
        return stats

    finally:
        release_lock(lock_file, lock_handle)


def main():
    parser = argparse.ArgumentParser(description=f"Indexation RAGDOC ({EMBEDDING_MODEL})")
    parser.add_argument('--force', action='store_true', help="Forcer réindexation")
    parser.add_argument('--delete-missing', action='store_true', help="Nettoyer docs supprimés")
    parser.add_argument('--source', action='append', help="Indexer seulement ce fichier Markdown (répétable)")
    args = parser.parse_args()

    try:
        stats = index_incremental(
            force_reindex=args.force,
            delete_missing=args.delete_missing,
            sources=args.source,
        )
        if stats['errors']:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nInterruption utilisateur.")
        sys.exit(1)
    except Exception as e:
        print(f"\nERREUR fatale: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
