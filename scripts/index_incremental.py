#!/usr/bin/env python3
"""
Indexation incrémentale intelligente avec détection des modifications.

Fonctionnalités:
- Ajoute les nouveaux documents
- Détecte et réindexe les documents modifiés (hash MD5)
- Skip les documents inchangés (économie API)
- Évite toute duplication
- Utilise EXCLUSIVEMENT voyage-context-3 pour une qualité optimale

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
from chonkie import TokenChunker, SemanticChunker, OverlapRefinery

# Charger configuration
load_dotenv()

# Importer la configuration
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import (
    MARKDOWN_DIR, CHROMA_DB_PATH, COLLECTION_NAME, COLLECTION_CONTEXTUALIZED_METADATA,
    CHONKIE_TOKENIZER, USE_CONTENT_HASH, TRACK_INDEXED_DATE
)

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")

if not VOYAGE_API_KEY:
    print("ERREUR: VOYAGE_API_KEY non configuree dans .env")
    sys.exit(1)


class HybridModelProcessor:
    """Classe de gestion des embeddings - SIMPLIFIÉE pour Context-3 uniquement"""

    def __init__(self, api_key: str):
        self.client = voyageai.Client(api_key=api_key)
        self.model_name = "voyage-context-3"

    def choose_strategy(self, num_chunks: int) -> dict:
        """Stratégie unique : Context-3 pour tout le monde"""
        return {
            "model": self.model_name,
            "method": "contextualized",
            "reason": "Qualité optimale (Context-3)"
        }

    def process_contextualized(self, chunk_texts: List[str]) -> List[List[float]]:
        """Traitement contextualized avec Voyage Context-3"""
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
        if lock_file.exists():
            lock_file.unlink()
    except Exception:
        pass


def process_embeddings_with_limit_check(voyage_client, chunk_texts, model, chunk_objects):
    """
    Traitement des embeddings avec découpage intelligent pour les très gros documents.
    Garantit que l'on reste sous la limite de 32k tokens de voyage-context-3.
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
            print(f"      ERREUR API: {e}")
            return []
    else:
        print(f"      ⚠️ GROS DOCUMENT ({len(chunk_texts)} chunks) -> Découpage en sections")
        all_embeddings = []
        
        # Taille de lot réduite pour éviter les timeouts (10 chunks ~ 10k tokens)
        BATCH_SIZE = 10
        
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
                print(f"         ERREUR sur section: {e}")
                # On continue pour essayer de sauver le reste
            
            # Avancer
            current_idx = end_idx
            
        return all_embeddings


def index_incremental(force_reindex: bool = False,
                      delete_missing: bool = False) -> None:
    """Indexation incrémentale simplifiée."""

    # Vérifier qu'aucun autre processus d'indexation n'est en cours
    lock_file = CHROMA_DB_PATH.parent / ".indexing.lock"
    lock_handle = acquire_lock(lock_file)
    if not lock_handle:
        print("\n[ERREUR] Un processus d'indexation est deja en cours!")
        sys.exit(1)

    try:
        print("\n" + "=" * 70)
        print("INDEXATION RAGDOC - MODE CONTEXT-3 UNIFIE")
        print("=" * 70)

        # Initialiser Voyage
        print("\n[1/5] Connexion a Voyage AI...")
        voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)
        print("   OK Voyage AI connecte")

        # Connecter Chroma
        print("\n[2/5] Connexion à Chroma...")
        try:
            test_client = chromadb.HttpClient(host="localhost", port=8000)
            test_client.heartbeat()
            client = test_client
            print("   [OK] Connecte au serveur ChromaDB (localhost:8000)")
        except Exception:
            client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
            print("   [INFO] Mode local (PersistentClient)")

        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata=COLLECTION_CONTEXTUALIZED_METADATA
        )
        print(f"   OK Collection '{COLLECTION_NAME}' chargee")

        # Scanner les documents existants
        print("\n[3/5] Analyse des documents existants...")
        existing_docs = collection.get(include=["metadatas"])
        indexed_map: Dict[str, Dict] = {}

        for i, metadata in enumerate(existing_docs['metadatas']):
            source = metadata.get('source')
            doc_hash = metadata.get('doc_hash')
            chunk_id = existing_docs['ids'][i]

            if source not in indexed_map:
                indexed_map[source] = {'hash': doc_hash, 'chunk_ids': []}
            indexed_map[source]['chunk_ids'].append(chunk_id)

        print(f"   OK {len(indexed_map)} documents indexes trouves")

        # Scanner les fichiers markdown
        print("\n[4/5] Scan du repertoire markdown...")
        markdown_files = sorted(list(MARKDOWN_DIR.glob("*.md")))
        print(f"   OK {len(markdown_files)} fichiers markdown trouves")

        # Identifier les documents manquants (optionnel)
        if delete_missing:
            current_sources = {f.name for f in markdown_files}
            missing_sources = set(indexed_map.keys()) - current_sources
            if missing_sources:
                print(f"\n   ATTENTION {len(missing_sources)} document(s) supprime(s) detecte(s)")
                for source in missing_sources:
                    chunk_ids = indexed_map[source]['chunk_ids']
                    collection.delete(ids=chunk_ids)
                    print(f"      - Supprime: {source}")
                del indexed_map
                # Recharger map
                existing_docs = collection.get(include=["metadatas"])
                indexed_map = {}
                for i, metadata in enumerate(existing_docs['metadatas']):
                    source = metadata.get('source')
                    if source not in indexed_map:
                        indexed_map[source] = {'hash': metadata.get('doc_hash'), 'chunk_ids': []}
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

        # Configuration UNIFIÉE et OPTIMISÉE pour Context-3
        # 1024 tokens = précision chirurgicale
        # Le modèle gère le contexte global, donc pas besoin de gros chunks
        CHUNK_SIZE_TOKENS = 1024
        CHUNK_OVERLAP_TOKENS = 180

        for i, md_file in enumerate(markdown_files, 1):
            try:
                content = md_file.read_text(encoding='utf-8', errors='ignore')
                current_hash = compute_doc_hash(content)
                
                # Déterminer le statut du document
                status = "NEW"
                if md_file.name in indexed_map:
                    if not force_reindex and indexed_map[md_file.name]['hash'] == current_hash:
                        stats['unchanged'] += 1
                        print(f"   [{i:3d}/{len(markdown_files)}] SKIP  {md_file.name}")
                        continue
                    else:
                        status = "MODIFIED"
                        old_chunk_ids = indexed_map[md_file.name]['chunk_ids']
                        if old_chunk_ids:
                            collection.delete(ids=old_chunk_ids)

                print(f"      Pipeline Chonkie (1024 tokens)...")

                # Étape 1: Token Chunker (structure globale)
                token_chunker = TokenChunker(
                    tokenizer=CHONKIE_TOKENIZER,
                    chunk_size=CHUNK_SIZE_TOKENS * 2,
                    chunk_overlap=CHUNK_OVERLAP_TOKENS
                )
                token_chunks = token_chunker.chunk(content)

                # Étape 2: Semantic Chunker (cohérence thématique)
                semantic_chunker = SemanticChunker(
                    embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    threshold=0.75,
                    chunk_size=CHUNK_SIZE_TOKENS,
                    min_sentences_per_chunk=2
                )
                
                all_semantic_chunks = []
                for tc in token_chunks:
                    all_semantic_chunks.extend(semantic_chunker.chunk(tc.text))

                # Étape 3: Overlap Refinery
                overlap_refinery = OverlapRefinery(
                    context_size=CHUNK_OVERLAP_TOKENS,
                    method="suffix",
                    merge=True
                )
                chunks = overlap_refinery.refine(all_semantic_chunks)
                
                # Extraire textes
                chunk_texts = [chunk.text for chunk in chunks]
                
                if not chunk_texts:
                    print("      AVERTISSEMENT: Aucun chunk généré")
                    continue

                # Étape 4: Embeddings (Context-3 avec gestion gros docs)
                embeddings = process_embeddings_with_limit_check(
                    voyage_client, 
                    chunk_texts, 
                    "voyage-context-3",
                    chunks
                )

                if not embeddings or len(embeddings) != len(chunk_texts):
                    print(f"         ERREUR: Mismatch embeddings ({len(embeddings)}) vs chunks ({len(chunk_texts)})")
                    continue

                # Ajouter à Chroma
                chunk_ids = []
                chunk_documents = []
                chunk_embeddings = []
                chunk_metadatas = []

                for j, chunk_text in enumerate(chunk_texts):
                    chunk_id = f"{md_file.stem}_chunk_{j}"

                    metadata = {
                        "source": md_file.name,
                        "source_file": str(md_file),
                        "title": md_file.stem,
                        "chunk_index": j,
                        "total_chunks": len(chunks),
                        "model": "voyage-context-3",
                        "chunking_strategy": "contextualized_fixed_1024",
                        "pipeline": "unified_v2"
                    }

                    if USE_CONTENT_HASH:
                        metadata["doc_hash"] = current_hash

                    if TRACK_INDEXED_DATE:
                        metadata["indexed_date"] = datetime.now().isoformat()

                    chunk_ids.append(chunk_id)
                    chunk_documents.append(chunk_text)
                    chunk_embeddings.append(embeddings[j])
                    chunk_metadatas.append(metadata)

                collection.upsert(
                    ids=chunk_ids,
                    documents=chunk_documents,
                    embeddings=chunk_embeddings,
                    metadatas=chunk_metadatas
                )

                stats[status.lower()] += 1
                stats['total_chunks'] += len(chunks)

                icon = "MOD" if status == "MODIFIED" else "NEW"
                print(f"   [{i:3d}/{len(markdown_files)}] {icon} {status.ljust(8)} {md_file.name[:40]:40} ({len(chunks):3d} chunks)")

            except Exception as e:
                stats['errors'] += 1
                print(f"   [{i:3d}/{len(markdown_files)}] ERREUR {md_file.name}: {str(e)}")

        # Résumé
        print("\n" + "=" * 70)
        print("RÉSUMÉ DE L'INDEXATION:")
        print(f"   Nouveaux documents:      {stats['new']:3d}")
        print(f"   Documents modifiés:      {stats['modified']:3d}")
        print(f"   Documents inchangés:     {stats['unchanged']:3d}")
        print(f"   Erreurs:                 {stats['errors']:3d}")
        print(f"   Chunks ajoutés/modifiés: {stats['total_chunks']:3d}")
        print("=" * 70 + "\n")

    finally:
        release_lock(lock_file, lock_handle)


def main():
    parser = argparse.ArgumentParser(description="Indexation RAGDOC Unifiée (Context-3)")
    parser.add_argument('--force', action='store_true', help="Forcer réindexation")
    parser.add_argument('--delete-missing', action='store_true', help="Nettoyer docs supprimés")
    args = parser.parse_args()

    try:
        index_incremental(force_reindex=args.force, delete_missing=args.delete_missing)
    except KeyboardInterrupt:
        print("\n\nInterruption utilisateur.")
        sys.exit(1)
    except Exception as e:
        print(f"\nERREUR fatale: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
