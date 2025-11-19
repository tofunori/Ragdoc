#!/usr/bin/env python3
"""
Indexation INCRÉMENTALE Contextualized Embeddings (Unified Voyage-Context-3)
Détecte et indexe uniquement les documents nouveaux ou modifiés
Utilise voyage-context-3 pour TOUS les documents (limite 32k tokens de contexte)

BASE DE DONNÉES: chroma_db_contextualized/
COLLECTION: ragdoc_contextualized_v1

Fonctionnalités:
- Détection MD5 des modifications
- Skip documents inchangés (économie API 90-95%)
- Suppression optionnelle des documents absents
- Stratégie unifiée: voyage-context-3 (plus de bascule vers large-3)

Commandes:
  python index_contextualized_incremental.py           # Indexation incrémentale
  python index_contextualized_incremental.py --force   # Réindexation forcée
  python index_contextualized_incremental.py --delete-missing  # Supprime docs absents
"""

import voyageai
import chromadb
from pathlib import Path
from typing import List, Tuple, Optional, TextIO
import sys
import io
import os
import hashlib
import argparse
from datetime import datetime
from dotenv import load_dotenv

# Charger les variables d'environnement depuis .env
load_dotenv()

# Fix encoding pour Windows (éviter UnicodeEncodeError)
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Imports depuis le package src
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import (
    MARKDOWN_DIR,
    CHROMA_DB_CONTEXTUALIZED_PATH,
    COLLECTION_CONTEXTUALIZED_NAME,
    COLLECTION_CONTEXTUALIZED_METADATA,
    CHUNK_SIZE,
    DEFAULT_MODEL
)

# Constantes voyage-context-3
CHARS_PER_TOKEN = 4             # Approximation

# ============================================================================
# UTILITY FUNCTIONS (from index_incremental.py)
# ============================================================================

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
                # Garder le handle ouvert
                return lock_handle
            except FileExistsError:
                # Fichier existe déjà = autre processus en cours
                return None
        else:
            # Unix/Linux
            import fcntl
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


# ============================================================================
# UNIFIED CONTEXTUALIZER
# ============================================================================

class UnifiedContextualizer:
    """Gère l'indexation unifiée avec voyage-context-3"""

    def __init__(self):
        # Récupérer la clé API depuis l'environnement
        api_key = os.getenv("VOYAGE_API_KEY")
        if not api_key:
            raise ValueError("VOYAGE_API_KEY non trouvée dans .env")

        self.voyage_client = voyageai.Client(api_key=api_key)
        self.stats = {
            'docs_processed': 0,
            'total_chunks': 0,
            'new_docs': 0,
            'modified_docs': 0,
            'unchanged_docs': 0,
            'errors': 0
        }

    def estimate_tokens(self, text: str) -> int:
        """Estime nombre de tokens (1 token ~= 4 chars)"""
        return len(text) // CHARS_PER_TOKEN

    def chunk_text(self, text: str, chunk_size: int = 2000) -> List[str]:
        """Découpe texte en chunks SANS overlap (requis pour contextualized)"""
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i + chunk_size])
        return chunks

    def process_document(
        self,
        content: str,
        doc_id: int,
        source_name: str,
        doc_hash: str
    ) -> Tuple[List[str], List[dict], List[str], List[list]]:
        """
        Traite un document et retourne chunks, métadonnées, IDs, embeddings
        
        Returns:
            chunks, metadatas, ids, embeddings
        """
        doc_tokens = self.estimate_tokens(content)
        doc_chars = len(content)

        print(f"   {source_name}: {doc_chars:,} chars (~{doc_tokens:,} tokens)")

        self.stats['docs_processed'] += 1

        # Utiliser la taille de chunk définie dans la config
        current_chunk_size = 1500 # Default for context-3
        
        chunks = self.chunk_text(content, chunk_size=current_chunk_size)

        # Embeddings contextualisés (document entier comme contexte)
        # Note: voyage-context-3 accepte jusqu'à 32k tokens de contexte
        # Si le document est plus grand, l'API tronquera automatiquement le contexte
        # pour les chunks de la fin, mais ça ne plantera pas.
        embds_obj = self.voyage_client.contextualized_embed(
            inputs=[chunks],  # Un seul document, plusieurs chunks
            model=DEFAULT_MODEL,
            input_type="document"
        )

        embeddings = embds_obj.results[0].embeddings
        strategy = "contextualized_full"
        model_used = DEFAULT_MODEL

        # Préparer métadonnées
        metadatas = []
        ids = []
        indexed_date = datetime.now().isoformat()

        for chunk_idx, chunk in enumerate(chunks):
            metadatas.append({
                "source": source_name,
                "chunk_index": chunk_idx,
                "total_chunks": len(chunks),
                "doc_id": doc_id,
                "doc_size_chars": doc_chars,
                "doc_size_tokens": doc_tokens,
                "embedding_strategy": strategy,
                "model": model_used,
                "doc_hash": doc_hash,
                "indexed_date": indexed_date
            })
            ids.append(f"doc_{doc_id}_chunk_{chunk_idx}")

        self.stats['total_chunks'] += len(chunks)

        print(f"      [OK] {len(chunks)} chunks, {len(embeddings)} embeddings (Model: {model_used})")

        return chunks, metadatas, ids, embeddings

    def print_stats(self):
        """Affiche statistiques d'indexation"""
        print("\n" + "="*70)
        print("STATISTIQUES D'INDEXATION INCREMENTALE (UNIFIEE)")
        print("="*70)
        print(f"Documents nouveaux:               {self.stats['new_docs']:4d}")
        print(f"Documents modifiés:               {self.stats['modified_docs']:4d}")
        print(f"Documents inchangés (skipped):    {self.stats['unchanged_docs']:4d}")
        print(f"Erreurs:                          {self.stats['errors']:4d}")
        print("-"*70)
        print(f"Total docs traités:               {self.stats['docs_processed']:4d}")
        print(f"Total chunks générés:             {self.stats['total_chunks']:4d}")
        print("="*70)


# ============================================================================
# MAIN INCREMENTAL INDEXATION FUNCTION
# ============================================================================

def index_contextualized_incremental(
    force_reindex: bool = False,
    delete_missing: bool = False
) -> str:
    """
    Indexation incrémentale contextualized avec détection des modifications.

    Args:
        force_reindex: Force la réindexation de tous les documents
        delete_missing: Supprime les chunks des documents absents du filesystem

    Returns:
        Collection name
    """

    # [1] File locking
    lock_file = CHROMA_DB_CONTEXTUALIZED_PATH.parent / ".indexing_contextualized.lock"
    lock_handle = acquire_lock(lock_file)

    if not lock_handle:
        print("[ERROR] Indexation déjà en cours (fichier .lock trouvé)")
        print("        Si vous êtes sûr qu'aucune indexation n'est en cours,")
        print(f"        supprimez le fichier: {lock_file}")
        sys.exit(1)

    try:
        # [2] Initialize
        print("[*] INDEXATION INCREMENTALE UNIFIEE (Context-3)")
        print("="*70)
        print(f"Base de donnees: {CHROMA_DB_CONTEXTUALIZED_PATH}")
        print(f"Collection: {COLLECTION_CONTEXTUALIZED_NAME}")
        if force_reindex:
            print("[FORCE] Mode réindexation forcée activé")
        if delete_missing:
            print("[DELETE] Suppression des documents absents activée")
        print("="*70)

        # [2a] Voyage AI
        contextualizer = UnifiedContextualizer()

        # [2b] ChromaDB
        CHROMA_DB_CONTEXTUALIZED_PATH.mkdir(parents=True, exist_ok=True)
        print(f"[OK] Repertoire ChromaDB cree/verifie")

        chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_CONTEXTUALIZED_PATH))

        # [2c] Get or create collection (NO deletion!)
        collection = chroma_client.get_or_create_collection(
            name=COLLECTION_CONTEXTUALIZED_NAME,
            metadata=COLLECTION_CONTEXTUALIZED_METADATA
        )
        print(f"[OK] Collection chargee/creee: {COLLECTION_CONTEXTUALIZED_NAME}\n")

        # [3] Scan existing index
        print("[*] Scan de l'index existant...")
        existing_docs = collection.get(include=["metadatas"])
        indexed_map = {}  # {source: {hash, chunk_ids}}

        for i, metadata in enumerate(existing_docs['metadatas']):
            source = metadata.get('source')
            doc_hash = metadata.get('doc_hash')
            chunk_id = existing_docs['ids'][i]

            if source not in indexed_map:
                indexed_map[source] = {'hash': doc_hash, 'chunk_ids': []}
            indexed_map[source]['chunk_ids'].append(chunk_id)

        print(f"[OK] Trouvé {len(indexed_map)} documents dans l'index ({len(existing_docs['ids'])} chunks)\n")

        # [4] Scan markdown files
        markdown_files = sorted(MARKDOWN_DIR.glob("*.md"))
        print(f"[INFO] Trouvé {len(markdown_files)} documents markdown\n")

        # [5] Optional: delete missing files
        if delete_missing:
            current_sources = {f.name for f in markdown_files}
            missing_sources = set(indexed_map.keys()) - current_sources

            if missing_sources:
                print(f"[DELETE] Suppression de {len(missing_sources)} documents absents:")
                for source in missing_sources:
                    chunk_ids = indexed_map[source]['chunk_ids']
                    if chunk_ids:
                        collection.delete(ids=chunk_ids)
                        print(f"   [DEL] {source} ({len(chunk_ids)} chunks)")
                print()

        # [6] Incremental processing
        for doc_id, md_file in enumerate(markdown_files):
            try:
                # Tenter UTF-8, puis latin-1 en fallback
                try:
                    content = md_file.read_text(encoding='utf-8')
                except UnicodeDecodeError:
                    print(f"   [WARNING] UTF-8 failed for {md_file.name}, trying latin-1...")
                    content = md_file.read_text(encoding='latin-1')

                current_hash = compute_doc_hash(content)

                # Determine status
                status = "NEW"
                if md_file.name in indexed_map:
                    if not force_reindex and indexed_map[md_file.name]['hash'] == current_hash:
                        # Document unchanged - skip
                        contextualizer.stats['unchanged_docs'] += 1
                        print(f"[SKIP] {md_file.name} (unchanged)")
                        continue
                    else:
                        status = "MODIFIED"
                        # Delete old chunks
                        old_chunk_ids = indexed_map[md_file.name]['chunk_ids']
                        if old_chunk_ids:
                            collection.delete(ids=old_chunk_ids)
                            print(f"[DEL] {md_file.name} (removing {len(old_chunk_ids)} old chunks)")

                # Process document with unified strategy
                print(f"\n[{status}] {md_file.name}")
                chunks, metadatas, ids, embeddings = contextualizer.process_document(
                    content, doc_id, md_file.name, current_hash
                )

                # Upsert to collection
                collection.upsert(
                    ids=ids,
                    documents=chunks,
                    embeddings=embeddings,
                    metadatas=metadatas
                )

                if status == "NEW":
                    contextualizer.stats['new_docs'] += 1
                else:
                    contextualizer.stats['modified_docs'] += 1

            except Exception as e:
                contextualizer.stats['errors'] += 1
                print(f"   [ERROR] {md_file.name}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # [7] Print statistics
        contextualizer.print_stats()

        print(f"\n[OK] INDEXATION INCREMENTALE TERMINEE!")
        print(f"[STATS] Collection: {COLLECTION_CONTEXTUALIZED_NAME}")
        print(f"[STATS] Total chunks dans collection: {collection.count()}")
        print(f"[STATS] Base de donnees: {CHROMA_DB_CONTEXTUALIZED_PATH}")

        return COLLECTION_CONTEXTUALIZED_NAME

    finally:
        release_lock(lock_file, lock_handle)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Indexation incrémentale unifiée (Context-3)"
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help="Force la réindexation de tous les documents"
    )
    parser.add_argument(
        '--delete-missing',
        action='store_true',
        help="Supprime les chunks des documents absents du filesystem"
    )
    args = parser.parse_args()

    try:
        collection_name = index_contextualized_incremental(
            force_reindex=args.force,
            delete_missing=args.delete_missing
        )
        print(f"\n[SUCCESS] Pret pour recherche avec: {collection_name}")
    except Exception as e:
        print(f"\n[FATAL] ERREUR FATALE: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
