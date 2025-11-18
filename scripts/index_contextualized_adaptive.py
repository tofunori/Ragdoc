#!/usr/bin/env python3
"""
Indexation Contextualized Embeddings avec stratégie ADAPTATIVE
Gère automatiquement les documents de toutes tailles

BASE DE DONNÉES SÉPARÉE : chroma_db_contextualized/

Stratégies :
1. Doc <20K tokens (~80K chars) → voyage-context-3 (contextualized)
2. Doc >=20K tokens → voyage-3-large (standard batched)

Batching: Les gros documents sont traités par batches de 200 chunks
pour respecter la limite de 120K tokens par appel API.
"""

import voyageai
import chromadb
from pathlib import Path
from typing import List, Tuple
import sys
import io
import os
from dotenv import load_dotenv

# Charger les variables d'environnement depuis .env
load_dotenv()

# Fix encoding pour Windows (éviter UnicodeEncodeError)
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent))
from indexing_config import (
    MARKDOWN_DIR,
    CHROMA_DB_CONTEXTUALIZED_PATH,
    COLLECTION_CONTEXTUALIZED_NAME,
    COLLECTION_CONTEXTUALIZED_METADATA,
    CHUNK_SIZE
)
from metadata_extractor import extract_metadata

# Constantes voyage-context-3
MAX_TOKENS_PER_DOC = 32000      # 32K tokens
MAX_TOKENS_PER_CALL = 120000    # 120K tokens total
MAX_CHUNKS_PER_CALL = 16000     # 16K chunks max
CHARS_PER_TOKEN = 4             # Approximation


class AdaptiveContextualizer:
    """Gère l'indexation adaptative selon taille documents"""

    def __init__(self):
        # Récupérer la clé API depuis l'environnement
        api_key = os.getenv("VOYAGE_API_KEY")
        if not api_key:
            raise ValueError("VOYAGE_API_KEY non trouvée dans .env")

        self.voyage_client = voyageai.Client(api_key=api_key)
        self.stats = {
            'small_docs': 0,      # <128K chars → entier
            'medium_docs': 0,     # 128K-480K → chunked contextualized
            'large_docs': 0,      # >480K → standard embeddings
            'total_chunks': 0
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
        doc_metadata: dict = None
    ) -> Tuple[List[str], List[dict], List[str], List[list]]:
        """
        Traite un document et retourne chunks, métadonnées, IDs, embeddings

        Args:
            content: Contenu du document
            doc_id: ID unique du document
            source_name: Nom du fichier source
            doc_metadata: Métadonnées extraites (author, date, title)

        Returns:
            chunks, metadatas, ids, embeddings
        """
        if doc_metadata is None:
            doc_metadata = {}
        doc_tokens = self.estimate_tokens(content)
        doc_chars = len(content)

        print(f"\n[DOC] {source_name}: {doc_chars:,} chars (~{doc_tokens:,} tokens)")

        # Seuil de sécurité : 20K tokens (~80K chars) pour éviter dépassements
        SAFE_TOKEN_LIMIT = 20000

        # === STRATÉGIE 1: Document petit → Contextualized ===
        if doc_tokens < SAFE_TOKEN_LIMIT:
            print(f"   [->] Strategie: CONTEXTUALIZED")
            self.stats['small_docs'] += 1

            # Créer chunks raisonnables (1500 chars pour meilleure granularité)
            chunks = self.chunk_text(content, chunk_size=1500)

            # Embeddings contextualisés (document entier comme contexte)
            embds_obj = self.voyage_client.contextualized_embed(
                inputs=[chunks],  # Un seul document, plusieurs chunks
                model="voyage-context-3",
                input_type="document"
            )

            embeddings = embds_obj.results[0].embeddings
            strategy = "contextualized_full"

        # === STRATÉGIE 2: Document grand → Standard avec batching ===
        else:
            print(f"   [->] Strategie: STANDARD (voyage-3-large) avec batching")
            self.stats['large_docs'] += 1

            # Chunks standard
            chunks = self.chunk_text(content, chunk_size=2000)

            # BATCHING: Envoyer par batches de 100 chunks (~50K tokens)
            # Marge de sécurité pour éviter de dépasser la limite de 120K tokens
            BATCH_SIZE = 100
            embeddings = []

            for i in range(0, len(chunks), BATCH_SIZE):
                batch_chunks = chunks[i:i + BATCH_SIZE]
                print(f"   [BATCH] Processing chunks {i}-{i+len(batch_chunks)} ({len(batch_chunks)} chunks)")

                embds_result = self.voyage_client.embed(
                    texts=batch_chunks,
                    model="voyage-3-large",
                    input_type="document"
                )

                embeddings.extend(embds_result.embeddings)

            strategy = "standard_batched"

            # Note: Ne plus compter medium_docs car stratégie 2 supprimée
            self.stats['medium_docs'] = 0

        # Préparer métadonnées
        metadatas = []
        ids = []

        for chunk_idx, chunk in enumerate(chunks):
            chunk_metadata = {
                "source": source_name,
                "chunk_index": chunk_idx,
                "total_chunks": len(chunks),
                "doc_id": doc_id,
                "doc_size_chars": doc_chars,
                "doc_size_tokens": doc_tokens,
                "embedding_strategy": strategy
            }

            # Ajouter métadonnées extraites (auteur, date, titre)
            if doc_metadata.get('author'):
                chunk_metadata['author'] = doc_metadata['author']
            if doc_metadata.get('date'):
                chunk_metadata['date'] = doc_metadata['date']
            if doc_metadata.get('title'):
                chunk_metadata['title'] = doc_metadata['title']

            metadatas.append(chunk_metadata)
            ids.append(f"doc_{doc_id}_chunk_{chunk_idx}")

        self.stats['total_chunks'] += len(chunks)

        print(f"   [OK] {len(chunks)} chunks, {len(embeddings)} embeddings")

        return chunks, metadatas, ids, embeddings

    def print_stats(self):
        """Affiche statistiques d'indexation"""
        print("\n" + "="*70)
        print("STATISTIQUES D'INDEXATION")
        print("="*70)
        print(f"Documents petits (<20K tokens):  {self.stats['small_docs']:4d} (contextualized)")
        print(f"Documents grands (>=20K tokens): {self.stats['large_docs']:4d} (standard batched)")
        print(f"Total chunks générés:            {self.stats['total_chunks']:4d}")
        print("="*70)


def index_with_adaptive_strategy():
    """Pipeline d'indexation adaptative"""

    print("[*] INDEXATION ADAPTATIVE CONTEXTUALIZED")
    print("="*70)
    print(f"Base de donnees: {CHROMA_DB_CONTEXTUALIZED_PATH}")
    print(f"Collection: {COLLECTION_CONTEXTUALIZED_NAME}")
    print("="*70)

    # 1. Initialiser
    contextualizer = AdaptiveContextualizer()

    # 2. Créer répertoire de la base si nécessaire
    CHROMA_DB_CONTEXTUALIZED_PATH.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Repertoire ChromaDB cree/verifie")

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_CONTEXTUALIZED_PATH))

    # 3. Créer collection (supprimer si existe)
    try:
        chroma_client.delete_collection(name=COLLECTION_CONTEXTUALIZED_NAME)
        print(f"[DEL] Collection existante supprimee")
    except:
        pass

    collection = chroma_client.create_collection(
        name=COLLECTION_CONTEXTUALIZED_NAME,
        metadata=COLLECTION_CONTEXTUALIZED_METADATA
    )

    print(f"[OK] Collection creee: {COLLECTION_CONTEXTUALIZED_NAME}\n")

    # 4. Traiter tous les documents
    md_files = sorted(MARKDOWN_DIR.glob("*.md"))
    print(f"[INFO] Trouve {len(md_files)} documents markdown\n")

    all_chunks = []
    all_metadatas = []
    all_ids = []
    all_embeddings = []

    for doc_id, md_file in enumerate(md_files):
        try:
            # Tenter UTF-8, puis latin-1 en fallback
            try:
                content = md_file.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                print(f"   [WARNING] UTF-8 failed for {md_file.name}, trying latin-1...")
                content = md_file.read_text(encoding='latin-1')

            # Extraire métadonnées (auteur, date, titre)
            doc_metadata = extract_metadata(content, md_file.name)
            print(f"   [META] Author: {doc_metadata.get('author', 'N/A')}, "
                  f"Date: {doc_metadata.get('date', 'N/A')}, "
                  f"Title: {doc_metadata.get('title', 'N/A')[:50] if doc_metadata.get('title') else 'N/A'}...")

            # Traitement adaptatif
            chunks, metadatas, ids, embeddings = contextualizer.process_document(
                content, doc_id, md_file.name, doc_metadata
            )

            all_chunks.extend(chunks)
            all_metadatas.extend(metadatas)
            all_ids.extend(ids)
            all_embeddings.extend(embeddings)

            # Indexer par batch de 1000 (éviter surcharge mémoire)
            if len(all_ids) >= 1000:
                print(f"\n   [SAVE] Indexation batch (1000 chunks)...")
                collection.add(
                    ids=all_ids,
                    documents=all_chunks,
                    embeddings=all_embeddings,
                    metadatas=all_metadatas
                )

                # Reset pour prochain batch
                all_chunks = []
                all_metadatas = []
                all_ids = []
                all_embeddings = []

        except Exception as e:
            print(f"   [ERROR] {e}")
            import traceback
            traceback.print_exc()
            continue

    # 5. Indexer dernier batch
    if all_ids:
        print(f"\n[SAVE] Indexation batch final ({len(all_ids)} chunks)...")
        collection.add(
            ids=all_ids,
            documents=all_chunks,
            embeddings=all_embeddings,
            metadatas=all_metadatas
        )

    # 6. Afficher stats
    contextualizer.print_stats()

    print(f"\n[OK] INDEXATION TERMINEE!")
    print(f"[STATS] Collection: {COLLECTION_CONTEXTUALIZED_NAME}")
    print(f"[STATS] Total chunks: {collection.count()}")
    print(f"[STATS] Base de donnees: {CHROMA_DB_CONTEXTUALIZED_PATH}")

    return COLLECTION_CONTEXTUALIZED_NAME


if __name__ == "__main__":
    try:
        collection_name = index_with_adaptive_strategy()
        print(f"\n[SUCCESS] Pret pour recherche avec: {collection_name}")
        print(f"\n[INFO] Pour tester: python tests/test_contextualized_search.py")
    except Exception as e:
        print(f"\n[FATAL] ERREUR FATALE: {e}")
        import traceback
        traceback.print_exc()
