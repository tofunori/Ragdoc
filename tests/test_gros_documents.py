#!/usr/bin/env python3
"""
Test de la gestion des gros documents avec les nouvelles fonctions
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import MARKDOWN_DIR
from dotenv import load_dotenv
import chromadb
import voyageai
from chonkie import TokenChunker

# Importer les nouvelles fonctions
from index_incremental import (
    process_embeddings_with_limit_check,
    process_in_batches,
    process_with_large_model,
    process_normally
)

load_dotenv()

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
if not VOYAGE_API_KEY:
    print("ERREUR: VOYAGE_API_KEY non trouve")
    sys.exit(1)

def test_gros_documents():
    """Tester spécifiquement les gros documents problématiques"""

    print("Test de la gestion des gros documents")
    print("=" * 50)

    # Initialiser
    voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)
    chunker = TokenChunker(
        tokenizer="gpt2",
        chunk_size=1024,
        chunk_overlap=180
    )

    # Documents problématiques identifiés
    problem_docs = [
        "1982_RGSP.md",  # 41 chunks (34,601 tokens)
        "A Model for the Spectral Albedo of Snow. I Pure Snow.md",  # 47 chunks (39,766 tokens)
        "Bond et al. - 2013 - Bounding the role of black carbon in the climate system A scientific assessment.md"  # 432 chunks (364,384 tokens)
    ]

    for doc_name in problem_docs:
        doc_path = MARKDOWN_DIR / doc_name
        if not doc_path.exists():
            print(f"   Document {doc_name} non trouve, skip")
            continue

        print(f"\nTest: {doc_name}")
        print("-" * 40)

        # Lire le contenu
        try:
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"   ERREUR lecture: {e}")
            continue

        # Chunking
        chunks = chunker.chunk(content)
        chunk_texts = [chunk.text for chunk in chunks]
        total_tokens = sum(chunk.token_count for chunk in chunks)

        print(f"   Chunks: {len(chunks)}")
        print(f"   Tokens: {total_tokens:,}")

        # Tester le traitement avec notre nouvelle fonction
        try:
            embeddings = process_embeddings_with_limit_check(
                voyage_client,
                chunk_texts,
                "voyage-context-3",
                chunks
            )

            print(f"   OK Succes: {len(embeddings)} embeddings generes")

            # Ajouter à ChromaDB pour test
            if embeddings:
                client = chromadb.PersistentClient(path=str(Path(__file__).parent / "chroma_db_fresh_v2"))
                collection = client.get_or_create_collection(
                    name="zotero_research_context_v2",
                    metadata={"hnsw:space": "cosine", "hnsw:construction_ef": 400, "hnsw:M": 64}
                )

                ids = [f"test_{doc_name}_chunk_{i:03d}" for i in range(len(chunks))]
                metadatas = [{
                    "filename": doc_name,
                    "chunk_index": i,
                    "token_count": chunk.token_count,
                    "test": True
                } for i, chunk in enumerate(chunks)]

                collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=chunk_texts,
                    metadatas=metadatas
                )
                print(f"   OK Ajoute a ChromaDB: {len(embeddings)} chunks")

        except Exception as e:
            print(f"   ERREUR traitement: {e}")

    print(f"\n" + "=" * 50)
    print("Test des gros documents termine")

if __name__ == "__main__":
    test_gros_documents()