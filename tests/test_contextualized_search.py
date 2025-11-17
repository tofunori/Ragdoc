#!/usr/bin/env python3
"""
Test de la recherche avec Contextualized Embeddings
Compare avec la base hybrid actuelle
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import chromadb
import voyageai
from indexing_config import (
    CHROMA_DB_HYBRID_PATH,
    CHROMA_DB_CONTEXTUALIZED_PATH,
    COLLECTION_HYBRID_NAME,
    COLLECTION_CONTEXTUALIZED_NAME
)


def test_both_databases(query: str, top_k: int = 5):
    """
    Compare résultats entre base hybrid et contextualized
    """

    print("="*70)
    print(f"QUERY: {query}")
    print("="*70)

    vo = voyageai.Client()

    # === BASE HYBRID (actuelle) ===
    print(f"\n[HYBRID] BASE HYBRID ({COLLECTION_HYBRID_NAME})")
    print("-"*70)

    try:
        client_hybrid = chromadb.PersistentClient(path=str(CHROMA_DB_HYBRID_PATH))
        collection_hybrid = client_hybrid.get_collection(name=COLLECTION_HYBRID_NAME)

        # Embedding query (standard)
        query_embd_std = vo.embed([query], model="voyage-3-large", input_type="query").embeddings[0]

        results_hybrid = collection_hybrid.query(
            query_embeddings=[query_embd_std],
            n_results=top_k
        )

        print(f"Chunks trouvés: {len(results_hybrid['documents'][0])}")
        for i, (doc, metadata) in enumerate(zip(results_hybrid['documents'][0], results_hybrid['metadatas'][0])):
            print(f"\n[{i+1}] Source: {metadata.get('source', 'N/A')}")
            print(f"    Chunk: {doc[:150]}...")

    except Exception as e:
        print(f"[ERROR] ERREUR: {e}")

    # === BASE CONTEXTUALIZED (nouvelle) ===
    print(f"\n\n[CONTEXTUALIZED] BASE CONTEXTUALIZED ({COLLECTION_CONTEXTUALIZED_NAME})")
    print("-"*70)

    try:
        client_ctx = chromadb.PersistentClient(path=str(CHROMA_DB_CONTEXTUALIZED_PATH))
        collection_ctx = client_ctx.get_collection(name=COLLECTION_CONTEXTUALIZED_NAME)

        # Embedding query (contextualized)
        query_embd_ctx = vo.contextualized_embed(
            inputs=[[query]],
            model="voyage-context-3",
            input_type="query"
        ).results[0].embeddings[0]

        results_ctx = collection_ctx.query(
            query_embeddings=[query_embd_ctx],
            n_results=top_k
        )

        print(f"Chunks trouvés: {len(results_ctx['documents'][0])}")
        for i, (doc, metadata) in enumerate(zip(results_ctx['documents'][0], results_ctx['metadatas'][0])):
            strategy = metadata.get('embedding_strategy', 'N/A')
            print(f"\n[{i+1}] Source: {metadata.get('source', 'N/A')}")
            print(f"    Stratégie: {strategy}")
            print(f"    Chunk: {doc[:150]}...")

    except Exception as e:
        print(f"[ERROR] ERREUR: {e}")

    print("\n" + "="*70 + "\n")


def check_databases_stats():
    """Affiche stats des deux bases"""

    print("\n[STATS] STATISTIQUES DES BASES DE DONNEES")
    print("="*70)

    # Base Hybrid
    try:
        client_hybrid = chromadb.PersistentClient(path=str(CHROMA_DB_HYBRID_PATH))
        collection_hybrid = client_hybrid.get_collection(name=COLLECTION_HYBRID_NAME)
        count_hybrid = collection_hybrid.count()

        # Récupérer un échantillon pour voir les métadonnées
        sample = collection_hybrid.get(limit=1, include=["metadatas"])

        print(f"\n[HYBRID] BASE HYBRID")
        print(f"   Chemin: {CHROMA_DB_HYBRID_PATH}")
        print(f"   Collection: {COLLECTION_HYBRID_NAME}")
        print(f"   Chunks: {count_hybrid:,}")
        if sample['metadatas']:
            print(f"   Exemple metadonnees: {sample['metadatas'][0]}")

    except Exception as e:
        print(f"\n[HYBRID] BASE HYBRID: [ERROR] {e}")

    # Base Contextualized
    try:
        client_ctx = chromadb.PersistentClient(path=str(CHROMA_DB_CONTEXTUALIZED_PATH))
        collection_ctx = client_ctx.get_collection(name=COLLECTION_CONTEXTUALIZED_NAME)
        count_ctx = collection_ctx.count()

        # Stats par stratégie
        all_data = collection_ctx.get(include=["metadatas"])
        strategies = {}
        for meta in all_data['metadatas']:
            strategy = meta.get('embedding_strategy', 'unknown')
            strategies[strategy] = strategies.get(strategy, 0) + 1

        print(f"\n[CONTEXTUALIZED] BASE CONTEXTUALIZED")
        print(f"   Chemin: {CHROMA_DB_CONTEXTUALIZED_PATH}")
        print(f"   Collection: {COLLECTION_CONTEXTUALIZED_NAME}")
        print(f"   Chunks: {count_ctx:,}")
        print(f"   Strategies utilisees:")
        for strategy, count in sorted(strategies.items()):
            pct = (count / count_ctx) * 100
            print(f"     - {strategy}: {count:,} chunks ({pct:.1f}%)")

    except Exception as e:
        print(f"\n[CONTEXTUALIZED] BASE CONTEXTUALIZED: [ERROR] {e}")

    print("\n" + "="*70)


if __name__ == "__main__":
    # Vérifier stats
    check_databases_stats()

    # Test queries
    test_queries = [
        "What is black carbon impact on glacier albedo?",
        "MODIS satellite albedo measurements accuracy",
        "BC concentration measurements in Arctic glaciers"
    ]

    for query in test_queries:
        test_both_databases(query, top_k=3)
        print("\n" + "="*70 + "\n")
