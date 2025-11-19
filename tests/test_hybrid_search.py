#!/usr/bin/env python3
"""
Test du système hybrid search (BM25 + Semantic)
Comparaison: Semantic seul vs Hybrid
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from hybrid_retriever import HybridRetriever
from indexing_config import COLLECTION_HYBRID_NAME, CHROMA_DB_HYBRID_PATH

import chromadb
import voyageai

load_dotenv()

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")


def compare_semantic_vs_hybrid():
    """
    Compare semantic-only vs hybrid search on test queries

    Test queries:
    1. Exact term query (BM25 should help)
    2. Conceptual query (Semantic should dominate)
    3. Mixed query (both should contribute)
    """

    print("=" * 80)
    print("SEMANTIC vs HYBRID SEARCH COMPARISON")
    print("=" * 80)

    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=str(CHROMA_DB_HYBRID_PATH))
    collection = client.get_collection(name=COLLECTION_HYBRID_NAME)

    # Initialize Voyage
    voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)

    def voyage_embed(texts):
        result = voyage_client.embed(texts=texts, model="voyage-context-3")
        return result.embeddings

    # Initialize hybrid retriever
    print("\nInitializing hybrid retriever...")
    retriever = HybridRetriever(collection, embedding_function=voyage_embed)
    print("OK")

    # Test queries
    test_queries = [
        {
            'query': 'albedo measurement methods',
            'type': 'exact_terms',
            'description': 'Exact technical terms (should favor BM25)'
        },
        {
            'query': 'How does ice darkening affect climate?',
            'type': 'conceptual',
            'description': 'Conceptual question (should favor semantic)'
        },
        {
            'query': 'black carbon concentration measurements on glaciers',
            'type': 'mixed',
            'description': 'Mixed: exact terms + concepts'
        }
    ]

    for test in test_queries:
        query = test['query']

        print("\n" + "=" * 80)
        print(f"Query: {query}")
        print(f"Type: {test['type']} - {test['description']}")
        print("=" * 80)

        # 1. Semantic-only search
        print("\n[1] SEMANTIC ONLY (ChromaDB vector search)")
        print("-" * 80)

        query_embedding = voyage_embed([query])[0]
        semantic_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            include=["documents", "metadatas", "distances"]
        )

        for i, (doc, meta, dist) in enumerate(zip(
            semantic_results['documents'][0],
            semantic_results['metadatas'][0],
            semantic_results['distances'][0]
        ), 1):
            similarity = 1 - dist
            source = meta.get('source', meta.get('filename', 'unknown'))
            chunk_idx = meta.get('chunk_index', 0)

            print(f"  [{i}] Score: {similarity:.4f}")
            print(f"      Source: {source} (chunk {chunk_idx})")
            print(f"      Text: {doc[:150]}...\n")

        # 2. Hybrid search (BM25 + Semantic)
        print("\n[2] HYBRID (BM25 + Semantic with RRF)")
        print("-" * 80)

        hybrid_results = retriever.search(
            query=query,
            top_k=5,
            alpha=0.7,  # 70% semantic, 30% BM25
            bm25_top_n=50,
            semantic_top_n=50
        )

        for i, result in enumerate(hybrid_results, 1):
            source = result['metadata'].get('source', result['metadata'].get('filename', 'unknown'))
            chunk_idx = result['metadata'].get('chunk_index', 0)

            print(f"  [{i}] Hybrid Score: {result['score']:.4f} "
                  f"(BM25: {result.get('bm25_score', 0):.3f}, Semantic: {result.get('semantic_score', 0):.3f})")
            print(f"      Rankings: BM25 #{result.get('bm25_rank')}, Semantic #{result.get('semantic_rank')}")
            print(f"      Source: {source} (chunk {chunk_idx})")
            print(f"      Text: {result['text'][:150]}...\n")

        # 3. Analysis
        print("\n[3] ANALYSIS")
        print("-" * 80)

        # Calculate overlap in top-5
        semantic_ids = set(semantic_results['ids'][0])
        hybrid_ids = set([r['id'] for r in hybrid_results])
        overlap = semantic_ids & hybrid_ids

        print(f"  Overlap in top-5: {len(overlap)}/5 results")
        print(f"  New results from BM25: {len(hybrid_ids - semantic_ids)}")
        print(f"  Lost results: {len(semantic_ids - hybrid_ids)}")

        # Analyze BM25 contribution
        avg_bm25_rank = sum(r.get('bm25_rank', 100) or 100 for r in hybrid_results) / len(hybrid_results)
        avg_sem_rank = sum(r.get('semantic_rank', 100) or 100 for r in hybrid_results) / len(hybrid_results)

        print(f"  Average BM25 rank in top-5: {avg_bm25_rank:.1f}")
        print(f"  Average Semantic rank in top-5: {avg_sem_rank:.1f}")

        if avg_bm25_rank < avg_sem_rank:
            print("  → BM25 is contributing significantly!")
        elif avg_sem_rank < avg_bm25_rank:
            print("  → Semantic is dominating (as expected for conceptual queries)")
        else:
            print("  → Balanced contribution from both methods")


def test_alpha_tuning():
    """
    Test different alpha values (semantic vs BM25 weight)

    alpha = 1.0 → 100% semantic (ignore BM25)
    alpha = 0.5 → 50/50 weight
    alpha = 0.0 → 100% BM25 (ignore semantic)
    """

    print("\n\n" + "=" * 80)
    print("ALPHA TUNING TEST (Semantic vs BM25 weight)")
    print("=" * 80)

    # Initialize
    client = chromadb.PersistentClient(path=str(CHROMA_DB_HYBRID_PATH))
    collection = client.get_collection(name=COLLECTION_HYBRID_NAME)
    voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)

    def voyage_embed(texts):
        result = voyage_client.embed(texts=texts, model="voyage-context-3")
        return result.embeddings

    retriever = HybridRetriever(collection, embedding_function=voyage_embed)

    # Test query (technical terms)
    query = "black carbon BC concentration measurements"

    # Test different alpha values
    alphas = [0.0, 0.3, 0.5, 0.7, 1.0]

    print(f"\nQuery: {query}")
    print("=" * 80)

    for alpha in alphas:
        print(f"\nAlpha = {alpha:.1f} ({int(alpha*100)}% Semantic, {int((1-alpha)*100)}% BM25)")
        print("-" * 80)

        results = retriever.search(query=query, top_k=3, alpha=alpha)

        for i, result in enumerate(results, 1):
            source = result['metadata'].get('source', 'unknown')
            print(f"  [{i}] {source[:40]:40} | Score: {result['score']:.4f}")

    print("\n" + "=" * 80)
    print("RECOMMENDATION:")
    print("  - For technical/scientific queries: alpha = 0.5-0.7 (balanced to semantic-leaning)")
    print("  - For exact term lookup: alpha = 0.3 (BM25-leaning)")
    print("  - For conceptual questions: alpha = 0.8-1.0 (semantic-heavy)")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test hybrid search")
    parser.add_argument("--mode", choices=["compare", "alpha", "both"], default="both",
                       help="Test mode")
    args = parser.parse_args()

    if args.mode in ["compare", "both"]:
        compare_semantic_vs_hybrid()

    if args.mode in ["alpha", "both"]:
        test_alpha_tuning()
