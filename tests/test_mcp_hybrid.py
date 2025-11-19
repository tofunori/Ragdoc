#!/usr/bin/env python3
"""
Test du serveur MCP ragdoc-hybrid
Vérifie que le hybrid search fonctionne
"""

import sys
import json
import os
from pathlib import Path

# Force Hybrid mode for testing
os.environ["RAGDOC_MODE"] = "hybrid"

# Import du serveur
sys.path.insert(0, str(Path(__file__).parent / "src"))
import server
from src.config import COLLECTION_HYBRID_NAME

def test_indexation_status():
    """Test 1: Verifier le statut de l'indexation"""
    print("=" * 70)
    print("TEST 1: Indexation Status")
    print("=" * 70)

    # Initialiser les clients
    server.init_clients()

    # Récupérer le statut via la fonction interne
    # Note: server.COLLECTION_NAME will be set based on RAGDOC_MODE
    collection = server.chroma_client.get_collection(name=COLLECTION_HYBRID_NAME)
    all_docs = collection.get(include=["metadatas"])

    total_chunks = len(all_docs['ids'])

    print(f"Total chunks indexes: {total_chunks}")
    print(f"Collection: {COLLECTION_HYBRID_NAME}")
    print("Mode: HYBRID (BM25 + Semantic + Reranking)")

    assert total_chunks > 0, "ERREUR: Aucun document indexe!"
    print("\nOK: Base de donnees prete\n")


def test_hybrid_search():
    """Test 2: Recherche hybrid avec BM25 + Semantic"""
    print("=" * 70)
    print("TEST 2: Hybrid Search - BC concentration")
    print("=" * 70)

    query = "BC concentration measurements"

    # Appeler la fonction interne de recherche hybrid
    result = server._perform_search_hybrid(query, top_k=3, alpha=0.7)
    print(result[:500])  # Afficher les premiers 500 caractères

    # Verifier que les rankings BM25 et Semantic sont presents
    assert "BM25 #" in result, "ERREUR: Rankings BM25 absents!"
    assert "Semantic #" in result, "ERREUR: Rankings Semantic absents!"
    assert "Hybrid:" in result, "ERREUR: Score Hybrid absent!"

    print("\n\nOK: RECHERCHE HYBRID FONCTIONNE avec rankings BM25 et Semantic\n")


if __name__ == "__main__":
    try:
        test_indexation_status()
        test_hybrid_search()

        print("=" * 70)
        print("TOUS LES TESTS REUSSIS!")
        print("=" * 70)
        print("\nLe serveur MCP ragdoc est pret avec:")
        print("  - Hybrid Search (BM25 + Semantic)")
        print("  - Voyage-3-Large embeddings")
        print("  - Cohere v3.5 reranking")
        print("  - Reciprocal Rank Fusion")
        print("\nRedemarrez Claude Desktop pour utiliser le MCP!")

    except Exception as e:
        print(f"\nERREUR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
