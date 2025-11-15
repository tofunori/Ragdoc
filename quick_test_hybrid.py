#!/usr/bin/env python3
"""
Test rapide du hybrid search
Vérifie que BM25 + Semantic fonctionne
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

print("=" * 80)
print("TEST RAPIDE DU HYBRID SEARCH")
print("=" * 80)

# 1. Vérifier les dépendances
print("\n[1/5] Vérification des dépendances...")
try:
    from rank_bm25 import BM25Okapi
    print("  ✅ rank-bm25 installé")
except ImportError as e:
    print(f"  ❌ rank-bm25 manquant: {e}")
    print("  → Installer avec: pip install rank-bm25")
    sys.exit(1)

try:
    import chromadb
    print("  ✅ chromadb installé")
except ImportError:
    print("  ❌ chromadb manquant")
    sys.exit(1)

try:
    import voyageai
    print("  ✅ voyageai installé")
except ImportError:
    print("  ❌ voyageai manquant")
    sys.exit(1)

# 2. Vérifier configuration
print("\n[2/5] Vérification de la configuration...")
load_dotenv()

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
if not VOYAGE_API_KEY:
    print("  ❌ VOYAGE_API_KEY non trouvé dans .env")
    sys.exit(1)
print("  ✅ VOYAGE_API_KEY configuré")

# 3. Charger ChromaDB
print("\n[3/5] Connexion à ChromaDB...")
from indexing_config import COLLECTION_HYBRID_NAME, CHROMA_DB_HYBRID_PATH

try:
    client = chromadb.PersistentClient(path=str(CHROMA_DB_HYBRID_PATH))
    collection = client.get_collection(name=COLLECTION_HYBRID_NAME)
    count = collection.count()
    print(f"  ✅ Collection '{COLLECTION_HYBRID_NAME}' chargée")
    print(f"  ℹ️  {count} chunks indexés")

    if count == 0:
        print("  ⚠️  La collection est vide. Indexez des documents d'abord.")
        sys.exit(1)
except Exception as e:
    print(f"  ❌ Erreur: {e}")
    sys.exit(1)

# 4. Initialiser le hybrid retriever
print("\n[4/5] Initialisation du Hybrid Retriever...")
try:
    from hybrid_retriever import HybridRetriever

    # Embedding function
    voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)

    def voyage_embed(texts):
        result = voyage_client.embed(texts=texts, model="voyage-context-3")
        return result.embeddings

    retriever = HybridRetriever(
        collection=collection,
        embedding_function=voyage_embed
    )
    print(f"  ✅ Hybrid Retriever initialisé")
    print(f"  ℹ️  BM25 index construit sur {len(retriever.docs)} documents")

except Exception as e:
    print(f"  ❌ Erreur: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 5. Test de recherche
print("\n[5/5] Test de recherche...")
test_query = "black carbon glacier albedo"
print(f"  Query: '{test_query}'")

try:
    # Hybrid search
    results = retriever.search(
        query=test_query,
        top_k=3,
        alpha=0.7  # 70% semantic, 30% BM25
    )

    print(f"\n  ✅ Hybrid search réussie! {len(results)} résultats:")
    print("  " + "-" * 76)

    for i, result in enumerate(results, 1):
        source = result['metadata'].get('source', 'unknown')
        chunk_idx = result['metadata'].get('chunk_index', 0)

        print(f"\n  [{i}] Score: {result['score']:.4f}")
        print(f"      BM25: {result.get('bm25_score', 0):.3f} (rank #{result.get('bm25_rank', 'N/A')})")
        print(f"      Semantic: {result.get('semantic_score', 0):.3f} (rank #{result.get('semantic_rank', 'N/A')})")
        print(f"      Source: {source} (chunk {chunk_idx})")
        print(f"      Preview: {result['text'][:120]}...")

except Exception as e:
    print(f"  ❌ Erreur lors de la recherche: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Success!
print("\n" + "=" * 80)
print("✅ TEST RÉUSSI - Le hybrid search fonctionne parfaitement!")
print("=" * 80)
print("\nProchaine étape: Activer en production avec:")
print("  python activate_hybrid_search.py")
print("=" * 80)
