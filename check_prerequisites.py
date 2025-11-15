#!/usr/bin/env python3
"""
Vérification des prérequis pour le hybrid search
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

print("=" * 80)
print("VÉRIFICATION DES PRÉREQUIS")
print("=" * 80)

load_dotenv()

all_ok = True

# 1. Vérifier les API keys
print("\n[1/4] API Keys...")
voyage_key = os.getenv("VOYAGE_API_KEY")
if voyage_key:
    print(f"  ✅ VOYAGE_API_KEY configuré ({voyage_key[:10]}...)")
else:
    print("  ❌ VOYAGE_API_KEY manquant")
    print("     → Ajouter dans .env: VOYAGE_API_KEY=votre_clé")
    all_ok = False

cohere_key = os.getenv("COHERE_API_KEY")
if cohere_key:
    print(f"  ✅ COHERE_API_KEY configuré ({cohere_key[:10]}...)")
else:
    print("  ⚠️  COHERE_API_KEY manquant (optionnel pour reranking)")

# 2. Vérifier les dépendances Python
print("\n[2/4] Dépendances Python...")

deps = {
    "chromadb": "chromadb",
    "voyageai": "voyageai",
    "cohere": "cohere",
    "rank_bm25": "rank-bm25",
    "fastmcp": "fastmcp",
    "chonkie": "chonkie"
}

for module, pip_name in deps.items():
    try:
        __import__(module)
        print(f"  ✅ {pip_name}")
    except ImportError:
        print(f"  ❌ {pip_name} manquant")
        print(f"     → Installer avec: pip install {pip_name}")
        all_ok = False

# 3. Vérifier la base ChromaDB
print("\n[3/4] Base de données ChromaDB...")

sys.path.insert(0, str(Path(__file__).parent / "scripts"))
try:
    from indexing_config import CHROMA_DB_HYBRID_PATH, COLLECTION_HYBRID_NAME

    if CHROMA_DB_HYBRID_PATH.exists():
        print(f"  ✅ ChromaDB existe: {CHROMA_DB_HYBRID_PATH}")

        # Vérifier la collection
        try:
            import chromadb
            client = chromadb.PersistentClient(path=str(CHROMA_DB_HYBRID_PATH))
            collection = client.get_collection(name=COLLECTION_HYBRID_NAME)
            count = collection.count()

            if count > 0:
                print(f"  ✅ Collection '{COLLECTION_HYBRID_NAME}': {count} chunks")
            else:
                print(f"  ⚠️  Collection '{COLLECTION_HYBRID_NAME}' existe mais est VIDE")
                print("     → Indexer vos documents d'abord:")
                print("       python scripts/index_incremental.py")
                all_ok = False

        except Exception as e:
            print(f"  ❌ Erreur lors de la vérification de la collection: {e}")
            print("     → La collection n'existe peut-être pas encore")
            print("     → Créer avec: python index_hybrid_collection.py")
            all_ok = False

    else:
        print(f"  ❌ ChromaDB n'existe pas: {CHROMA_DB_HYBRID_PATH}")
        print("     → Indexer vos documents d'abord:")
        print("       python index_hybrid_collection.py")
        all_ok = False

except Exception as e:
    print(f"  ❌ Erreur: {e}")
    all_ok = False

# 4. Vérifier les documents markdown
print("\n[4/4] Documents source...")
try:
    from indexing_config import MARKDOWN_DIR

    if MARKDOWN_DIR.exists():
        md_files = list(MARKDOWN_DIR.glob("*.md"))
        if md_files:
            print(f"  ✅ {len(md_files)} fichiers markdown trouvés dans {MARKDOWN_DIR}")
        else:
            print(f"  ⚠️  Aucun fichier .md dans {MARKDOWN_DIR}")
            print("     → Ajouter vos documents avant d'indexer")
            all_ok = False
    else:
        print(f"  ❌ Répertoire markdown n'existe pas: {MARKDOWN_DIR}")
        print(f"     → Créer le répertoire: mkdir -p {MARKDOWN_DIR}")
        all_ok = False

except Exception as e:
    print(f"  ❌ Erreur: {e}")
    all_ok = False

# Résumé
print("\n" + "=" * 80)
if all_ok:
    print("✅ TOUS LES PRÉREQUIS SONT OK")
    print("=" * 80)
    print("\n🚀 Vous pouvez installer le hybrid search:")
    print("   ./install_hybrid_search.sh")
    print("\nOu manuellement:")
    print("   python quick_test_hybrid.py")
    print("   python activate_hybrid_search.py")
else:
    print("❌ CERTAINS PRÉREQUIS MANQUENT")
    print("=" * 80)
    print("\n📋 Étapes à suivre:")
    print("   1. Installer les dépendances manquantes (voir ci-dessus)")
    print("   2. Configurer les API keys dans .env")
    print("   3. Indexer vos documents si nécessaire")
    print("   4. Relancer ce script pour vérifier")

print("=" * 80)

sys.exit(0 if all_ok else 1)
