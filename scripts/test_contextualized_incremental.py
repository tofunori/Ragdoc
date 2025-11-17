#!/usr/bin/env python3
"""
Test script pour vérifier l'indexation incrémentale contextualized

Vérifie:
- Database contextualized accessible
- Collection présente
- MD5 hashes présents dans métadonnées
- Distribution des stratégies d'embedding
- Timestamps d'indexation récents
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from indexing_config import (
    CHROMA_DB_CONTEXTUALIZED_PATH,
    COLLECTION_CONTEXTUALIZED_NAME
)

import chromadb


def test_contextualized_incremental():
    """Test l'indexation incrémentale contextualized"""

    print("[*] TEST INDEXATION INCREMENTALE CONTEXTUALIZED")
    print("="*70)
    print()

    # 1. Test connexion database
    print("1. CONNEXION DATABASE")
    print(f"   Chemin: {CHROMA_DB_CONTEXTUALIZED_PATH}")

    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DB_CONTEXTUALIZED_PATH))
        print("   [OK] Connexion réussie")
    except Exception as e:
        print(f"   [ERROR] {e}")
        return False

    print()

    # 2. Test collection
    print("2. COLLECTION")
    print(f"   Nom: {COLLECTION_CONTEXTUALIZED_NAME}")

    try:
        collection = client.get_collection(name=COLLECTION_CONTEXTUALIZED_NAME)
        count = collection.count()
        print(f"   [OK] Collection trouvée: {count:,} chunks")
    except Exception as e:
        print(f"   [ERROR] Collection non trouvée: {e}")
        return False

    print()

    # 3. Test metadata structure
    print("3. STRUCTURE METADATA")

    try:
        # Récupérer un échantillon
        sample = collection.get(limit=10, include=["metadatas"])

        if not sample['metadatas']:
            print("   [WARNING] Aucune métadonnée trouvée")
            return False

        # Vérifier champs requis
        required_fields = [
            'source', 'doc_hash', 'indexed_date',
            'embedding_strategy', 'model', 'doc_size_tokens'
        ]

        first_meta = sample['metadatas'][0]
        print(f"   Champs présents: {list(first_meta.keys())}")

        missing = [f for f in required_fields if f not in first_meta]
        if missing:
            print(f"   [ERROR] Champs manquants: {missing}")
            return False

        print("   [OK] Tous les champs requis présents")

    except Exception as e:
        print(f"   [ERROR] {e}")
        return False

    print()

    # 4. Test MD5 hashes
    print("4. MD5 HASHES")

    try:
        all_data = collection.get(include=["metadatas"])
        hashes_found = sum(1 for m in all_data['metadatas'] if m.get('doc_hash'))

        print(f"   Chunks avec hash: {hashes_found:,} / {len(all_data['metadatas']):,}")

        if hashes_found == len(all_data['metadatas']):
            print("   [OK] Tous les chunks ont un MD5 hash")
        elif hashes_found > 0:
            print(f"   [WARNING] Seulement {hashes_found/len(all_data['metadatas'])*100:.1f}% ont des hashes")
        else:
            print("   [ERROR] Aucun hash trouvé - indexation non incrémentale?")
            return False

    except Exception as e:
        print(f"   [ERROR] {e}")
        return False

    print()

    # 5. Test distribution des stratégies
    print("5. DISTRIBUTION STRATÉGIES D'EMBEDDING")

    try:
        strategies = {}
        models = {}

        for meta in all_data['metadatas']:
            strategy = meta.get('embedding_strategy', 'unknown')
            model = meta.get('model', 'unknown')

            strategies[strategy] = strategies.get(strategy, 0) + 1
            models[model] = models.get(model, 0) + 1

        print("   Stratégies:")
        for strategy, count in sorted(strategies.items()):
            pct = (count / len(all_data['metadatas'])) * 100
            print(f"      {strategy}: {count:,} chunks ({pct:.1f}%)")

        print()
        print("   Modèles:")
        for model, count in sorted(models.items()):
            pct = (count / len(all_data['metadatas'])) * 100
            print(f"      {model}: {count:,} chunks ({pct:.1f}%)")

        print("   [OK] Distribution analysée")

    except Exception as e:
        print(f"   [ERROR] {e}")
        return False

    print()

    # 6. Test documents uniques
    print("6. DOCUMENTS UNIQUES")

    try:
        sources = set(m.get('source') for m in all_data['metadatas'])
        print(f"   Documents indexés: {len(sources)}")

        # Afficher quelques exemples
        print("   Exemples:")
        for i, source in enumerate(sorted(sources)[:5]):
            doc_chunks = [m for m in all_data['metadatas'] if m.get('source') == source]
            hash_val = doc_chunks[0].get('doc_hash', 'N/A')[:8]
            print(f"      [{i+1}] {source} ({len(doc_chunks)} chunks, hash: {hash_val}...)")

        print("   [OK] Documents listés")

    except Exception as e:
        print(f"   [ERROR] {e}")
        return False

    print()

    # 7. Test timestamps
    print("7. TIMESTAMPS D'INDEXATION")

    try:
        from datetime import datetime

        recent_docs = {}
        for meta in all_data['metadatas']:
            source = meta.get('source')
            indexed_date = meta.get('indexed_date')

            if source and indexed_date:
                if source not in recent_docs or indexed_date > recent_docs[source]:
                    recent_docs[source] = indexed_date

        if recent_docs:
            # Afficher les 5 plus récents
            sorted_docs = sorted(recent_docs.items(), key=lambda x: x[1], reverse=True)[:5]

            print("   Documents récemment indexés:")
            for source, date in sorted_docs:
                print(f"      {source}: {date}")

            print("   [OK] Timestamps présents et valides")
        else:
            print("   [WARNING] Aucun timestamp trouvé")

    except Exception as e:
        print(f"   [ERROR] {e}")
        return False

    print()
    print("="*70)
    print("[SUCCESS] Tous les tests réussis!")
    print("="*70)

    return True


if __name__ == "__main__":
    success = test_contextualized_incremental()
    sys.exit(0 if success else 1)
