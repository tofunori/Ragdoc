#!/usr/bin/env python3
"""
Script de test pour diagnostiquer les problèmes de la base de données Chroma
"""

import sys
from pathlib import Path

# Ajouter le chemin du projet
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import CHROMA_DB_PATH, COLLECTION_NAME

import chromadb
from dotenv import load_dotenv
import os

load_dotenv()

print("=" * 70)
print("TEST DE LA BASE DE DONNÉES CHROMA")
print("=" * 70)
print()

# 1. Vérifier le chemin
print(f"1. Chemin de la base de données: {CHROMA_DB_PATH}")
print(f"   Existe: {CHROMA_DB_PATH.exists()}")
print()

# 2. Connexion à Chroma
try:
    print("2. Connexion à Chroma DB...")
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    print("   [OK] Connexion reussie")
    print()
except Exception as e:
    print(f"   [ERREUR] {e}")
    sys.exit(1)

# 3. Lister les collections
try:
    print("3. Collections disponibles:")
    collections = client.list_collections()
    if not collections:
        print("   [WARNING] Aucune collection trouvee!")
    else:
        for col in collections:
            print(f"   - {col.name}")
    print()
except Exception as e:
    print(f"   [ERREUR] {e}")
    print()

# 4. Vérifier la collection principale
try:
    print(f"4. Vérification de la collection '{COLLECTION_NAME}':")
    collection = client.get_collection(name=COLLECTION_NAME)
    count = collection.count()
    print(f"   [OK] Collection trouvee")
    print(f"   Nombre de documents: {count}")
    print()
    
    if count == 0:
        print("   [WARNING] La collection est vide!")
    else:
        # Récupérer quelques exemples
        sample = collection.get(limit=5, include=["metadatas", "documents"])
        print(f"   Échantillon (5 premiers):")
        for i, (doc_id, metadata, doc) in enumerate(zip(
            sample['ids'][:5],
            sample['metadatas'][:5],
            sample['documents'][:5]
        ), 1):
            print(f"   [{i}] ID: {doc_id[:30]}...")
            print(f"       Source: {metadata.get('source', 'N/A')}")
            print(f"       Chunk: {metadata.get('chunk_index', 'N/A')}/{metadata.get('total_chunks', 'N/A')}")
            print(f"       Doc length: {len(doc)} chars")
            print()
        
        # Vérifier les métadonnées
        all_docs = collection.get(limit=100, include=["metadatas"])
        sources = set()
        models = set()
        has_hash = 0
        has_date = 0
        
        for metadata in all_docs['metadatas']:
            if 'source' in metadata:
                sources.add(metadata['source'])
            if 'model' in metadata and metadata['model']:
                models.add(metadata['model'])
            if 'doc_hash' in metadata and metadata['doc_hash']:
                has_hash += 1
            if 'indexed_date' in metadata and metadata['indexed_date']:
                has_date += 1
        
        print(f"   Statistiques des métadonnées:")
        print(f"   - Sources uniques: {len(sources)}")
        print(f"   - Modèles utilisés: {', '.join(models) if models else 'Aucun'}")
        print(f"   - Documents avec hash: {has_hash}/{len(all_docs['ids'])}")
        print(f"   - Documents avec date: {has_date}/{len(all_docs['ids'])}")
        print()
        
except Exception as e:
    print(f"   ✗ ERREUR: {e}")
    import traceback
    traceback.print_exc()
    print()

# 5. Tester une recherche simple
try:
    print("5. Test de recherche simple:")
    collection = client.get_collection(name=COLLECTION_NAME)
    
    # Récupérer un embedding d'exemple
    sample = collection.get(limit=1, include=["embeddings"])
    if sample['embeddings'] and len(sample['embeddings'][0]) > 0:
        test_embedding = sample['embeddings'][0]
        results = collection.query(
            query_embeddings=[test_embedding],
            n_results=3
        )
        print(f"   [OK] Recherche reussie")
        print(f"   Resultats trouves: {len(results['documents'][0])}")
    else:
        print("   [WARNING] Aucun embedding trouve pour tester")
    print()
except Exception as e:
    print(f"   [ERREUR] {e}")
    import traceback
    traceback.print_exc()
    print()

# 6. Vérifier les autres bases de données disponibles
print("6. Bases de données disponibles dans le projet:")
base_dir = Path(__file__).parent
for db_dir in base_dir.glob("chroma_db*"):
    if db_dir.is_dir():
        print(f"   - {db_dir.name}")
        try:
            test_client = chromadb.PersistentClient(path=str(db_dir))
            test_collections = test_client.list_collections()
            if test_collections:
                for col in test_collections:
                    col_obj = test_client.get_collection(name=col.name)
                    print(f"     → Collection '{col.name}': {col_obj.count()} documents")
        except Exception as e:
            print(f"     → Erreur: {e}")
print()

print("=" * 70)
print("FIN DES TESTS")
print("=" * 70)

