#!/usr/bin/env python3
"""
Script pour vérifier l'état de l'indexation en cours
"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from indexing_config import CHROMA_DB_PATH, COLLECTION_NAME

import chromadb

try:
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    collections = client.list_collections()
    
    if COLLECTION_NAME in [c.name for c in collections]:
        collection = client.get_collection(name=COLLECTION_NAME)
        count = collection.count()
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Documents indexes: {count}")
        
        if count > 0:
            # Afficher quelques statistiques
            sample = collection.get(limit=100, include=["metadatas"])
            sources = set()
            for metadata in sample['metadatas']:
                if 'source' in metadata:
                    sources.add(metadata['source'])
            
            print(f"  Sources uniques: {len(sources)}")
            print(f"  Derniere mise a jour: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Collection '{COLLECTION_NAME}' n'existe pas encore")
        print("  L'indexation est en cours ou n'a pas encore commence...")
        
except Exception as e:
    print(f"[ERREUR] {e}")


