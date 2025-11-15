#!/usr/bin/env python3
"""
Script pour monitorer l'indexation en cours avec mises à jour périodiques
"""

import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from indexing_config import CHROMA_DB_PATH, COLLECTION_NAME

import chromadb

print("=" * 70)
print("MONITORING DE L'INDEXATION")
print("=" * 70)
print("Appuyez sur Ctrl+C pour arreter\n")

previous_count = 0
check_count = 0

try:
    while True:
        check_count += 1
        try:
            client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
            collections = client.list_collections()
            
            if COLLECTION_NAME in [c.name for c in collections]:
                collection = client.get_collection(name=COLLECTION_NAME)
                count = collection.count()
                
                timestamp = datetime.now().strftime('%H:%M:%S')
                
                if count > previous_count:
                    diff = count - previous_count
                    print(f"[{timestamp}] ✓ {count} documents (+{diff} depuis la derniere verification)")
                    previous_count = count
                else:
                    print(f"[{timestamp}] {count} documents (pas de changement)")
                
                # Afficher des stats toutes les 5 vérifications
                if check_count % 5 == 0:
                    sample = collection.get(limit=min(500, count), include=["metadatas"])
                    sources = set()
                    for metadata in sample['metadatas']:
                        if 'source' in metadata:
                            sources.add(metadata['source'])
                    print(f"  → Sources uniques: {len(sources)}")
                    print()
            else:
                timestamp = datetime.now().strftime('%H:%M:%S')
                print(f"[{timestamp}] Collection '{COLLECTION_NAME}' n'existe pas encore...")
                
        except Exception as e:
            timestamp = datetime.now().strftime('%H:%M:%S')
            print(f"[{timestamp}] [ERREUR] {e}")
        
        time.sleep(10)  # Vérifier toutes les 10 secondes
        
except KeyboardInterrupt:
    print("\n" + "=" * 70)
    print("Monitoring arrete")
    print("=" * 70)


