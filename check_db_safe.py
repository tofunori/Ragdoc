#!/usr/bin/env python3
"""
Vérification sécurisée de la base de données
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from indexing_config import CHROMA_DB_PATH, COLLECTION_NAME

try:
    import chromadb
    
    # Essayer une connexion simple
    print("Connexion à Chroma DB...")
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    
    print("Liste des collections...")
    collections = client.list_collections()
    print(f"Collections trouvées: {len(collections)}")
    
    if COLLECTION_NAME in [c.name for c in collections]:
        print(f"Tentative de récupération de la collection '{COLLECTION_NAME}'...")
        try:
            collection = client.get_collection(name=COLLECTION_NAME)
            count = collection.count()
            print(f"[SUCCES] Documents indexés: {count}")
            
            # Essayer de récupérer quelques métadonnées
            try:
                sample = collection.get(limit=5, include=["metadatas"])
                print(f"Échantillon récupéré: {len(sample['ids'])} documents")
            except Exception as e:
                print(f"[ATTENTION] Impossible de lire les métadonnées: {e}")
                
        except Exception as e:
            print(f"[ERREUR] Impossible d'accéder à la collection: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Collection '{COLLECTION_NAME}' non trouvée")
        print(f"Collections disponibles: {[c.name for c in collections]}")
        
except Exception as e:
    print(f"[ERREUR] {e}")
    import traceback
    traceback.print_exc()


