#!/usr/bin/env python3
"""
Script pour corriger le problème HNSW dans Chroma DB
Supprime complètement la base corrompue et en crée une nouvelle
"""

import sys
import shutil
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from indexing_config import CHROMA_DB_PATH, COLLECTION_NAME, COLLECTION_METADATA

print("=" * 70)
print("CORRECTION DU PROBLEME HNSW DANS CHROMA")
print("=" * 70)
print()

# 1. Arrêter tous les processus Python récents qui pourraient utiliser la DB
print("1. Arret des processus Python recents...")
try:
    import subprocess
    # Tuer les processus Python qui ont démarré récemment
    result = subprocess.run(
        ["powershell", "-Command", 
         "Get-Process python -ErrorAction SilentlyContinue | "
         "Where-Object {$_.StartTime -gt (Get-Date).AddMinutes(-60)} | "
         "Stop-Process -Force -ErrorAction SilentlyContinue"],
        capture_output=True,
        text=True
    )
    print("   [OK] Processus arretes")
except Exception as e:
    print(f"   [ATTENTION] {e}")

time.sleep(2)

# 2. Supprimer complètement le répertoire corrompu
print(f"\n2. Suppression du repertoire corrompu: {CHROMA_DB_PATH}")
if CHROMA_DB_PATH.exists():
    try:
        # Essayer de supprimer avec shutil
        shutil.rmtree(CHROMA_DB_PATH, ignore_errors=True)
        time.sleep(1)
        
        # Vérifier
        if CHROMA_DB_PATH.exists():
            print("   [ATTENTION] Le repertoire existe encore, tentative forcee...")
            # Essayer avec PowerShell
            import subprocess
            subprocess.run(
                ["powershell", "-Command", f"Remove-Item -Recurse -Force '{CHROMA_DB_PATH}' -ErrorAction SilentlyContinue"],
                capture_output=True
            )
            time.sleep(1)
        
        if CHROMA_DB_PATH.exists():
            print("   [ERREUR] Impossible de supprimer le repertoire")
            print("   Veuillez le supprimer manuellement et relancer ce script")
            sys.exit(1)
        else:
            print("   [OK] Repertoire supprime")
    except Exception as e:
        print(f"   [ERREUR] {e}")
        sys.exit(1)
else:
    print("   [OK] Repertoire n'existe pas deja")

# 3. Créer une nouvelle base de données propre
print(f"\n3. Creation d'une nouvelle base de donnees propre...")
try:
    import chromadb
    
    # Créer le client avec une nouvelle base
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    print("   [OK] Client Chroma cree")
    
    # Vérifier qu'il n'y a pas de collections
    collections = client.list_collections()
    if collections:
        print(f"   [ATTENTION] {len(collections)} collections existantes, suppression...")
        for col in collections:
            try:
                client.delete_collection(name=col.name)
            except:
                pass
    
    # Créer la collection avec la bonne configuration HNSW
    print(f"\n4. Creation de la collection '{COLLECTION_NAME}' avec configuration HNSW optimisee...")
    
    # Configuration HNSW plus robuste pour éviter la corruption
    safe_metadata = {
        "hnsw:space": "cosine",
        "hnsw:construction_ef": 200,  # Réduit pour éviter les problèmes
        "hnsw:M": 16,  # Réduit pour plus de stabilité
        "hnsw:search_ef": 100
    }
    
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata=safe_metadata
    )
    print("   [OK] Collection creee avec succes")
    
    # Vérification
    test_collections = client.list_collections()
    print(f"\n5. Verification:")
    print(f"   Collections: {len(test_collections)}")
    print(f"   Collection '{COLLECTION_NAME}': {collection.count()} documents")
    
    print("\n" + "=" * 70)
    print("[SUCCES] Base de donnees recreee avec succes!")
    print("=" * 70)
    print("\nVous pouvez maintenant relancer l'indexation:")
    print("  python scripts/index_incremental.py")
    print("=" * 70)
    
except Exception as e:
    print(f"\n[ERREUR] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

