#!/usr/bin/env python3
"""
Script pour réinitialiser complètement la base de données Chroma
Supprime toutes les collections et crée une base vide
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from indexing_config import CHROMA_DB_PATH, COLLECTION_NAME

import chromadb

print("=" * 70)
print("REINITIALISATION DE LA BASE DE DONNEES CHROMA")
print("=" * 70)
print()

print(f"Chemin: {CHROMA_DB_PATH}")
print()

# Connexion
try:
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    print("[OK] Connexion etablie")
except Exception as e:
    print(f"[ERREUR] Connexion: {e}")
    sys.exit(1)

# Lister et supprimer toutes les collections
try:
    collections = client.list_collections()
    print(f"Collections trouvees: {len(collections)}")
    
    for col in collections:
        print(f"  - Suppression de '{col.name}' ({col.count()} documents)...")
        client.delete_collection(name=col.name)
        print(f"    [OK] Supprimee")
    
    print()
    
    # Vérification
    remaining = client.list_collections()
    if remaining:
        print(f"[ATTENTION] {len(remaining)} collections restantes")
    else:
        print("[SUCCES] Toutes les collections ont ete supprimees")
        print("La base de donnees est maintenant vide et prete pour une nouvelle indexation.")
        
except Exception as e:
    print(f"[ERREUR] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 70)
print("Pour reindexer, executez:")
print("  python scripts/index_incremental.py")
print("=" * 70)


