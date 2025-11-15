#!/usr/bin/env python3
"""
Script pour supprimer proprement toutes les bases de données Chroma
"""

import shutil
from pathlib import Path
import time

base_dir = Path(__file__).parent

# Trouver tous les répertoires chroma_db*
chroma_dirs = [d for d in base_dir.glob("chroma_db*") if d.is_dir()]

if not chroma_dirs:
    print("Aucun repertoire chroma_db trouve.")
else:
    print(f"Repertoires trouves: {len(chroma_dirs)}")
    for chroma_dir in chroma_dirs:
        print(f"  - {chroma_dir.name}")
    
    print("\nSuppression en cours...")
    
    for chroma_dir in chroma_dirs:
        try:
            # Essayer de supprimer
            print(f"Suppression de {chroma_dir.name}...")
            shutil.rmtree(chroma_dir, ignore_errors=True)
            time.sleep(0.5)
            
            # Vérifier
            if chroma_dir.exists():
                print(f"  [ATTENTION] {chroma_dir.name} existe encore - peut-etre verrouille")
            else:
                print(f"  [OK] {chroma_dir.name} supprime")
        except Exception as e:
            print(f"  [ERREUR] {chroma_dir.name}: {e}")

# Vérification finale
remaining = [d for d in base_dir.glob("chroma_db*") if d.is_dir()]
if remaining:
    print(f"\n[ATTENTION] {len(remaining)} repertoires restants:")
    for r in remaining:
        print(f"  - {r.name}")
    print("\nCes repertoires peuvent etre verrouilles par un processus.")
    print("Fermez tous les processus Python/Chroma et reessayez.")
else:
    print("\n[SUCCES] Tous les repertoires chroma_db ont ete supprimes!")


