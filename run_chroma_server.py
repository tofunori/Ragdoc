#!/usr/bin/env python3
"""
Script de lancement du serveur ChromaDB pour RAGDOC
Lance un serveur ChromaDB sur localhost:8000
"""

import sys
import os
from pathlib import Path

# Configuration
PROJECT_DIR = Path(__file__).parent
CHROMA_DB_PATH = PROJECT_DIR / "chroma_db_new"
HOST = "localhost"
PORT = 8000

# Configurer les variables d'environnement pour ChromaDB
os.environ["CHROMA_SERVER_HOST"] = HOST
os.environ["CHROMA_SERVER_HTTP_PORT"] = str(PORT)
os.environ["PERSIST_DIRECTORY"] = str(CHROMA_DB_PATH)
os.environ["IS_PERSISTENT"] = "TRUE"
os.environ["CHROMA_SERVER_CORS_ALLOW_ORIGINS"] = '["*"]'

print(f"ChromaDB Server Configuration:")
print(f"  Host: {HOST}")
print(f"  Port: {PORT}")
print(f"  Database: {CHROMA_DB_PATH}")
print(f"=" * 70)

try:
    import uvicorn
    import chromadb

    # Vérifier la version de ChromaDB
    print(f"ChromaDB version: {chromadb.__version__}")

    # Lancer le serveur
    uvicorn.run(
        "chromadb.app:app",
        host=HOST,
        port=PORT,
        log_level="info",
        access_log=True
    )

except ImportError as e:
    print(f"Erreur: Dépendances manquantes")
    print(f"  {e}")
    print()
    print("Installez avec:")
    print("  pip install chromadb uvicorn")
    sys.exit(1)

except Exception as e:
    print(f"Erreur lors du démarrage du serveur: {e}")
    sys.exit(1)
