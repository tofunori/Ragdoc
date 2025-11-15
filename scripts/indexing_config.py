#!/usr/bin/env python3
"""
Configuration centralisée pour l'indexation incrémentale Chroma
Paramètres de chunking, modèles, et chemins
"""

from pathlib import Path

# ============================================================================
# CHEMINS
# ============================================================================

MARKDOWN_DIR = Path(__file__).parent.parent / "articles_markdown"
CHROMA_DB_PATH = Path(__file__).parent.parent / "chroma_db_new"

# Nouvelle base de données pour pipeline hybride
CHROMA_DB_HYBRID_PATH = CHROMA_DB_PATH

# ============================================================================
# COLLECTION CHROMA
# ============================================================================

# Collection originale (TokenChunker simple)
COLLECTION_NAME = "zotero_research_context_v2"

# Nouvelle collection hybride (Token + Semantic + Overlap)
COLLECTION_HYBRID_NAME = "zotero_research_context_hybrid_v3"

# Métadonnées de la collection (HNSW optimisation)
COLLECTION_METADATA = {
    "hnsw:space": "cosine",
    "hnsw:construction_ef": 400,
    "hnsw:M": 64
}

# Métadonnées pour collection hybride (mêmes paramètres HNSW)
COLLECTION_HYBRID_METADATA = {
    "hnsw:space": "cosine",
    "hnsw:construction_ef": 400,
    "hnsw:M": 64,
    "pipeline": "hybrid_token_semantic_overlap"
}

# ============================================================================
# PARAMÈTRES DE CHUNKING
# ============================================================================

# Chunking pour documents normaux
CHUNK_SIZE = 2000  # caractères
CHUNK_OVERLAP = 400  # caractères

# Chunking pour documents volumineux
LARGE_DOC_CHUNK_SIZE = 1024  # caractères
LARGE_DOC_CHUNK_OVERLAP = 200  # caractères

# ============================================================================
# PARAMÈTRES CHONKIE (TOKENS)
# ============================================================================

# Configuration Chonkie optimale pour Voyage Context-3
CHONKIE_CHUNK_SIZE = 1024      # Tokens (standard pour RAG, 25% fenêtre Context-3)
CHONKIE_CHUNK_OVERLAP = 180    # Tokens (~17.5% optimal pour Context-3)
CHONKIE_TOKENIZER = "gpt2"     # Compatible avec Voyage AI

# ============================================================================
# CONTEXT WINDOW EXPANSION (Retrieval)
# ============================================================================

# Nombre de chunks adjacents à retourner avant/après le résultat principal
CONTEXT_WINDOW_SIZE = 2  # Retourne [chunk_n-2, chunk_n-1, chunk_n, chunk_n+1, chunk_n+2]

# ============================================================================
# MODÈLES D'EMBEDDING VOYAGE AI
# ============================================================================

# Modèle par défaut pour documents normaux
DEFAULT_MODEL = "voyage-context-3"

# Modèle pour documents volumineux (>50K chars)
LARGE_DOC_MODEL = "voyage-3-large"

# Seuil pour basculer vers le modèle volumineux
LARGE_DOC_THRESHOLD = 50000  # caractères

# ============================================================================
# FEATURES DE DÉDUPLICATION
# ============================================================================

# Calcul et stockage du hash MD5 du contenu
USE_CONTENT_HASH = True

# Suivi de la date d'indexation
TRACK_INDEXED_DATE = True

# Suppression automatique des chunks des documents absents
AUTO_DELETE_MISSING = False

# ============================================================================
# LOGGING
# ============================================================================

# Niveaux de détail: DEBUG, INFO, WARNING
LOG_LEVEL = "INFO"

# ============================================================================
# VALIDATION
# ============================================================================

def validate_paths():
    """Valider que les chemins existent."""
    if not MARKDOWN_DIR.exists():
        raise FileNotFoundError(f"Répertoire markdown non trouvé: {MARKDOWN_DIR}")

    if not CHROMA_DB_PATH.exists():
        print(f"⚠️  Répertoire Chroma sera créé: {CHROMA_DB_PATH}")


if __name__ == "__main__":
    # Test de la configuration
    validate_paths()
    print("✓ Configuration valide")
    print(f"  Répertoire markdown: {MARKDOWN_DIR.resolve()}")
    print(f"  Répertoire Chroma: {CHROMA_DB_PATH.resolve()}")
    print(f"  Collection: {COLLECTION_NAME}")
    print(f"  Modèle défaut: {DEFAULT_MODEL}")
    print(f"  Modèle volumineux: {LARGE_DOC_MODEL} (>50K chars)")
