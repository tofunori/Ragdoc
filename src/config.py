#!/usr/bin/env python3
"""
RAGDOC Configuration Module
Centralizes all configuration constants and paths.
Migrated from scripts/indexing_config.py for better package structure.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# API KEYS
# ============================================================================
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# ============================================================================
# PATHS
# ============================================================================

# Project root determination
# If installed as package, we might need a different strategy, but for now
# assuming we are running from source or standard layout relative to this file
# src/config.py -> project_root is parent of parent
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MARKDOWN_DIR = PROJECT_ROOT / "articles_markdown"

# Main database paths
# Main database paths
# NOTE: Using chroma_db_new for unified database (server + MCP + scripts)
CHROMA_DB_NEW_PATH = PROJECT_ROOT / "chroma_db_new"
CHROMA_DB_CONTEXTUALIZED_PATH = CHROMA_DB_NEW_PATH  # Point to chroma_db_new
CHROMA_DB_PATH = CHROMA_DB_NEW_PATH
CHROMA_DB_HYBRID_PATH = CHROMA_DB_PATH  # Legacy compatibility

# ============================================================================
# CHROMA COLLECTIONS
# ============================================================================

# Legacy collection (Simple TokenChunker)
COLLECTION_NAME_LEGACY = "zotero_research_context_v2"

# Hybrid collection (Token + Semantic + Overlap)
COLLECTION_HYBRID_NAME = "zotero_research_context_hybrid_v3"

# Contextualized Embeddings collection (voyage-context-3)
COLLECTION_CONTEXTUALIZED_NAME = "ragdoc_contextualized_v1"

# Force Contextualized Collection regardless of mode
# RAGDOC_MODE kept only for legacy scripts compatibility
RAGDOC_MODE = "contextualized"
COLLECTION_NAME = COLLECTION_CONTEXTUALIZED_NAME
ACTIVE_DB_PATH = CHROMA_DB_CONTEXTUALIZED_PATH

# Collection Metadata (HNSW optimization)
COLLECTION_METADATA = {
    "hnsw:space": "cosine",
    "hnsw:construction_ef": 400,
    "hnsw:M": 64
}

# Metadata for hybrid collection
COLLECTION_HYBRID_METADATA = {
    "hnsw:space": "cosine",
    "hnsw:construction_ef": 400,
    "hnsw:M": 64,
    "pipeline": "hybrid_token_semantic_overlap"
}

# Metadata for contextualized collection
COLLECTION_CONTEXTUALIZED_METADATA = {
    "hnsw:space": "cosine",
    "hnsw:construction_ef": 400,
    "hnsw:M": 64,
    "pipeline": "contextualized_adaptive",
    "embedding_model": "voyage-context-3",
    "description": "Contextualized chunk embeddings with adaptive strategy"
}

# ============================================================================
# CHUNKING PARAMETERS
# ============================================================================

# Standard documents
CHUNK_SIZE = 2000  # characters
CHUNK_OVERLAP = 400  # characters

# Large documents
LARGE_DOC_CHUNK_SIZE = 1024  # characters
LARGE_DOC_CHUNK_OVERLAP = 200  # characters

# ============================================================================
# CHONKIE PARAMETERS (Tokens)
# ============================================================================

# Optimal Chonkie config for Voyage Context-3
CHONKIE_CHUNK_SIZE = 1024      # Tokens (standard for RAG, 25% of Context-3 window)
CHONKIE_CHUNK_OVERLAP = 180    # Tokens (~17.5% optimal for Context-3)
CHONKIE_TOKENIZER = "gpt2"     # Compatible with Voyage AI

# ============================================================================
# RETRIEVAL & CONTEXT
# ============================================================================

# Number of adjacent chunks to retrieve (before/after main result)
CONTEXT_WINDOW_SIZE = 4  # Returns [n-4 ... n ... n+4]

# ============================================================================
# VOYAGE AI MODELS
# ============================================================================

# Default model for standard docs
DEFAULT_MODEL = "voyage-context-3"

# Model for large docs (>50K chars)
LARGE_DOC_MODEL = "voyage-3-large"

# Threshold to switch to large doc model
LARGE_DOC_THRESHOLD = 50000  # characters

# ============================================================================
# FEATURES & FLAGS
# ============================================================================

# Calculate and store MD5 hash of content
USE_CONTENT_HASH = True

# Track indexation date
TRACK_INDEXED_DATE = True

# Auto-delete chunks for missing files
AUTO_DELETE_MISSING = False

# ============================================================================
# LOGGING
# ============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ============================================================================
# VALIDATION
# ============================================================================

def validate_paths():
    """Validate that critical paths exist."""
    if not MARKDOWN_DIR.exists():
        # Only raise warning, don't crash, as user might be setting up
        print(f"WARNING: Markdown directory not found: {MARKDOWN_DIR}")

    if not CHROMA_DB_PATH.exists() and not CHROMA_DB_CONTEXTUALIZED_PATH.exists():
        print(f"INFO: ChromaDB directories will be created upon usage.")

if __name__ == "__main__":
    validate_paths()
    print("RAGDOC Configuration Loaded")
    print(f"Mode: {RAGDOC_MODE}")
    print(f"Collection: {COLLECTION_NAME}")
