#!/usr/bin/env python3
"""
Configuration Loader for RAGDOC
Loads YAML configurations and provides backward compatibility with indexing_config.py
"""

import yaml
from pathlib import Path
from typing import Any, Dict

# Base paths
CONFIG_DIR = Path(__file__).parent
PROJECT_ROOT = CONFIG_DIR.parent


class ConfigLoader:
    """Load and manage YAML configurations"""

    def __init__(self):
        self.models = self._load_yaml("models.yaml")
        self.chunking = self._load_yaml("chunking.yaml")
        self.database = self._load_yaml("database.yaml")

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load a YAML configuration file"""
        config_path = CONFIG_DIR / filename
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def get(self, path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Examples:
            config.get("models.embedding.default.name")  # "voyage-context-3"
            config.get("chunking.token_chunker.chunk_size")  # 1024
        """
        keys = path.split('.')

        # Determine which config dict to use
        if keys[0] == "models":
            obj = self.models
        elif keys[0] == "chunking":
            obj = self.chunking
        elif keys[0] == "database":
            obj = self.database
        else:
            return default

        # Navigate through the keys
        for key in keys[1:]:
            if isinstance(obj, dict) and key in obj:
                obj = obj[key]
            else:
                return default

        return obj


# Singleton instance
_config = ConfigLoader()


# ============================================================================
# BACKWARD COMPATIBILITY INTERFACE
# Provides same constants as indexing_config.py
# ============================================================================

# Paths
MARKDOWN_DIR = PROJECT_ROOT / _config.get("database.paths.markdown_dir")
CHROMA_DB_PATH = PROJECT_ROOT / _config.get("database.paths.chroma_db")
CHROMA_DB_HYBRID_PATH = CHROMA_DB_PATH

# Collections
COLLECTION_NAME = _config.get("database.collections.legacy.name")
COLLECTION_HYBRID_NAME = _config.get("database.collections.hybrid.name")

# Collection metadata (HNSW)
COLLECTION_METADATA = {
    "hnsw:space": _config.get("database.hnsw.space"),
    "hnsw:construction_ef": _config.get("database.hnsw.construction_ef"),
    "hnsw:M": _config.get("database.hnsw.M")
}

COLLECTION_HYBRID_METADATA = {
    **COLLECTION_METADATA,
    "pipeline": _config.get("database.collections.hybrid.metadata.pipeline")
}

# Legacy chunking parameters
CHUNK_SIZE = _config.get("chunking.legacy.normal.chunk_size")
CHUNK_OVERLAP = _config.get("chunking.legacy.normal.chunk_overlap")
LARGE_DOC_CHUNK_SIZE = _config.get("chunking.legacy.large.chunk_size")
LARGE_DOC_CHUNK_OVERLAP = _config.get("chunking.legacy.large.chunk_overlap")

# Chonkie parameters
CHONKIE_CHUNK_SIZE = _config.get("chunking.token_chunker.chunk_size")
CHONKIE_CHUNK_OVERLAP = _config.get("chunking.token_chunker.chunk_overlap")
CHONKIE_TOKENIZER = _config.get("chunking.token_chunker.tokenizer")

# Embedding models
DEFAULT_MODEL = _config.get("models.embedding.default.name")
LARGE_DOC_MODEL = _config.get("models.embedding.large.name")
LARGE_DOC_THRESHOLD = _config.get("models.embedding.large_doc_threshold")

# Retrieval
CONTEXT_WINDOW_SIZE = _config.get("models.retrieval.context_window_size")

# Deduplication
USE_CONTENT_HASH = _config.get("database.deduplication.use_content_hash")
TRACK_INDEXED_DATE = _config.get("database.deduplication.track_indexed_date")
AUTO_DELETE_MISSING = _config.get("database.deduplication.auto_delete_missing")

# Logging
LOG_LEVEL = _config.get("database.logging.level")


# ============================================================================
# VALIDATION
# ============================================================================

def validate_paths():
    """Validate that required paths exist"""
    if not MARKDOWN_DIR.exists():
        raise FileNotFoundError(f"Markdown directory not found: {MARKDOWN_DIR}")

    if not CHROMA_DB_PATH.exists():
        print(f"⚠️  ChromaDB directory will be created: {CHROMA_DB_PATH}")


def validate_config():
    """Validate configuration consistency"""
    errors = []

    # Check chunk sizes are positive
    if CHONKIE_CHUNK_SIZE <= 0:
        errors.append("chunking.token_chunker.chunk_size must be > 0")

    # Check overlap is less than chunk size
    if CHONKIE_CHUNK_OVERLAP >= CHONKIE_CHUNK_SIZE:
        errors.append("chunking.token_chunker.chunk_overlap must be < chunk_size")

    # Check model names are valid
    valid_models = ["voyage-context-3", "voyage-3-large", "voyage-3", "voyage-2"]
    if DEFAULT_MODEL not in valid_models:
        errors.append(f"models.embedding.default.name must be one of {valid_models}")

    if errors:
        raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))


# ============================================================================
# MAIN (for testing)
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("RAGDOC Configuration Loader".center(70))
    print("=" * 70)

    # Validate
    try:
        validate_config()
        validate_paths()
        print("\n[OK] Configuration valid")
    except Exception as e:
        print(f"\n[ERROR] Configuration error: {e}")
        exit(1)

    # Display key settings
    print(f"\n{'Paths:':<30}")
    print(f"  {'Markdown directory:':<28} {MARKDOWN_DIR.resolve()}")
    print(f"  {'ChromaDB directory:':<28} {CHROMA_DB_PATH.resolve()}")

    print(f"\n{'Collections:':<30}")
    print(f"  {'Active collection:':<28} {COLLECTION_HYBRID_NAME}")
    print(f"  {'Legacy collection:':<28} {COLLECTION_NAME}")

    print(f"\n{'Chunking:':<30}")
    print(f"  {'Token chunk size:':<28} {CHONKIE_CHUNK_SIZE} tokens")
    print(f"  {'Token chunk overlap:':<28} {CHONKIE_CHUNK_OVERLAP} tokens")
    print(f"  {'Tokenizer:':<28} {CHONKIE_TOKENIZER}")

    print(f"\n{'Embedding Models:':<30}")
    print(f"  {'Default model:':<28} {DEFAULT_MODEL}")
    print(f"  {'Large doc model:':<28} {LARGE_DOC_MODEL}")
    print(f"  {'Large doc threshold:':<28} {LARGE_DOC_THRESHOLD} chars")

    print(f"\n{'HNSW Settings:':<30}")
    print(f"  {'Distance metric:':<28} {COLLECTION_METADATA['hnsw:space']}")
    print(f"  {'Construction EF:':<28} {COLLECTION_METADATA['hnsw:construction_ef']}")
    print(f"  {'M (links):':<28} {COLLECTION_METADATA['hnsw:M']}")

    print("\n" + "=" * 70)
