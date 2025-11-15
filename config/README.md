# RAGDOC Configuration

This directory contains YAML-based configuration files for the RAGDOC system.

## Configuration Files

### Core Configurations (Committed to Git)

- **`models.yaml`** - Embedding models and reranking settings
  - Voyage AI embedding models (default, large)
  - Cohere reranking configuration
  - Retrieval parameters

- **`chunking.yaml`** - Text chunking pipeline settings
  - Legacy chunking parameters
  - Chonkie Token Chunker settings
  - Semantic Chunker configuration
  - Overlap Refinery settings

- **`database.yaml`** - ChromaDB and storage configuration
  - Database paths
  - Collection names
  - HNSW index parameters
  - Deduplication settings

### Local Overrides (Ignored by Git)

You can create local override files that won't be committed:

- `local.yaml` - Main local overrides
- `*_local.yaml` - Specific local overrides (e.g., `models_local.yaml`)
- `*.local.yaml` - Alternative naming pattern

## Usage

### Python Scripts

Import the configuration loader:

```python
from config.loader import (
    MARKDOWN_DIR,
    CHROMA_DB_PATH,
    COLLECTION_HYBRID_NAME,
    DEFAULT_MODEL,
    CHONKIE_CHUNK_SIZE,
    # ... other constants
)
```

Or use the advanced dot-notation API:

```python
from config.loader import _config

# Get specific values
model_name = _config.get("models.embedding.default.name")
chunk_size = _config.get("chunking.token_chunker.chunk_size")
```

### Testing Configuration

Run the configuration loader to validate settings:

```bash
python config/loader.py
```

This will display all current settings and validate the configuration.

## Backward Compatibility

The `loader.py` module exports the same constants as the legacy `scripts/indexing_config.py`, ensuring all existing scripts continue to work without modification.

## Adding New Settings

1. Edit the relevant YAML file (`models.yaml`, `chunking.yaml`, or `database.yaml`)
2. Add the new setting in a logical section
3. Update `loader.py` to export the setting as a constant (if needed for backward compatibility)
4. Test with `python config/loader.py`

## Example: Overriding Settings Locally

Create `config/local.yaml`:

```yaml
# Local development overrides
database:
  paths:
    chroma_db: "chroma_db_test"  # Use test database

models:
  embedding:
    default:
      name: "voyage-3-large"  # Use large model by default
```

## Configuration Priority

1. Local overrides (`*_local.yaml`, `local.yaml`) - *Not implemented yet*
2. Environment variables (`.env` file)
3. Default YAML files (`models.yaml`, `chunking.yaml`, `database.yaml`)

## Migration from Python Config

The old `scripts/indexing_config.py` file is now deprecated but still works. To migrate:

1. Settings are now in YAML files (easier to edit)
2. Import from `config.loader` instead of `indexing_config`
3. Use the same constant names - no code changes needed

Example migration:

```python
# Old (still works)
from indexing_config import CHUNK_SIZE, DEFAULT_MODEL

# New (recommended)
from config.loader import CHUNK_SIZE, DEFAULT_MODEL
```
