# Release Notes v1.7.0 - The Contextualized Unification 🧠

This release marks a major architectural shift for RAGDOC, moving away from the complex "Hybrid" mode to a fully **Contextualized** pipeline powered by **Voyage-Context-3**.

## 🚀 Major Changes

### 1. Full Adoption of Voyage-Context-3
-   **Unified Model**: We have standardized on `voyage-context-3` (32k context window) for ALL document processing.
-   **Why?**: This model allows the embedding of chunks to "see" the entire document context, significantly improving retrieval accuracy without needing complex "large vs small" document logic.
-   **Robustness**: Implemented smart batching (size 10) and extended timeouts (20 min) to handle massive documents (700k+ tokens) without crashing.

### 2. Deprecation of "Hybrid Mode"
-   **Simplified Pipeline**: The old logic that switched between `voyage-large-2` and `voyage-context-3` based on document size has been removed.
-   **One Path**: Now, every document follows the same robust, high-quality indexing path.
-   **Cleanup**: Removed `index_hybrid_collection.py`, `scripts/index_contextualized_adaptive.py`, and related legacy code.

### 3. Professional User Interface
-   **New TUI**: `ragdoc-menu.py` has been completely rewritten with `rich` and `questionary`.
-   **Features**: Arrow key navigation, real-time indexing logs, and clear status indicators.

### 4. Codebase Hygiene
-   **Massive Cleanup**: Deleted over 15 obsolete utility scripts from the root directory.
-   **Reorganization**: Moved all root-level tests to the `tests/` directory.
-   **Result**: A clean, maintainable project structure.

## 📦 Upgrade Instructions

```bash
git pull origin master
pip install -r requirements.txt
# Re-index your documents to benefit from the new model
python ragdoc-menu.py
# Select "Réindexation Forcée"
```
