# RAGDOC - Semantic RAG System for Scientific Literature

**Advanced Retrieval-Augmented Generation system with contextualized embeddings, smart batching, and reranking for scientific research papers.**

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![ChromaDB](https://img.shields.io/badge/vectordb-ChromaDB-orange.svg)](https://www.trychroma.com/)
[![Voyage AI](https://img.shields.io/badge/embeddings-Voyage%20Context%203-green.svg)](https://www.voyageai.com/)
[![Cohere](https://img.shields.io/badge/reranking-Cohere%20v4.0%20Pro-purple.svg)](https://cohere.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready Model Context Protocol (MCP) server with a fully **Contextualized** pipeline for academic research, optimized for scientific literature retrieval.

Scientific reliability updates: canonical versioned document reads, structured
evidence/citation tools, recoverable index writes, explicit search fallbacks and an
annotation-gated evaluation workflow. Read the [migration and validation guide](docs/SCIENTIFIC_RELIABILITY.md)
before upgrading an existing library. Production retrieval quality has not been
measured by the new draft benchmark.

> Note: The legacy “hybrid mode” has been removed. Production search now fuses Voyage Context 4 with a revision-pinned SQLite FTS index; function names are preserved for compatibility.

## 🚀 Key Features

-   **Contextualized Search**: Powered by **Voyage Context 4** with 1024-dimensional contextualized embeddings.
-   **Smart Batching**: Robust handling of massive documents (700k+ tokens) with automatic batching and timeout management.
-   **Professional TUI**: New `ragdoc-menu.py` interface with arrow navigation and real-time indexing feedback.
-   **Evaluation System**: Comprehensive RAG metrics (Recall, Precision, MRR, NDCG) with automated benchmarking.
-   **Cohere Reranking**: v4.0 Pro for high-accuracy multilingual result ranking.
-   **MCP Integration**: Native integration with Claude Desktop and compatible applications.
-   **Incremental Indexing**: MD5-based change detection for efficient updates.

## 📋 Table of Contents

-   [Installation](#installation)
-   [Configuration](#configuration)
-   [Usage](#usage)
-   [Evaluation & Quality Metrics](#evaluation--quality-metrics)
-   [Architecture](#architecture)
-   [Troubleshooting](#troubleshooting)
-   [Performance](#performance)
-   [Contributing](#contributing)

## 🛠️ Installation

### Prerequisites

-   Python 3.10 or higher
-   API Keys: Voyage AI, Cohere (optional)
-   4GB+ RAM recommended

### Quick Install (Windows/macOS/Linux)

```bash
# 1. Clone the repository
git clone https://github.com/tofunori/Ragdoc.git
cd Ragdoc

# 2. Create virtual environment
python -m venv ragdoc-env

# Windows
ragdoc-env\Scripts\activate
# macOS/Linux
source ragdoc-env/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API keys (see Configuration section)
```

For a reproducible development/MCP environment with `uv`, use the committed lockfile:

```bash
uv sync --locked --python 3.12 --extra dev
uv run --locked python -m nltk.downloader -d .venv/nltk_data stopwords
uv run --locked python src/server.py --check-runtime
```

The environment must be isolated from system packages. `--check-runtime` checks
local SDK capabilities and tokenizer data without opening Chroma or calling APIs.
It reports missing API keys separately; successful offline checks do not establish
external service availability. The same diagnostics are exposed as `get_runtime_status`.

Use identical connection settings for the server and indexer:
`RAGDOC_CHROMA_MODE=persistent` selects only `CHROMA_DB_PATH`; `http` selects only
`RAGDOC_CHROMA_HOST`/`RAGDOC_CHROMA_PORT` and fails if unavailable. The legacy
`auto` default probes HTTP first, then falls back to the local path. The active
connection is logged and reported in runtime diagnostics.

### Detailed Installation

#### Windows (PowerShell)
```powershell
# Create virtual environment
python -m venv ragdoc-env
.\ragdoc-env\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Set environment variables
$env:VOYAGE_API_KEY = "your_voyage_api_key"
$env:COHERE_API_KEY = "your_cohere_api_key"
```

#### macOS/Linux (bash/zsh)
```bash
# Create virtual environment
python3 -m venv ragdoc-env
source ragdoc-env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export VOYAGE_API_KEY="your_voyage_api_key"
export COHERE_API_KEY="your_cohere_api_key"
```

#### Alternative: .env File
Create a `.env` file in the project root (copy from `.env.example`):
```env
VOYAGE_API_KEY=your_voyage_api_key
COHERE_API_KEY=your_cohere_api_key
```

## ⚙️ Configuration

### Required API Keys

1.  **Voyage AI** (required)
    -   Sign up: https://voyageai.com/
    -   Model used: **voyage-context-4** (32k per chunk, contextualized)
    -   Cost: ~$0.06 per 1M tokens (Contextualized)

2.  **Cohere** (optional, for reranking)
    -   Sign up: https://cohere.com/
    -   Model used: rerank-v4.0-pro
    -   Free tier available

### Claude Desktop Setup

1.  Install Claude Desktop: https://claude.ai/download
2.  Configure MCP server in Claude settings:

**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "ragdoc": {
      "command": "python",
      "args": ["src/server.py"],
      "cwd": "/path/to/Ragdoc"
    }
  }
}
```

### YAML Configuration

Configuration files are located in `config/`:

-   `models.yaml` - Embedding and reranking models
-   `chunking.yaml` - Chunking pipeline settings
-   `database.yaml` - ChromaDB and HNSW parameters

See `config/README.md` for detailed documentation.

## 🎯 Usage

### Via Claude Desktop

Once configured, use directly in Claude conversations:

```
Search for information about glacier albedo
Find articles about ice mass measurement techniques
What are the remote sensing methods for albedo analysis?
```

### Available MCP Tools

#### Search Tools
-   `semantic_search_hybrid(query, top_k=10, alpha=0.5)` - Contextualized search (BM25 + contextualized embeddings) with reranking
-   `search_by_source(query, sources, top_k=10, alpha=0.5)` - Search limited to specific documents

#### Document Management Tools
-   `list_documents()` - List all indexed documents
-   `get_document_content(source, format="markdown", max_length=None)` - Retrieve complete document content
-   `get_chunk_with_context(chunk_id, context_size=2, highlight=True)` - Show chunk with surrounding context

#### Database Tools
-   `get_indexation_status()` - Database statistics

### Tool Examples

#### Search and Discovery
```python
# Contextualized search (BM25 + contextualized embeddings) - alpha=0.5 is balanced fusion (default)
semantic_search_hybrid("black carbon impact on glacier albedo", top_k=10, alpha=0.5)

# Adjust semantic/lexical weight (alpha=0.5 = equal weight)
semantic_search_hybrid("remote sensing albedo measurement", alpha=0.5)

# Search in specific documents only
search_by_source("glacier albedo", sources=["1982_RGSP.md"])
search_by_source("ice mass balance", sources=["Warren_1982.md", "Painter_2009.md"], top_k=5)

# Get document list
list_documents()
```

#### Document Reading
```python
# Read complete document in markdown format
get_document_content("1982_RGSP.md", format="markdown")

# Read document as plain text with length limit
get_document_content("1982_RGSP.md", format="text", max_length=5000)

# View document as individual chunks with metadata
get_document_content("1982_RGSP.md", format="chunks")
```

#### Context Exploration
```python
# Show chunk with 2 surrounding chunks on each side (default)
get_chunk_with_context("1982_RGSP_chunk_042", context_size=2, highlight=True)

# Show more context (5 chunks before and after)
get_chunk_with_context("1982_RGSP_chunk_042", context_size=5)

# Show context without highlighting
get_chunk_with_context("1982_RGSP_chunk_042", context_size=3, highlight=False)
```

#### Database Management
```python
# Get database statistics
get_indexation_status()
```

### Evaluation & Quality Metrics

RAGDOC includes a comprehensive evaluation system to measure and optimize retrieval quality:

```bash
# Quick Start: Generate test dataset and evaluate
python scripts/generate_test_dataset.py --n_queries 30
python tests/evaluate_ragdoc.py

# View results
cat tests/results/evaluation_report_latest.md
```

**Metrics Measured:**
-   **Recall@K**: What % of relevant documents are found in top-K results?
-   **Precision@K**: What % of top-K results are relevant?
-   **MRR (Mean Reciprocal Rank)**: How early does first relevant result appear?
-   **NDCG@K**: How well are results ranked?

**Historical recognition regression (2025-11-15, 30 queries):**
The archived report records Recall@10 of 0.9667, MRR of 0.9190 and NDCG@10 of
0.9298 for alpha=0.5. Its queries reuse text from indexed passages to retrieve
their source documents. These numbers do not measure current performance on
independent scientific questions. The new 50-question benchmark remains unannotated;
see [scientific evaluation requirements](docs/SCIENTIFIC_RELIABILITY.md#evaluation).

**Configuration Tuning:**
```bash
# Test different alpha values (BM25 vs Semantic weight)
python tests/evaluate_ragdoc.py --alpha 0.3 0.5 0.7 1.0

# Custom dataset
python tests/evaluate_ragdoc.py --dataset tests/test_datasets/my_queries.json
```

**Output Files:**
-   `evaluation_report_latest.md` - Comparison report
-   `evaluation_detailed_latest.json` - Full results
-   `evaluation_aggregate_latest.csv` - Metrics table

See [docs/EVALUATION_GUIDE.md](docs/EVALUATION_GUIDE.md) for complete documentation.

### Indexing Your Documents

```bash
# 1. Add markdown files to articles_markdown/
cp your_paper.md articles_markdown/

# 2. Run the Menu
python ragdoc-menu.py
# Select "Indexation Incrémentale"
```

### Importing local Zotero PDFs with MinerU

With Zotero Desktop running and its local API enabled, create a read-only inventory:

```bash
uv run --locked python scripts/import_zotero_mineru.py inventory
```

Run a small quality-control batch before a whole-library import:

```bash
uv run --locked python scripts/import_zotero_mineru.py import --limit 10
uv run --locked python scripts/import_zotero_mineru.py import
```

The importer reads local PDF attachment paths and bibliographic metadata from
Zotero without modifying its library. It deduplicates attachments by Zotero MD5,
uses MinerU `vlm` with table and formula extraction, keeps reference sections, and
stores one Markdown file plus a metadata sidecar per unique PDF. MinerU content JSON
is retained under `build/zotero_mineru/` to support page provenance; returned archives
and their duplicate PDF payloads are discarded. Completed articles are skipped on
later runs. The inventory and journal are local, ignored build artifacts.

MinerU is a cloud service: every imported PDF is uploaded to it. A token must be
stored in `~/.mineru_token`. The precision API accepts at most 200 MB and 200 pages
per request. The importer automatically divides longer PDFs into 190-page parts and
reassembles their Markdown and page locators. For a malformed large attachment,
preserve the Zotero original and import a repaired, split copy:

```bash
uv run --locked python scripts/import_zotero_mineru.py import --repair-pdf ATTACHMENT_KEY
```

Conversion does not create embeddings. Run the incremental indexer afterward with
`VOYAGE_API_KEY` configured; this can incur Voyage API costs.

## 🏗️ Architecture

### Contextualized Search Pipeline (v1.7.0)

```
Query
  ↓
┌─────────────────────────────┐
│ Fielded SQLite FTS5 BM25    │ → Top 100 diverse candidates (lexical)
│ Voyage Context 4            │ → Top 100 candidates (semantic)
└─────────────────────────────┘
  ↓
┌─────────────────────────────┐
│ Reciprocal Rank Fusion      │ → Top 50 merged results
│ (Weighted RRF)              │
└─────────────────────────────┘
  ↓
┌─────────────────────────────┐
│ Cohere v4.0 Pro Reranking   │ → Top 10 final results
└─────────────────────────────┘
  ↓
┌─────────────────────────────┐
│ Context Window Expansion    │ → Results with adjacent chunks
└─────────────────────────────┘
```

### Technologies Used

-   **SQLite FTS5 BM25**: persistent, revision-pinned fielded search over body, title, authors, identifiers and source, without loading the corpus into RAM
-   **Voyage AI**: **voyage-context-4** contextualized embeddings (1024 dimensions)
-   **ChromaDB 0.5.0+**: HNSW-optimized vector database
-   **Cohere v4.0 Pro**: Multilingual, high-accuracy result reranking
-   **FastMCP**: High-performance MCP server
-   **Rich & Questionary**: Professional TUI

### Document Database

-   **501 research documents** in the current production corpus
-   **70,522 passages** with contextualized indexing
-   **Rich metadata** (source, chunk_index, total_chunks, doc_hash, indexed_date)
-   **Continuous updates** with incremental indexing

## 🔧 Troubleshooting

### Common Issues

#### API Keys Not Found
```
ERROR: VOYAGE_API_KEY not found
```
**Solution**: Check environment variables or `.env` file configuration

#### Import Error
```
ModuleNotFoundError: No module named 'fastmcp'
```
**Solution**: Reactivate virtual environment and reinstall:
```bash
source ragdoc-env/bin/activate  # macOS/Linux
# or
.\ragdoc-env\Scripts\activate   # Windows
pip install -r requirements.txt
```

#### Empty Database
```
Collection empty or not found
```
**Solution**: Run indexation:
```bash
python ragdoc-menu.py
```

#### Slow Performance
-   Check internet connection (Voyage AI embeddings require API calls)
-   Enable GPU if available (CUDA)
-   Reduce number of results in searches
-   Use local ChromaDB server for faster access

### Technical Support

-   **Logs**: Check console output for detailed errors
-   **Status**: Use `get_indexation_status()` for diagnostics
-   **Reset**: Delete `chroma_db_new/` and reindex if necessary

## 📈 Performance

### Performance validation

The operational recognition benchmark exercises the live MCP with deterministic
catalogue-derived DOI and title probes. It records Hit@1, Hit@5, MRR, retrieval
mode, reranking availability and warnings. It does not replace the separate,
human-reviewed scientific-question benchmark.

```bash
python scripts/benchmark_operational_retrieval.py \
  --output outputs/benchmarks/operational.json
```

The persistent lexical sidecar is synchronized after incremental writes and is
used only when its revision and passage count exactly match the Chroma collection.
It caps each source at ten lexical candidates, skips macOS `._` sidecars and uses
separate weights for body, title, authors, identifiers and source. Search may expand
its candidate pool for source diversity, with at most 100 passages sent to Cohere
reranking.

## 🤝 Contributing

Contributions are welcome! To contribute:

1.  Fork the project
2.  Create a feature branch
3.  Add your documents to `articles_markdown/`
4.  Run indexation: `python ragdoc-menu.py`
5.  Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

-   Built with [Chonkie](https://github.com/bhavnicksm/chonkie) for advanced chunking
-   Powered by [Voyage AI](https://voyageai.com/) embeddings
-   Enhanced with [Cohere](https://cohere.com/) reranking
-   Integrated with [Claude Desktop](https://claude.ai/) via MCP

---

**Developed for the scientific research community** 🔬

For questions or issues, please open an issue on GitHub.
