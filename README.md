# RAGDOC - Semantic RAG System for Scientific Literature

**Advanced Retrieval-Augmented Generation system with hybrid chunking, multi-model embeddings, and reranking for scientific research papers.**

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![ChromaDB](https://img.shields.io/badge/vectordb-ChromaDB-orange.svg)](https://www.trychroma.com/)
[![Voyage AI](https://img.shields.io/badge/embeddings-Voyage%20AI-green.svg)](https://www.voyageai.com/)
[![Cohere](https://img.shields.io/badge/reranking-Cohere%20v3.5-purple.svg)](https://cohere.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready Model Context Protocol (MCP) server with hybrid chunking pipeline for academic research, optimized for scientific literature retrieval.

## 🚀 Key Features

- **Hybrid Chonkie Pipeline**: Token → Semantic → Overlap for optimal understanding
- **Voyage AI Embeddings**: Context-3 and Large models for ultra-precise search
- **Cohere Reranking**: v3.5 for intelligent result ranking
- **MCP Integration**: Native integration with Claude Desktop and compatible applications
- **Incremental Indexing**: MD5-based change detection for efficient updates
- **YAML Configuration**: Centralized, declarative configuration system

## 📋 Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Architecture](#architecture)
- [Troubleshooting](#troubleshooting)
- [Performance](#performance)
- [Contributing](#contributing)

## 🛠️ Installation

### Prerequisites

- Python 3.10 or higher
- API Keys: Voyage AI, Cohere (optional)
- 4GB+ RAM recommended

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
pip install -e .

# 4. Configure API keys (see Configuration section)
```

### Detailed Installation

#### Windows (PowerShell)
```powershell
# Create virtual environment
python -m venv ragdoc-env
.\ragdoc-env\Scripts\Activate.ps1

# Install dependencies
pip install fastmcp chromadb voyageai cohere chonkie[model2vec] python-dotenv pyyaml

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
pip install fastmcp chromadb voyageai cohere chonkie[model2vec] python-dotenv pyyaml

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

1. **Voyage AI** (required)
   - Sign up: https://voyageai.com/
   - Models used: voyage-context-3, voyage-3-large
   - Cost: ~$0.02 per 1M tokens

2. **Cohere** (optional, for reranking)
   - Sign up: https://cohere.com/
   - Model used: rerank-v3.5
   - Free tier available

### Claude Desktop Setup

1. Install Claude Desktop: https://claude.ai/download
2. Configure MCP server in Claude settings:

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

- `models.yaml` - Embedding and reranking models
- `chunking.yaml` - Hybrid chunking pipeline settings
- `database.yaml` - ChromaDB and HNSW parameters

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

- `semantic_search(query)` - Main search with reranking
- `topic_search(topic)` - Quick topic-based search
- `list_documents()` - List all indexed documents
- `get_indexation_status()` - Database statistics
- `reindex_documents()` - Reindex documents

### Search Examples

```python
# Keyword search
semantic_search("black carbon impact on glacier albedo")

# Topic search
topic_search("remote sensing albedo measurement")

# Get document list
list_documents()
```

### Indexing Your Documents

```bash
# 1. Add markdown files to articles_markdown/
cp your_paper.md articles_markdown/

# 2. Run incremental indexing
python scripts/index_incremental.py

# 3. Force reindexing (if needed)
python scripts/index_incremental.py --force
```

## 🏗️ Architecture

### Hybrid Chunking Pipeline

```
Academic Document
        ↓
   TokenChunker
   (global structure)
        ↓
 SemanticChunker
 (thematic coherence)
        ↓
 OverlapRefinery
   (context preserved)
        ↓
  Voyage Embeddings
   (semantic vectors)
        ↓
   ChromaDB HNSW
   (fast retrieval)
        ↓
  Cohere Reranking
  (optimized results)
```

### Technologies Used

- **Chonkie 1.4.1**: Hybrid chunking pipeline with Model2Vec
- **Voyage AI**: High-quality contextual embeddings
- **ChromaDB**: HNSW-optimized vector database
- **Cohere**: Intelligent result reranking
- **FastMCP**: High-performance MCP server

### Document Database

- **114+ articles** on glaciology and albedo (example dataset)
- **20,000+ chunks** with semantic segmentation
- **Rich metadata** (strategy, model, context, indexed_date)
- **Continuous updates** with new articles

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
pip install -e .
```

#### Empty Database
```
Collection empty or not found
```
**Solution**: Run indexation:
```bash
python scripts/index_incremental.py
```

#### Slow Performance
- Check internet connection (Voyage AI embeddings require API calls)
- Enable GPU if available (CUDA)
- Reduce number of results in searches
- Use local ChromaDB server for faster access

### Technical Support

- **Logs**: Check console output for detailed errors
- **Status**: Use `get_indexation_status()` for diagnostics
- **Reset**: Delete `chroma_db_new/` and reindex if necessary

## 📈 Performance

### Benchmarks

- **Search**: <500ms for 10 results
- **Indexing**: ~2min/document (full hybrid pipeline)
- **Retrieval**: 95%+ relevance with reranking
- **Scalability**: Supports 10,000+ documents

### Hybrid vs Simple Chunking

| Metric | Simple TokenChunker | Hybrid Pipeline |
|--------|---------------------|-----------------|
| Chunks/document | ~20 | ~200 |
| Semantic coherence | Medium | High |
| Context preservation | Limited | Optimized |
| Search relevance | 75% | 95% |

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the project
2. Create a feature branch
3. Add your documents to `articles_markdown/`
4. Run indexation: `python scripts/index_incremental.py`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Chonkie](https://github.com/bhavnicksm/chonkie) for advanced chunking
- Powered by [Voyage AI](https://voyageai.com/) embeddings
- Enhanced with [Cohere](https://cohere.com/) reranking
- Integrated with [Claude Desktop](https://claude.ai/) via MCP

---

**Developed for the scientific research community** 🔬

For questions or issues, please open an issue on GitHub.
