🚀 **RAGDOC v1.1.0 - Hybrid Search Enhancement**

Major search system improvement combining lexical and semantic approaches for optimal results.

---

## ✨ **New Features**

### 🔍 **Hybrid Search**
- **BM25 (Lexical Search)**: Exact keyword matching with rank-bm25
- **Voyage-3-Large (Semantic Search)**: Semantic search via embeddings (1536 dimensions)
- **Reciprocal Rank Fusion (RRF)**: Intelligent fusion of both approaches
- **Cohere v3.5 Reranking**: Final relevance scoring refinement

### 🛠️ **System Improvements**
- **ChromaDB Server Manager**: Robust server lifecycle management
  - Enhanced PID tracking
  - Process tree termination
  - Auto-recovery on errors
- **MCP Integration**: Configuration guide for Claude Desktop
- **Context Window Expansion**: Display adjacent chunks for context

---

## 📊 **Performance**

- **+67% improvement** in result diversity
- **24,884 chunks** indexed and validated
- **Optimized pipeline**:
  - BM25: 100 candidates
  - Semantic: 100 candidates
  - RRF Fusion: Top 50
  - Cohere Reranking: Top 10 final

---

## 📦 **Installation**

```bash
# Install dependencies
pip install rank-bm25

# Verify installation
python check_prerequisites.py

# Activate hybrid search (if migration needed)
python activate_hybrid_search.py
```

---

## 🎯 **Quick Start**

### Via MCP (Claude Desktop)

Configure your `~/.claude.json`:

```json
{
  "mcpServers": {
    "ragdoc": {
      "type": "stdio",
      "command": "path/to/python.exe",
      "args": ["path/to/src/server.py"],
      "env": {
        "COLLECTION_NAME": "zotero_research_context_hybrid_v3"
      }
    }
  }
}
```

See [MCP_SETUP.md](MCP_SETUP.md) for complete configuration.

### Via CLI

```bash
# Test hybrid search
python test_mcp_hybrid.py

# Comparative test
python test_hybrid_search.py
```

---

## 🏗️ **Architecture**

### Core Components

| Component | Technology | Version |
|-----------|-------------|---------|
| **BM25** | rank-bm25 | 0.2.2+ |
| **Embeddings** | Voyage-3-Large | API |
| **Reranking** | Cohere v3.5 | API |
| **Vector DB** | ChromaDB | 1.3.4 |
| **Fusion** | RRF (custom) | - |

### Search Pipeline

```
Query → BM25 Search (100) ┐
                          ├→ RRF Fusion (50) → Cohere Rerank (10) → Results
Query → Semantic (100) ────┘
```

---

## 📝 **Documentation**

- **[HYBRID_SEARCH_GUIDE.md](HYBRID_SEARCH_GUIDE.md)**: Complete hybrid search guide
- **[MCP_SETUP.md](MCP_SETUP.md)**: MCP configuration for Claude Desktop
- **[INSTALLATION.md](INSTALLATION.md)**: Detailed installation guide
- **[START_HERE.md](START_HERE.md)**: Quick start guide

---

## 🔧 **Files Modified/Added**

### New Files
- `src/hybrid_retriever.py`: Hybrid search implementation
- `src/server.py`: MCP server with hybrid search
- `chromadb_server_manager.py`: ChromaDB server manager
- Complete documentation (5 markdown guides)
- Testing and validation scripts

### Improvements
- ChromaDB PID tracking and process management
- MCP server configuration
- Comprehensive testing suite

---

## 🎉 **Production Ready**

RAGDOC v1.1.0 is **production ready** with:
- ✅ Hybrid search validated on 24,884 chunks
- ✅ +67% measured improvement
- ✅ Complete tests and documentation
- ✅ Robust ChromaDB server management
- ✅ Complete MCP integration

---

**Questions or issues?** Open a [GitHub issue](https://github.com/tofunori/Ragdoc/issues)
