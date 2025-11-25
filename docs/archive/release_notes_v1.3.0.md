# 🎯 RAGDOC v1.3.0 - Metadata Filtering & Document-Specific Search

**Release Date:** January 15, 2025
**Type:** Minor Release (New Features)

A powerful enhancement bringing ChromaDB metadata filtering capabilities to RAGDOC, enabling precise document-specific searches and advanced query filtering.

---

## ✨ New Features

### 🔍 search_by_source - Document-Specific Search

**New MCP Tool for searching within specific documents.**

```python
# Search in a single document
search_by_source("glacier albedo", sources=["1982_RGSP.md"])

# Search across multiple documents
search_by_source("ice mass balance",
                 sources=["Warren_1982.md", "Painter_2009.md"],
                 top_k=5)
```

**Key Benefits:**
- 🎯 **Focused results:** Limit search to papers you've already identified as relevant
- ⚡ **Faster searches:** Smaller search space = quicker results
- 🔄 **Paper comparison:** Run same query across 2-3 specific papers
- 📚 **Deep dive:** Explore concepts within a known paper

**Use Cases:**
- Re-reading a specific paper for details
- Verifying where a concept is mentioned
- Comparing how different papers discuss the same topic
- Following up on search results with targeted queries

---

### 🛠️ Enhanced Filtering Infrastructure

**ChromaDB Metadata Filtering Support:**

All search functions now support advanced filtering through ChromaDB's `where` and `where_document` parameters.

**Available Filters:**

**Metadata Filtering (`where`):**
- `source`: Document filename
- `model`: Embedding model (voyage-3-large, voyage-context-3)
- `indexed_date`: Indexation date
- `chunk_index`: Chunk position
- `total_chunks`: Document size
- `title`: Document title

**Content Filtering (`where_document`):**
- `$contains`: Full-text search within documents

**Operators:**
- Comparison: `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`
- Set membership: `$in`, `$nin`
- Logical: `$and`, `$or`

**Examples:**

```python
# Filter by document and position
_perform_search_hybrid(
    "albedo",
    where={
        "$and": [
            {"source": "1982_RGSP.md"},
            {"chunk_index": {"$lte": 50}}
        ]
    }
)

# Filter by multiple documents
_perform_search_hybrid(
    "ice mass",
    where={"source": {"$in": ["paper1.md", "paper2.md"]}}
)

# Filter by content
_perform_search_hybrid(
    "climate",
    where_document={"$contains": "temperature"}
)
```

---

### 🔧 Technical Enhancements

**HybridRetriever Updates:**
- Added `where` parameter to `search()` method
- Added `where_document` parameter to `search()` method
- Modified `_semantic_search()` to accept and pass filters to ChromaDB
- Filters applied to semantic search (BM25 remains in-memory)

**_perform_search_hybrid Updates:**
- Added `where` and `where_document` parameters
- Propagates filters through entire search pipeline
- Maintains backward compatibility (all parameters optional)

**Backward Compatibility:**
- ✅ All existing code continues to work without modification
- ✅ `semantic_search_hybrid()` unchanged in signature
- ✅ Optional parameters don't affect existing workflows

---

## 📊 Statistics

### MCP Tools Count
**Total: 6 tools** (was 5 in v1.2.0)

1. `semantic_search_hybrid` - Hybrid search across all documents
2. `search_by_source` ⭐ **NEW** - Search within specific documents
3. `list_documents` - List all indexed papers
4. `get_document_content` - Retrieve complete document
5. `get_chunk_with_context` - View chunk with context
6. `get_indexation_status` - Database statistics

### Code Changes
- **+280 lines** of code and documentation
- **3 commits** with targeted improvements
- **5 test cases** validating all filtering features

---

## 🧪 Testing

### Test Suite: test_filtering.py

**5 comprehensive tests:**
1. ✅ search_by_source with single document
2. ✅ search_by_source with multiple documents
3. ✅ Complex where filters ($and, $lte)
4. ✅ where_document content filtering
5. ✅ Backward compatibility verification

**Results:**
- All filtering features validated
- search_by_source working correctly
- Metadata filtering functional
- Content filtering operational
- 100% backward compatibility maintained

---

## 📚 Documentation

### Updated Files

**README.md:**
- Added search_by_source to Search Tools
- Examples for single/multiple document filtering
- Updated version references to v1.3.0

**MCP_TOOLS_GUIDE.md:**
- Complete documentation for search_by_source
- Signature, parameters, examples, use cases
- 5 practical use cases
- Usage tips for effective filtering

**Total documentation:** 450+ lines covering all 6 MCP tools

---

## 🎯 Upgrade Instructions

### For Existing Users

**1. Pull latest changes:**
```bash
git pull origin master
```

**2. Restart MCP server**
- Close Cursor or Claude Desktop
- Reopen to reload MCP server with new tools

**3. Test new features:**
```bash
python test_filtering.py
```

**4. Try search_by_source:**
```python
# Get list of documents first
list_documents()

# Search in a specific document
search_by_source("your query", sources=["DocumentName.md"])
```

### For New Users

Follow installation instructions in [README.md](README.md)

---

## 💡 Use Case Examples

### Example 1: Deep Dive into a Paper

```python
# Step 1: Find relevant papers
results = semantic_search_hybrid("glacier albedo measurements", top_k=10)

# Step 2: Deep dive into most relevant paper (Warren 1982)
detailed = search_by_source(
    "spectral albedo near-infrared",
    sources=["1982_RGSP.md"],
    top_k=10
)

# Step 3: View specific findings in context
get_chunk_with_context("1982_RGSP_chunk_042", context_size=3)
```

### Example 2: Compare Across Papers

```python
# Compare how different papers discuss same topic
search_by_source(
    "black carbon impact",
    sources=["Warren_1982.md", "Painter_2009.md", "Ren_2021.md"],
    top_k=5
)
```

### Example 3: Verify Citation

```python
# Find where a concept is mentioned in a known paper
search_by_source(
    "specific absorption cross-section",
    sources=["1982_RGSP.md"],
    alpha=0.5  # More emphasis on exact terms
)
```

---

## 🔮 What's Next?

**Planned for v1.4.0:**
- Advanced metadata search (author, year, keywords)
- Date range filtering for temporal queries
- Model comparison (voyage-3-large vs voyage-context-3 results)

**Planned for v2.0.0:**
- Multi-collection support
- Custom metadata fields
- Advanced filtering UI

---

## 🙏 Acknowledgments

**Built with:**
- [ChromaDB](https://www.trychroma.com/) - Vector database with powerful filtering
- [Voyage AI](https://voyageai.com/) - High-quality embeddings
- [Cohere](https://cohere.com/) - Advanced reranking
- [FastMCP](https://github.com/jlowin/fastmcp) - MCP server framework

**Special thanks** to the ChromaDB team for their excellent filtering API documentation.

---

**For questions or issues, please open an issue on [GitHub](https://github.com/tofunori/Ragdoc/issues).**

**Developed for the scientific research community** 🔬
