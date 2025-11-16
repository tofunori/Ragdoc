# MCP Tools Guide - RAGDOC v1.3.0

Complete guide to all available MCP tools in RAGDOC for scientific literature research.

## Table of Contents

1. [Search Tools](#search-tools)
2. [Document Management Tools](#document-management-tools)
3. [Database Tools](#database-tools)
4. [Workflow Examples](#workflow-examples)

---

## Search Tools

### semantic_search_hybrid

**Hybrid search combining BM25 (lexical) and Voyage-3-Large (semantic) with Cohere v3.5 reranking.**

#### Signature
```python
semantic_search_hybrid(
    query: str,
    top_k: int = 10,
    alpha: float = 0.5
) -> str
```

#### Parameters
- `query` (str, required): Search query about your indexed knowledge base
- `top_k` (int, default=10): Number of results to return (1-50 recommended)
- `alpha` (float, default=0.5): Semantic weight
  - `1.0` = 100% semantic (embeddings only)
  - `0.7` = 70% semantic, 30% BM25
  - `0.5` = 50% semantic, 50% BM25 (recommended - balanced hybrid)
  - `0.3` = 30% semantic, 70% BM25 (BM25-heavy)
  - `0.0` = 100% BM25 (lexical only)

#### Returns
Formatted search results with:
- Reranking scores (Cohere v3.5)
- Hybrid scores (RRF fusion)
- Source document names
- Chunk positions
- BM25 and semantic rankings
- Context window (adjacent chunks)

#### Examples

**Basic search:**
```python
semantic_search_hybrid("glacier albedo measurements")
```

**More results:**
```python
semantic_search_hybrid("ice mass balance techniques", top_k=20)
```

**Adjust semantic/lexical balance:**
```python
# More emphasis on exact keyword matching
semantic_search_hybrid("black carbon in snow", alpha=0.5)

# Full semantic search
semantic_search_hybrid("optical properties of ice", alpha=1.0)

# Full lexical search
semantic_search_hybrid("specific absorption coefficient", alpha=0.0)
```

#### Use Cases
- Finding relevant research papers by topic
- Discovering papers that mention specific terms
- Searching for acronyms and technical terms
- Finding numerical values and measurements
- Exploring related concepts

---

### search_by_source

**Hybrid search limited to specific documents. (NEW v1.3.0)**

#### Signature
```python
search_by_source(
    query: str,
    sources: list,
    top_k: int = 10,
    alpha: float = 0.5
) -> str
```

#### Parameters
- `query` (str, required): Search query about your indexed knowledge base
- `sources` (list, required): List of document filenames to search in
  - Single document: `["1982_RGSP.md"]`
  - Multiple documents: `["Warren_1982.md", "Painter_2009.md"]`
- `top_k` (int, default=10): Number of results to return
- `alpha` (float, default=0.5): Semantic weight (same as semantic_search_hybrid - balanced hybrid)

#### Returns
Formatted search results with same structure as semantic_search_hybrid, but limited to specified documents only.

#### Examples

**Search in a single document:**
```python
search_by_source("glacier albedo measurements", sources=["1982_RGSP.md"])
```

**Search in multiple documents:**
```python
search_by_source(
    "ice mass balance techniques",
    sources=["Warren_1982.md", "Painter_2009.md", "Ren_et_al_2021.md"],
    top_k=5
)
```

**Adjust semantic/lexical balance:**
```python
# More emphasis on exact terms
search_by_source("black carbon", sources=["1982_RGSP.md"], alpha=0.5)
```

#### Use Cases
- **Re-reading a specific paper:** Search within a paper you've already found relevant
- **Comparing across papers:** Search the same query in 2-3 specific papers
- **Topic-focused research:** Limit search to papers from a specific author or year
- **Verification:** Find where a specific concept is mentioned in a known paper
- **Deep dive:** After finding a relevant paper, search for related concepts within it

#### Tips
- Get document names first with `list_documents()`
- Use exact filenames from the list (case-sensitive)
- Combine with `get_chunk_with_context()` to explore results
- For all documents, use `semantic_search_hybrid()` instead

---

## Document Management Tools

### list_documents

**List all indexed documents with metadata.**

#### Signature
```python
list_documents() -> str
```

#### Returns
Formatted list showing:
- Document count
- Document filenames
- Titles
- Number of chunks per document

#### Example
```python
list_documents()
```

**Output:**
```
INDEXED DOCUMENTS: 124 papers
======================================================================

[1] 1982_RGSP.md
    Title: Optical Properties of Snow
    Chunks: 258

[2] 2009_RSE_Painter.md
    Title: Retrieval of subpixel snow covered area
    Chunks: 142

...
```

#### Use Cases
- Discovering what's in your database
- Finding document filenames for `get_document_content()`
- Checking if a paper has been indexed
- Verifying total document count

---

### get_document_content

**Retrieve complete document content by reconstructing from chunks.**

#### Signature
```python
get_document_content(
    source: str,
    format: str = "markdown",
    max_length: int = None
) -> str
```

#### Parameters
- `source` (str, required): Document source filename (from `list_documents()`)
- `format` (str, default="markdown"): Output format
  - `"markdown"` - Formatted with metadata header (recommended)
  - `"text"` - Plain text only (no formatting)
  - `"chunks"` - Individual chunks with indices (for debugging)
- `max_length` (int, optional): Maximum characters to return
  - `None` (default) - Return entire document
  - `5000` - Return first 5000 characters (with "... (truncated)" indicator)

#### Returns
Complete document reconstructed from chunks with:
- **Markdown format**: Title, metadata header, full text
- **Text format**: Full text only
- **Chunks format**: Individual chunks with indices

#### Examples

**Read full document in markdown:**
```python
get_document_content("1982_RGSP.md")
```

**Output:**
```markdown
# Optical Properties of Snow

**Source:** 1982_RGSP.md
**Total chunks:** 258
**Indexed:** 2025-11-12T13:21:17.521474
**Model:** voyage-3-large
**Hash:** a31307a19427aa1a225e819e44ad8739

======================================================================

[Full reconstructed text of the paper...]
```

**Read as plain text with length limit:**
```python
get_document_content("2009_RSE_Painter.md", format="text", max_length=5000)
```

**View chunks structure:**
```python
get_document_content("1982_RGSP.md", format="chunks", max_length=2000)
```

**Output:**
```
DOCUMENT: 1982_RGSP.md
======================================================================
Title: Optical Properties of Snow
Total chunks: 258
Indexed: 2025-11-12T13:21:17.521474
Model: voyage-3-large
Hash: a31307a19427aa1a225e819e44ad8739
======================================================================

[Chunk 0]
---
zotero_source: 1982_RGSP.pdf
converted_date: 2025-11-10T21:50:43.418862
---
REVIEWS OF GEOPHYSICS...
----------------------------------------------------------------------

[Chunk 1]
Optical Properties of Snow
Stephen G. Warren...
----------------------------------------------------------------------

...
```

#### Use Cases
- Reading complete papers after finding them via search
- Extracting text for external processing
- Debugging chunking issues
- Verifying document content
- Creating summaries or analyses of full documents

---

### get_chunk_with_context

**Show a specific chunk with surrounding chunks for context.**

#### Signature
```python
get_chunk_with_context(
    chunk_id: str,
    context_size: int = 2,
    highlight: bool = True
) -> str
```

#### Parameters
- `chunk_id` (str, required): ID of target chunk
  - Format: `{source}_chunk_{index}` (e.g., `1982_RGSP_chunk_042`)
  - Obtain from search results or by inspecting database
- `context_size` (int, default=2): Number of chunks before and after target
  - `1` = 1 chunk before + target + 1 chunk after (3 total)
  - `2` = 2 chunks before + target + 2 chunks after (5 total, recommended)
  - `5` = 5 chunks before + target + 5 chunks after (11 total)
- `highlight` (bool, default=True): Visually mark the target chunk
  - `True` - Show `>>> [TARGET CHUNK] <<<` markers
  - `False` - No special highlighting

#### Returns
Context window showing:
- Target chunk with visual markers (if `highlight=True`)
- Adjacent chunks (±`context_size`)
- Chunk indices and total chunks
- Full text of all chunks in range

#### Examples

**Basic context view:**
```python
get_chunk_with_context("1982_RGSP_chunk_042")
```

**Output:**
```
CHUNK CONTEXT: 1982_RGSP.md
======================================================================
Showing chunk 42/258 with ±2 chunks of context
======================================================================

[Chunk 40/258]
Snow grain size typically ranges from 0.05 mm to 2.0 mm...
----------------------------------------------------------------------

[Chunk 41/258]
The specific surface area (SSA) is inversely proportional to grain size...
----------------------------------------------------------------------

>>> [TARGET CHUNK] <<<
[Chunk 42/258] *** MATCHED RESULT ***
For pure snow, the visible albedo is very high (0.95-0.99)...
>>> [END TARGET CHUNK] <<<
----------------------------------------------------------------------

[Chunk 43/258]
In the near-infrared, snow albedo decreases significantly...
----------------------------------------------------------------------

[Chunk 44/258]
The absorption coefficient increases with wavelength beyond 700 nm...
----------------------------------------------------------------------
```

**More context:**
```python
get_chunk_with_context("2009_RSE_Painter_chunk_028", context_size=5)
```

**Without highlighting:**
```python
get_chunk_with_context("1982_RGSP_chunk_042", context_size=2, highlight=False)
```

#### Use Cases
- Understanding search results in broader context
- Reading surrounding paragraphs after finding a relevant chunk
- Verifying that a match is truly relevant
- Exploring narrative flow around a specific passage
- Debugging chunking boundaries

---

## Database Tools

### get_indexation_status

**Get current indexation database statistics.**

#### Signature
```python
get_indexation_status() -> str
```

#### Returns
Formatted report with:
- Total documents and chunks
- Average chunks per document
- Metadata verification stats
- Model distribution (which embedding models were used)
- Retrieval mode confirmation

#### Example
```python
get_indexation_status()
```

**Output:**
```
INDEXATION STATUS - HYBRID SEARCH ENABLED
======================================================================

GLOBAL STATISTICS:
   Number of documents: 124
   Total chunks: 24884
   Average chunks/doc: 200.7

METADATA VERIFICATION:
   Documents with MD5 hash: 124/124
   Documents with date: 124/124

MODEL DISTRIBUTION:
   voyage-3-large              124 docs (100.0%)

======================================================================
RETRIEVAL MODE: HYBRID (BM25 + Semantic + Reranking)
======================================================================
```

#### Use Cases
- Verifying database health
- Checking total indexed content
- Debugging indexing issues
- Confirming hybrid search is enabled
- Reporting database statistics

---

## Workflow Examples

### Complete Research Workflow

**1. Discovery: Find relevant papers**
```python
# Search for topic
results = semantic_search_hybrid("glacier mass balance remote sensing", top_k=10)
# → Returns 10 most relevant chunks with source filenames
```

**2. Exploration: See what's available**
```python
# List all available documents
docs = list_documents()
# → Shows 124 papers with titles
```

**3. Deep Dive: Read specific chunk in context**
```python
# From search results, you found an interesting chunk: "2009_RSE_Painter_chunk_042"
context = get_chunk_with_context("2009_RSE_Painter_chunk_042", context_size=3)
# → Shows chunk 42 with 3 chunks before and after (7 total)
```

**4. Full Reading: Read complete paper**
```python
# Read the full document
paper = get_document_content("2009_RSE_Painter.md", format="markdown")
# → Returns complete paper with metadata
```

### Focused Reading Workflow

**1. Find specific term**
```python
# Search for exact term using lower alpha (more BM25)
results = semantic_search_hybrid("specific absorption cross-section", alpha=0.3, top_k=5)
```

**2. Read surrounding context**
```python
# Assume first result is chunk "1982_RGSP_chunk_127"
context = get_chunk_with_context("1982_RGSP_chunk_127", context_size=4)
# → Shows 9 chunks total (4 before + target + 4 after)
```

**3. Extract full document if needed**
```python
# Extract as plain text for external analysis
full_text = get_document_content("1982_RGSP.md", format="text")
```

### Database Exploration Workflow

**1. Check what's indexed**
```python
status = get_indexation_status()
# → 124 documents, 24,884 chunks
```

**2. List all documents**
```python
docs = list_documents()
# → Full list with titles and chunk counts
```

**3. Sample random documents**
```python
# Pick first document from list
first_doc = get_document_content("1982_RGSP.md", format="markdown", max_length=3000)
# → Preview first 3000 characters
```

---

## Tips and Best Practices

### Search Tips

1. **Use alpha=0.7 (default) for most searches**: Best balance of semantic understanding and exact matching

2. **Lower alpha (0.3-0.5) for specific terms**: Better for finding exact numbers, acronyms, formulas

3. **Higher alpha (0.8-1.0) for conceptual searches**: Better for finding related ideas even without exact keywords

4. **Adjust top_k based on query specificity**:
   - Specific queries: `top_k=5-10`
   - Broad topics: `top_k=15-30`

### Document Reading Tips

1. **Start with search, then expand**: Find relevant chunks first, then read full documents

2. **Use context view for verification**: Always check `get_chunk_with_context()` before assuming a search result is relevant

3. **Use format="chunks" for debugging**: If document reconstruction looks wrong, inspect individual chunks

4. **Set max_length for previews**: Use `max_length=3000-5000` to preview documents before reading fully

### Performance Tips

1. **List documents before searching**: Use `list_documents()` to know what's available

2. **Check status regularly**: Use `get_indexation_status()` to verify database health

3. **Use smaller context_size first**: Start with `context_size=2`, expand to `context_size=5` only if needed

4. **Limit top_k in searches**: Don't request more results than you need (increases latency)

---

## Troubleshooting

### "Document not found" errors
- Run `list_documents()` to get exact filename
- Check spelling and case sensitivity
- Verify document is indexed with `get_indexation_status()`

### "Chunk not found" errors
- Chunk IDs have format: `{source}_chunk_{index}`
- Use search results to get valid chunk IDs
- Check if document was re-indexed (chunk IDs may change)

### Empty search results
- Try different alpha values (0.3, 0.5, 0.7, 1.0)
- Simplify query (remove complex phrases)
- Check indexation status to verify database is populated

### Slow performance
- Reduce `top_k` in searches (10 is usually sufficient)
- Use smaller `context_size` (2 is recommended)
- Set `max_length` when reading documents
- Consider using ChromaDB server mode for faster access

---

**For more information, see:**
- [README.md](README.md) - Installation and setup
- [HYBRID_SEARCH_GUIDE.md](HYBRID_SEARCH_GUIDE.md) - Hybrid search details
- [MCP_SETUP.md](MCP_SETUP.md) - MCP server configuration

---

**RAGDOC v1.1.0** - Hybrid Search Enhancement
Built for the scientific research community 🔬
