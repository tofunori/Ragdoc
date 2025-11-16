# Migration Plan: ChromaDB Native Hybrid Search (v2.0)

**Status:** 📋 Planning Phase
**Target Version:** v2.0.0
**Estimated Effort:** 8-12 days
**Risk Level:** Medium-High
**Last Updated:** 2024

---

## 🎯 Executive Summary

ChromaDB now supports native hybrid search combining:
- **Dense embeddings** (semantic - Voyage)
- **Sparse embeddings** (BM25-like - TF-IDF)
- **RRF** (Reciprocal Rank Fusion) - built-in C++

### Benefits
- ✅ **93% less custom code** (300 lines → 20 lines)
- ✅ **Persistent BM25 index** (no rebuild on restart)
- ✅ **2-3x faster** (C++ vs Python)
- ✅ **Official ChromaDB support**

### Costs
- ❌ **8-12 days effort**
- ❌ **Full reindexing** (24,884 chunks)
- ❌ **Less control** over tokenization
- ❌ **Migration risk**

---

##  📊 Current vs Target Architecture

### Current Architecture (v1.5.0)

```
RAGDOC Custom Hybrid
├── BM25Okapi (rank_bm25)
│   ├── Tokenization: Advanced (stemming + stopwords + compounds)
│   ├── Index: RAM (rebuilt each start)
│   ├── Performance: Python
│   └── Code: 300 lines custom
├── Voyage Embeddings
│   ├── Storage: ChromaDB
│   └── Type: Dense vectors only
└── RRF Fusion (Custom)
    ├── Implementation: Python
    └── Flexibility: Full control
```

### Target Architecture (v2.0)

```
ChromaDB Native Hybrid
├── Dense Embeddings (Voyage)
│   ├── Storage: ChromaDB
│   └── Type: Dense vectors
├── Sparse Embeddings (TF-IDF/BM25)
│   ├── Generation: Custom (TfidfVectorizer with AdvancedTokenizer)
│   ├── Storage: ChromaDB metadata['sparse_embedding']
│   ├── Index: Persisted on disk
│   └── Performance: C++ native
└── RRF (Native ChromaDB)
    ├── Implementation: C++ built-in
    ├── API: Search().rank(Rrf(...))
    └── Code: ~20 lines
```

---

## 🗺️ Migration Phases

### Phase 1: Research & Prototyping (2 days)

#### Objectives
- Validate ChromaDB native hybrid on small dataset
- Test sparse embedding generation quality
- Benchmark performance vs current system
- Identify technical limitations

#### Tasks

**1. Create Test Environment**
```bash
mkdir chroma_hybrid_test
cd chroma_hybrid_test
```

**2. Implement Sparse Embedding Generator**

```python
# src/embeddings/sparse_generator.py

from sklearn.feature_extraction.text import TfidfVectorizer
from tokenizers import AdvancedTokenizer

class SparseEmbeddingGenerator:
    """
    Generate sparse embeddings for ChromaDB hybrid search.

    Uses TF-IDF with our advanced tokenizer to maintain
    stemming and compound term benefits.
    """

    def __init__(self, max_features: int = 10000):
        # Use our advanced tokenizer
        self.advanced_tokenizer = AdvancedTokenizer()

        # TF-IDF vectorizer with custom tokenizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            lowercase=False,  # Already done by tokenizer
            tokenizer=self.advanced_tokenizer.tokenize,
            token_pattern=None,  # Use custom tokenizer
            ngram_range=(1, 1),  # Unigrams (compounds handled by tokenizer)
        )

    def fit(self, documents: List[str]):
        """Fit vectorizer on document corpus"""
        self.vectorizer.fit(documents)

    def transform(self, text: str) -> dict:
        """
        Transform text to sparse embedding (dict format for ChromaDB).

        Returns:
            Sparse vector as dict: {index: tf-idf_value}
        """
        vector = self.vectorizer.transform([text])

        # Convert scipy sparse matrix to dict
        sparse_dict = {}
        cx = vector.tocoo()
        for idx, value in zip(cx.col, cx.data):
            sparse_dict[int(idx)] = float(value)

        return sparse_dict

    def save(self, path: str):
        """Save fitted vectorizer"""
        import joblib
        joblib.dump(self.vectorizer, path)

    def load(self, path: str):
        """Load fitted vectorizer"""
        import joblib
        self.vectorizer = joblib.load(path)
```

**3. Test on Sample Dataset (100 documents)**

```python
# scripts/test_chromadb_native.py

from chromadb import Search, K, Knn, Rrf
import chromadb

# 1. Setup
client = chromadb.Client()
collection = client.create_collection("test_hybrid")

# 2. Load 100 sample documents
sample_docs = load_sample_documents(100)

# 3. Generate embeddings
sparse_gen = SparseEmbeddingGenerator()
sparse_gen.fit([doc.text for doc in sample_docs])

for doc in sample_docs:
    # Dense embedding (Voyage)
    dense_emb = voyage_client.embed([doc.text])[0]

    # Sparse embedding
    sparse_emb = sparse_gen.transform(doc.text)

    # Add to collection
    collection.add(
        ids=[doc.id],
        documents=[doc.text],
        embeddings=[dense_emb],
        metadatas=[{
            **doc.metadata,
            'sparse_embedding': sparse_emb
        }]
    )

# 4. Test hybrid search
query = "black carbon on glacier albedo"
query_sparse = sparse_gen.transform(query)

hybrid_rank = Rrf(
    ranks=[
        Knn(query=query, return_rank=True, limit=100),
        Knn(query=query_sparse, key="sparse_embedding",
            return_rank=True, limit=100)
    ],
    weights=[0.7, 0.3],
    k=60
)

search = Search().rank(hybrid_rank).limit(10)
results = collection.search(search)

# 5. Compare vs current system
current_results = hybrid_retriever.search(query, top_k=10)
compare_results(results, current_results)
```

**4. Benchmark Performance**

```python
# Metrics to measure:
# - Recall@10 (vs current)
# - Precision@10 (vs current)
# - Search latency
# - Index size on disk
# - Startup time (no rebuild needed)
```

#### Deliverables
- [ ] Proof of concept code working
- [ ] Benchmark results (quality + performance)
- [ ] Technical limitations documented
- [ ] Go/No-go decision

---

### Phase 2: Data Migration Strategy (1 day)

#### Full Reindexing (Recommended)

**Pros:**
- Clean slate
- No legacy issues
- Simple rollback

**Cons:**
- 3-4 hours downtime
- One-time reindexing cost

**Migration Script:**

```python
# scripts/migrate_to_sparse.py

def migrate_collection(
    old_collection_name: str,
    new_collection_name: str
):
    """
    Migrate from v1.5 (custom hybrid) to v2.0 (ChromaDB native).

    Steps:
    1. Fetch all data from old collection
    2. Fit sparse generator on corpus
    3. Create new collection
    4. Reindex with sparse embeddings
    5. Validate data integrity
    """

    print("=" * 70)
    print("RAGDOC v2.0 Migration: ChromaDB Native Hybrid")
    print("=" * 70)

    # 1. Fetch all data
    print("\n[1/5] Fetching data from old collection...")
    old_collection = client.get_collection(old_collection_name)
    all_data = old_collection.get(
        include=["documents", "metadatas", "embeddings"]
    )

    total_docs = len(all_data['ids'])
    print(f"✓ Found {total_docs} chunks to migrate")

    # 2. Fit sparse generator
    print("\n[2/5] Fitting sparse embedding generator...")
    sparse_gen = SparseEmbeddingGenerator(max_features=10000)
    sparse_gen.fit(all_data['documents'])
    print("✓ Sparse generator fitted")

    # Save generator for future use
    sparse_gen.save("models/sparse_vectorizer_v2.pkl")
    print("✓ Sparse generator saved to models/")

    # 3. Create new collection
    print("\n[3/5] Creating new collection...")
    new_collection = client.create_collection(
        name=new_collection_name,
        metadata={"version": "2.0", "hybrid": "native"}
    )
    print(f"✓ Collection '{new_collection_name}' created")

    # 4. Reindex with sparse embeddings
    print("\n[4/5] Reindexing with sparse embeddings...")
    batch_size = 100
    migrated_count = 0

    for i in range(0, total_docs, batch_size):
        batch_ids = all_data['ids'][i:i+batch_size]
        batch_docs = all_data['documents'][i:i+batch_size]
        batch_embeddings = all_data['embeddings'][i:i+batch_size]
        batch_metadatas = all_data['metadatas'][i:i+batch_size]

        # Add sparse embeddings to metadata
        for j, doc in enumerate(batch_docs):
            sparse_emb = sparse_gen.transform(doc)
            batch_metadatas[j]['sparse_embedding'] = sparse_emb

        # Add to new collection
        new_collection.add(
            ids=batch_ids,
            documents=batch_docs,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas
        )

        migrated_count += len(batch_ids)
        progress = (migrated_count / total_docs) * 100
        print(f"  Progress: {migrated_count}/{total_docs} ({progress:.1f}%)")

    print(f"✓ All {total_docs} chunks migrated")

    # 5. Validate
    print("\n[5/5] Validating migration...")
    new_count = new_collection.count()

    if new_count == total_docs:
        print(f"✓ Validation passed: {new_count}/{total_docs} chunks")
    else:
        print(f"✗ WARNING: Count mismatch! {new_count} vs {total_docs}")
        return False

    print("\n" + "=" * 70)
    print("✓ Migration completed successfully!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Test hybrid search on new collection")
    print("2. Compare quality with old collection")
    print("3. Update server.py to use new collection")
    print("4. Archive old collection")

    return True


if __name__ == "__main__":
    # Run migration
    success = migrate_collection(
        old_collection_name="zotero_research_context_hybrid_v3",
        new_collection_name="zotero_research_context_hybrid_v4_native"
    )

    if success:
        print("\n✓ Ready to deploy v2.0")
    else:
        print("\n✗ Migration failed - please review logs")
```

#### Rollback Plan

```python
# If migration fails or quality degrades:

# 1. Simply switch back to old collection
COLLECTION_NAME = "zotero_research_context_hybrid_v3"  # Old collection

# 2. Restart server
# Done! Back to v1.5.0 in 30 seconds
```

---

### Phase 3: Code Refactoring (2-3 days)

#### Files to DELETE
- `src/hybrid_retriever.py` (300 lines → 0)

#### Files to MODIFY
- `src/server.py` - Use ChromaDB native

#### New Implementation

**Before (v1.5.0):** `src/server.py`
```python
# 50+ lines
from hybrid_retriever import HybridRetriever

hybrid_retriever = HybridRetriever(
    collection,
    embedding_function=voyage_embed
)

@mcp.tool()
def semantic_search_hybrid(query: str, top_k: int = 10, alpha: float = 0.5):
    # 1. Hybrid search (BM25 + Semantic)
    results = hybrid_retriever.search(
        query=query,
        top_k=50,
        alpha=alpha,
        bm25_top_n=100,
        semantic_top_n=100
    )

    # 2. Rerank
    docs = [r['text'] for r in results]
    reranked = cohere_client.rerank(
        model="rerank-v3.5",
        query=query,
        documents=docs,
        top_n=top_k
    )

    # 3. Format & return
    return format_results(reranked)
```

**After (v2.0):** `src/server.py`
```python
# ~20 lines
from chromadb import Search, K, Knn, Rrf
from embeddings.sparse_generator import SparseGeneratorService

# Initialize sparse generator (singleton)
sparse_gen = SparseGeneratorService.get_instance()

@mcp.tool()
def semantic_search_hybrid(query: str, top_k: int = 10, alpha: float = 0.5):
    # 1. Generate query sparse embedding
    query_sparse = sparse_gen.transform(query)

    # 2. Create hybrid rank
    hybrid_rank = Rrf(
        ranks=[
            Knn(query=query, return_rank=True, limit=200),
            Knn(query=query_sparse, key="sparse_embedding",
                return_rank=True, limit=200)
        ],
        weights=[alpha, 1-alpha],
        k=60
    )

    # 3. Execute search
    search = (Search()
        .rank(hybrid_rank)
        .limit(50)  # Oversample for reranking
        .select(K.DOCUMENT, K.SCORE, "title", "source")
    )

    results = collection.search(search)

    # 4. Optional Cohere reranking
    if enable_reranking:
        results = cohere_rerank(results, query, top_k)

    # 5. Format & return
    return format_results(results)
```

**Code reduction:** 300 lines → 20 lines (93%)

---

### Phase 4: Testing & Validation (2-3 days)

#### Test Suite

**1. Unit Tests**
- Sparse embedding generation
- RRF scoring correctness
- Metadata handling

**2. Integration Tests**
- End-to-end hybrid search
- Filtering + hybrid search
- Reranking pipeline

**3. Quality Tests**
```python
# tests/test_v2_quality.py

def test_recall_vs_v15():
    """Ensure v2.0 recall >= v1.5.0"""
    v15_recall = evaluate_v15(test_queries)
    v20_recall = evaluate_v20(test_queries)

    assert v20_recall >= v15_recall - 0.02,  # Allow 2% tolerance
           f"Recall regression: {v20_recall} < {v15_recall}"

def test_precision_vs_v15():
    """Ensure v2.0 precision >= v1.5.0"""
    v15_precision = evaluate_v15_precision(test_queries)
    v20_precision = evaluate_v20_precision(test_queries)

    assert v20_precision >= v15_precision - 0.02
```

**4. Performance Tests**
- Latency benchmarks
- Throughput tests
- Memory usage
- Startup time (should be faster - no index rebuild)

#### Acceptance Criteria
- [ ] Recall@10 >= v1.5.0 (no regression)
- [ ] Latency < 200ms average
- [ ] All existing MCP tools work
- [ ] No data loss in migration
- [ ] Startup time < 5 seconds (vs ~30s in v1.5)

---

### Phase 5: Documentation & Deployment (1 day)

#### Documentation Updates
- [ ] README.md
- [ ] MCP_TOOLS_GUIDE.md
- [ ] HYBRID_SEARCH_GUIDE.md
- [ ] API documentation
- [ ] Release notes v2.0.0

#### Deployment Checklist
- [ ] Backup current database
- [ ] Run migration script
- [ ] Validate data integrity
- [ ] Deploy new server code
- [ ] Monitor initial performance
- [ ] Rollback plan ready

---

## ⚖️ Decision Matrix

| Factor | Custom (v1.5.0) | ChromaDB Native (v2.0) |
|--------|-----------------|------------------------|
| **Code Maintenance** | 300 lines | 20 lines ✅ |
| **BM25 Index** | RAM (rebuilt each start) | Disk (persistent) ✅ |
| **Startup Time** | ~30s (rebuild index) | <5s ✅ |
| **Performance** | Python | C++ (2-3x faster) ✅ |
| **Tokenization Control** | Full control ✅ | Limited ❌ |
| **Migration Effort** | 0 days ✅ | 8-12 days ❌ |
| **Risk** | Low ✅ | Medium ❌ |
| **Support** | DIY ❌ | Official ChromaDB ✅ |

---

## 🚦 Go/No-Go Criteria

### ✅ GO if:
- [ ] Phase 1 prototype shows quality >= v1.5.0
- [ ] Performance 2x+ faster than current
- [ ] Team has 2 weeks available
- [ ] ChromaDB native is stable (not beta)
- [ ] Sparse embeddings maintain advanced tokenization benefits

### ⛔ NO-GO if:
- [ ] Quality regression vs v1.5.0
- [ ] Tokenization too limited for scientific terms
- [ ] Migration complexity too high
- [ ] ChromaDB native has critical bugs

---

## 📅 Timeline

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| 1. Research & Prototype | 2 days | - |
| 2. Migration Strategy | 1 day | Phase 1 |
| 3. Code Refactoring | 2-3 days | Phase 2 |
| 4. Testing & Validation | 2-3 days | Phase 3 |
| 5. Documentation & Deploy | 1 day | Phase 4 |
| **TOTAL** | **8-10 days** | |
| **Buffer** | +2 days | Unexpected issues |
| **Total with buffer** | **10-12 days** | |

---

## 💰 Cost-Benefit Analysis

### Costs
- **Engineering time:** 10-12 days
- **Risk:** Migration bugs, potential data loss
- **Downtime:** 3-4 hours
- **Learning curve:** New ChromaDB API

### Benefits
- **Code reduction:** -280 lines (93% less code)
- **Performance:** 2-3x faster search
- **Maintenance:** Much simpler (20 vs 300 lines)
- **Reliability:** Official ChromaDB support
- **Startup:** 6x faster (~30s → ~5s)
- **Persistence:** BM25 index on disk (no rebuild)

### ROI Calculation
- **Cost:** 10 days × $X/day = $X
- **Savings:** ~2h/month maintenance × $Y/hour = $Z/year
- **Break-even:** ~6 months

---

## 🎯 Recommendation

### Current Status (2024)
**Recommendation:** ⏸️ WAIT

**Reasons:**
1. ✅ v1.5.0 achieves same quality (+15% recall)
2. ✅ Custom approach gives full tokenization control
3. ⏸️ ChromaDB native still maturing
4. 💰 10-12 days better spent on features
5. 🧪 Need more time to validate ChromaDB native stability

### Future Trigger Points

**Revisit migration when:**
1. **Maintenance burden** > 2h/month
2. **ChromaDB native** adds better tokenization hooks
3. **Performance** becomes critical bottleneck
4. **Team availability** = 2+ weeks
5. **ChromaDB stability** = 1+ year proven track record

### Suggested Timeline
- **Q2 2025:** Reassess ChromaDB native maturity
- **Q3 2025:** Run Phase 1 prototype if mature enough
- **Q4 2025:** Execute migration if prototype successful

---

## 📚 References

- [ChromaDB Hybrid Search Docs](https://docs.trychroma.com/guides/hybrid-search)
- [Sparse Embeddings Guide](https://docs.trychroma.com/guides/sparse-embeddings)
- [RRF Algorithm Paper](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [ChromaDB GitHub](https://github.com/chroma-core/chroma)

---

**Last Updated:** 2024
**Owner:** RAGDOC Team
**Status:** 📋 Planning - Not Approved
**Next Review:** Q2 2025
