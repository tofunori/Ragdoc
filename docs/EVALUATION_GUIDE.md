# RAGDOC Evaluation Guide

Complete guide to evaluating and optimizing RAGDOC retrieval quality using standard RAG metrics.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Understanding RAG Metrics](#understanding-rag-metrics)
3. [Running Evaluations](#running-evaluations)
4. [Interpreting Results](#interpreting-results)
5. [Creating Custom Datasets](#creating-custom-datasets)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Prerequisites

RAGDOC must be fully set up with:
- ChromaDB database indexed
- Voyage AI and Cohere API keys configured
- Python environment with all dependencies

### Run Your First Evaluation

```bash
# 1. Generate synthetic test dataset (30 queries)
python scripts/generate_test_dataset.py --n_queries 30

# 2. Run evaluation with default settings
python tests/evaluate_ragdoc.py

# 3. View results
cat tests/results/evaluation_report_latest.md
```

**Evaluation takes ~2-3 minutes for 30 queries.**

**Output files:**
- `evaluation_report_latest.md` - Human-readable comparison report
- `evaluation_detailed_latest.json` - Full results for analysis
- `evaluation_aggregate_latest.csv` - Metrics table (Excel-friendly)

---

## Understanding RAG Metrics

RAGDOC uses 4 standard Information Retrieval metrics to measure retrieval quality.

### 1. Recall@K

**What it measures:** What proportion of relevant documents were found in the top-K results?

**Formula:** `|relevant ∩ retrieved@K| / |relevant|`

**Example:**
- Relevant documents: {doc2, doc4, doc7}
- Top-10 results: [doc1, doc2, doc3, doc4, doc5, ...]
- Found in top-10: {doc2, doc4} = 2 out of 3
- **Recall@10 = 0.67 (67%)**

**Interpretation:**
- **0.70-0.85:** Good - Finding most relevant docs
- **0.85-0.95:** Excellent - Finding almost all relevant docs
- **0.95-1.00:** Outstanding - Finding all relevant docs

**When to prioritize:** When you need comprehensive coverage (research, legal)

---

### 2. Precision@K

**What it measures:** What proportion of retrieved documents are relevant?

**Formula:** `|relevant ∩ retrieved@K| / K`

**Example:**
- Top-10 results: [doc1, doc2, doc3, doc4, doc5, ...]
- Relevant in top-10: {doc2, doc4} = 2 out of 10
- **Precision@10 = 0.20 (20%)**

**Interpretation:**
- **0.40-0.60:** Good - Low noise in results
- **0.60-0.80:** Excellent - Very clean results
- **0.80-1.00:** Outstanding - Nearly perfect results

**When to prioritize:** When you need high-quality results (RAG with LLMs)

**Note:** Precision is often lower than Recall because large databases have many irrelevant documents.

---

### 3. MRR (Mean Reciprocal Rank)

**What it measures:** How early does the first relevant document appear?

**Formula:** `1 / rank_of_first_relevant`

**Example:**
- Top-10 results: [doc1, doc2, doc3, doc4, ...]
- First relevant: doc3 at position 3
- **MRR = 1/3 = 0.33 (33%)**

**Interpretation:**
- **MRR = 1.0:** First result is relevant (perfect!)
- **MRR = 0.5:** First relevant at position 2
- **MRR = 0.33:** First relevant at position 3
- **MRR = 0.1:** First relevant at position 10
- **MRR = 0.0:** No relevant results found

**Interpretation ranges:**
- **0.50-0.70:** Good - Relevant result in top 2-3
- **0.70-0.85:** Excellent - Relevant result in top 1-2
- **0.85-1.00:** Outstanding - First result usually relevant

**When to prioritize:** When users only look at top results (search engines)

---

### 4. NDCG@K (Normalized Discounted Cumulative Gain)

**What it measures:** How well are documents ranked? (considers graded relevance and position)

**How it works:**
- Assigns higher scores to relevant docs that appear earlier
- Normalizes by the ideal ranking (perfect order)
- Range: 0.0 (worst) to 1.0 (perfect ranking)

**Formula:** `DCG@K / IDCG@K`

**Interpretation:**
- **0.60-0.75:** Good - Decent ranking quality
- **0.75-0.85:** Excellent - High-quality ranking
- **0.85-1.00:** Outstanding - Near-perfect ranking

**When to prioritize:** When ranking quality matters more than just presence (recommendations)

**Note:** NDCG is especially useful when you have graded relevance (highly relevant, relevant, marginally relevant).

---

### 5. F1@K (Bonus Metric)

**What it measures:** Harmonic mean of Precision and Recall

**Formula:** `2 * (Precision * Recall) / (Precision + Recall)`

**When to use:** When you want a single metric balancing both Precision and Recall

---

## Running Evaluations

### Basic Evaluation

Evaluate with default settings (alpha 0.3, 0.5, 0.7, 1.0):

```bash
python tests/evaluate_ragdoc.py
```

### Custom Alpha Values

Test specific alpha configurations:

```bash
# Test pure BM25 vs pure Semantic
python tests/evaluate_ragdoc.py --alpha 0.0 1.0

# Test hybrid configurations
python tests/evaluate_ragdoc.py --alpha 0.4 0.5 0.6 0.7

# Test many configurations for fine-tuning
python tests/evaluate_ragdoc.py --alpha 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
```

**Alpha parameter explained:**
- `alpha=0.0`: Pure BM25 (lexical matching only)
- `alpha=0.5`: Balanced hybrid (50% BM25, 50% Semantic)
- `alpha=0.7`: Semantic-favored hybrid (30% BM25, 70% Semantic)
- `alpha=1.0`: Pure Semantic (embeddings only)

### Custom K Values

Test different result counts:

```bash
python tests/evaluate_ragdoc.py --k_values 3 5 10
```

### Custom Dataset

Use your own test queries:

```bash
python tests/evaluate_ragdoc.py --dataset tests/test_datasets/my_custom_dataset.json
```

### Full Example

```bash
python tests/evaluate_ragdoc.py \
    --dataset tests/test_datasets/synthetic_ragdoc_qa.json \
    --alpha 0.5 0.7 0.9 \
    --k_values 5 10 20 \
    --output_dir tests/results/experiment_001
```

---

## Interpreting Results

### Reading the Comparison Report

Example report:

```markdown
## Performance by Alpha Value

| Alpha | Mode | Recall@10 | Precision@10 | MRR | NDCG@10 | F1@10 | Time(s) |
|-------|------|-----------|--------------|-----|---------|-------|---------|
| 0.50 | hybrid | 0.9667 | 0.0967 | 0.9190 | 0.9298 | 0.1758 | 24.40 |
| 0.70 | hybrid | 0.9667 | 0.0967 | 0.9167 | 0.9298 | 0.1758 | 25.69 |
```

**How to interpret this:**

1. **Recall@10 = 0.9667 (96.67%)**
   - RAGDOC finds 96.67% of relevant documents in top-10
   - **Verdict:** Excellent! Nearly comprehensive retrieval

2. **Precision@10 = 0.0967 (9.67%)**
   - 9.67% of top-10 results are relevant
   - **Context:** With 25,000+ chunks, this is actually good
   - For RAG: Reranking and LLM will filter further

3. **MRR = 0.9190 (91.90%)**
   - First relevant result appears at position ~1.1
   - **Verdict:** Outstanding! Top result is almost always relevant

4. **NDCG@10 = 0.9298 (92.98%)**
   - Ranking quality is near-perfect
   - **Verdict:** Excellent ordering of results

5. **Time = 24.40s for 30 queries**
   - ~0.8s per query (acceptable for batch evaluation)
   - Production MCP uses caching (much faster)

### Comparing Configurations

**When alpha=0.5 beats alpha=0.7:**
- Queries benefit from exact term matching
- Scientific terminology, formulas, acronyms
- **Action:** Consider lowering default alpha

**When alpha=0.7 beats alpha=0.5:**
- Queries are more conceptual
- Semantic understanding helps
- **Action:** Keep current default

**When all alphas are similar:**
- System is robust across configurations
- **Action:** Stick with alpha=0.5 (current default - balanced hybrid)

### Quality Benchmarks

Based on RAG literature and industry standards:

| Metric | Acceptable | Good | Excellent | Outstanding |
|--------|-----------|------|-----------|-------------|
| Recall@10 | 0.60+ | 0.70-0.85 | 0.85-0.95 | 0.95+ |
| Precision@10 | 0.30+ | 0.40-0.60 | 0.60-0.80 | 0.80+ |
| MRR | 0.40+ | 0.50-0.70 | 0.70-0.85 | 0.85+ |
| NDCG@10 | 0.50+ | 0.60-0.75 | 0.75-0.85 | 0.85+ |

**RAGDOC's typical performance:**
- Recall@10: 0.95-0.97 (Outstanding)
- Precision@10: 0.10-0.15 (Context-dependent*)
- MRR: 0.90-0.95 (Outstanding)
- NDCG@10: 0.90-0.95 (Outstanding)

_*Precision appears low due to large corpus (25K+ chunks) and binary relevance. With reranking and LLM filtering, effective precision is much higher._

---

## Creating Custom Datasets

### Synthetic Dataset (Automated)

```bash
# Generate 50 diverse queries
python scripts/generate_test_dataset.py \
    --n_queries 50 \
    --diversity_mode balanced \
    --output tests/test_datasets/my_dataset.json
```

**Diversity modes:**
- `balanced`: Equal queries per document (recommended)
- `random`: Pure random sampling
- `mixed`: 70% balanced + 30% random

### Manual Dataset (High Quality)

Create a JSON file with this structure:

```json
{
  "name": "My Custom Test Set",
  "version": "1.0",
  "description": "Manually curated queries for glacier research",
  "created": "2025-11-15",
  "num_queries": 10,
  "strategy": "manual",
  "queries": [
    {
      "id": "q001",
      "query": "How does black carbon affect glacier albedo?",
      "relevant_chunks": ["1982_RGSP_chunk_042", "Bond_2013_chunk_105"],
      "relevant_docs": ["1982_RGSP.md", "Bond_et_al_2013.md"],
      "relevance_scores": {
        "1982_RGSP_chunk_042": 3,
        "Bond_2013_chunk_105": 2
      },
      "category": "impurities",
      "difficulty": "medium"
    },
    {
      "id": "q002",
      "query": "Remote sensing techniques for ice mass balance",
      "relevant_chunks": ["Bolch_2010_chunk_023"],
      "relevant_docs": ["Bolch_et_al_2010.md"],
      "relevance_scores": {
        "Bolch_2010_chunk_023": 3
      },
      "category": "remote_sensing",
      "difficulty": "hard"
    }
  ]
}
```

**How to create manual queries:**

1. **Think of real research questions** your users would ask

2. **Find relevant chunks** using RAGDOC:
   ```python
   # In Cursor/Claude Desktop
   search_by_source("your query", sources=["DocumentName.md"])
   ```

3. **Add chunk IDs and relevance scores:**
   - Score 3: Highly relevant (answers the query directly)
   - Score 2: Relevant (contains related information)
   - Score 1: Marginally relevant (tangentially related)
   - Score 0: Not relevant

4. **Save to JSON** and test:
   ```bash
   python tests/evaluate_ragdoc.py --dataset tests/test_datasets/my_manual_set.json
   ```

### Hybrid Approach (Best of Both Worlds)

1. Generate 30 synthetic queries (fast baseline)
2. Add 10-20 manual queries (high-quality validation)
3. Combine into single dataset

```bash
# Generate synthetic
python scripts/generate_test_dataset.py --n_queries 30 --output synthetic.json

# Manually create manual.json with 10 queries

# Combine (manually merge JSON files or write script)
# Use combined dataset for evaluation
```

---

## Troubleshooting

### Empty/Low Results

**Problem:** All metrics near 0.0

**Possible causes:**
1. Dataset chunk IDs don't match ChromaDB IDs
2. ChromaDB collection name mismatch
3. Documents not indexed

**Solutions:**
```bash
# Check ChromaDB collection
python -c "import chromadb; client = chromadb.PersistentClient(path='chroma_db_new'); print(client.get_collection('zotero_research_context_hybrid_v3').count())"

# Verify dataset chunk IDs exist
# Open tests/test_datasets/synthetic_ragdoc_qa.json
# Check if chunk IDs match format: "DocumentName_chunk_042"

# Reindex if needed
python scripts/index_incremental.py
```

### Slow Evaluation

**Problem:** Evaluation takes >5 minutes for 30 queries

**Possible causes:**
1. Network latency (Voyage API)
2. Too many queries
3. Large K values

**Solutions:**
```bash
# Reduce queries
python tests/evaluate_ragdoc.py --dataset small_dataset.json

# Reduce K values
python tests/evaluate_ragdoc.py --k_values 10

# Reduce alpha configurations
python tests/evaluate_ragdoc.py --alpha 0.5 0.7
```

### Inconsistent Results

**Problem:** Results vary significantly between runs

**Possible causes:**
1. Non-deterministic BM25 (shouldn't happen)
2. Database changes between runs
3. Different random seeds

**Solutions:**
```bash
# Use same dataset with fixed seed
python scripts/generate_test_dataset.py --seed 42

# Verify database hasn't changed
# Check indexed_date in metadata
```

### "Module not found" Errors

**Problem:** `ModuleNotFoundError: No module named 'fastmcp'`

**Solution:**
```bash
# Use correct Python environment
"C:\Users\thier\miniforge3\envs\ragdoc-env\python.exe" tests/evaluate_ragdoc.py

# Or activate environment first
source ragdoc-env/bin/activate  # macOS/Linux
.\ragdoc-env\Scripts\activate   # Windows
python tests/evaluate_ragdoc.py
```

---

## Advanced Topics

### Statistical Significance Testing

Compare two configurations statistically:

```python
from scipy import stats

# Load results from two evaluations
results_a = ...  # Alpha=0.5 per-query recalls
results_b = ...  # Alpha=0.7 per-query recalls

# Paired t-test
t_stat, p_value = stats.ttest_rel(results_a, results_b)

if p_value < 0.05:
    print("Difference is statistically significant!")
```

### Cross-Validation

Split your dataset for more robust evaluation:

```python
# 70% queries for tuning, 30% for final validation
# Prevents overfitting to test set
```

### Temporal Evaluation

Track metrics over time as you improve RAGDOC:

```bash
# Run evaluation weekly
python tests/evaluate_ragdoc.py --output_dir results/week_$(date +%V)

# Compare trends
python scripts/compare_temporal_results.py
```

---

## Best Practices

1. **Start with synthetic datasets** (fast, reproducible)
2. **Add manual queries** for validation (10-20 is enough)
3. **Evaluate multiple alpha values** (0.5, 0.7 at minimum)
4. **Track metrics over time** (detect regressions)
5. **Focus on Recall and MRR** for RAG applications
6. **Don't over-optimize** on a small test set

---

## Further Reading

- [Information Retrieval Metrics (Wikipedia)](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval))
- [RAGAS Framework](https://docs.ragas.io/) - RAG-specific evaluation
- [MS MARCO Benchmark](https://microsoft.github.io/msmarco/) - Large-scale IR evaluation

---

**Questions or issues?** Open an issue on [GitHub](https://github.com/tofunori/Ragdoc/issues).

**Built for the scientific research community** 🔬
