# RAGDOC v1.6.0 - Enhanced Context Retrieval

## Summary

Improved context retrieval and visibility without compromising Voyage Context-3 best practices. This release delivers significantly better information access while remaining 100% compliant with Voyage AI's official recommendations.

## Key Improvements

### 1. Enhanced Preview Display (2.6x Improvement)
- Main chunk preview: **300 → 800 characters** (+166%)
- Adjacent chunk preview: **150 → 400 characters** (+166%)
- Result: Critical information now fully visible in search results

### 2. Expanded Context Window
- Context window size: **±2 → ±4 chunks**
- Total chunks returned: **5 → 9 chunks** per result
- Result: Broader context capture for better comprehension

### 3. Zero-Overlap Architecture Maintained
- Continues to respect Voyage AI's official recommendation: **NO overlap for voyage-context-3**
- Late chunking architecture already captures document context effectively
- Maintains 78.9% document coherence (vs 47.8% for standard embeddings)

## Technical Details

**Modified Files:**
- `src/server.py` (line 301): Increased preview lengths for better visibility
- `scripts/indexing_config.py` (line 86): Expanded CONTEXT_WINDOW_SIZE from 2 to 4

**No Re-indexation Required:** All improvements are at the retrieval/display level

## Validation

**Test Case:** Query "albedo feedback contribution to glacier melt temperature change"
- Target: Johnson & Rupper (2020) statistic "up to 80%"
- Result: ✓ Successfully displays complete information across 9-chunk context window

## Performance Characteristics

- **Embedding Model:** voyage-context-3 (contextualized)
- **Total Chunks:** 6,159 chunks from 131 documents
- **Indexing Strategy:** 47.6% contextualized_full, 52.4% standard_batched
- **Hybrid Retrieval:** BM25 + Semantic + Cohere v3.5 reranking (unchanged)
- **Document Coherence:** 78.9% (unchanged)

## Upgrade Instructions

```bash
git pull origin master
# No re-indexation needed - changes are display-only
```

## Compatibility

- Fully compatible with existing `chroma_db_contextualized` database
- No API changes to MCP tools
- Works seamlessly with Claude Desktop

## References

- [Voyage AI Context-3 Documentation](https://docs.voyageai.com/docs/embeddings#contextualized-embeddings)
- Voyage AI Recommendation: "We recommend that supplied chunks _not_ have any overlap"

---

**Release Date:** 2025-11-16
**Commit:** 92fe719
**Previous Version:** v1.5.0
