# RAGDOC Configuration Comparison

**Evaluation Date:** 2025-11-15T19:43:08.263434

**Number of Queries:** 30


## Performance by Alpha Value

| Alpha | Mode | Recall@10 | Precision@10 | MRR | NDCG@10 | F1@10 | Time(s) |
|-------|------|-----------|--------------|-----|---------|-------|---------|
| 0.50 | hybrid        | 0.9667 | 0.0967 | 0.9190 | 0.9298 | 0.1758 | 24.40 |
| 0.70 | hybrid        | 0.9667 | 0.0967 | 0.9167 | 0.9298 | 0.1758 | 25.69 |

## Best Configurations

- **Best Recall@10:** alpha=0.50 (0.9667)
- **Best Precision@10:** alpha=0.50 (0.0967)
- **Best MRR:** alpha=0.50 (0.9190)

## Metrics @ K=5

| Alpha | Recall | Precision | F1 | NDCG |
|-------|--------|-----------|-----|------|
| 0.50 | 0.9667 | 0.1933 | 0.3222 | 0.9298 |
| 0.70 | 0.9667 | 0.1933 | 0.3222 | 0.9298 |

## Metrics @ K=10

| Alpha | Recall | Precision | F1 | NDCG |
|-------|--------|-----------|-----|------|
| 0.50 | 0.9667 | 0.0967 | 0.1758 | 0.9298 |
| 0.70 | 0.9667 | 0.0967 | 0.1758 | 0.9298 |

## Metrics @ K=20

| Alpha | Recall | Precision | F1 | NDCG |
|-------|--------|-----------|-----|------|
| 0.50 | 1.0000 | 0.0500 | 0.0952 | 0.9383 |
| 0.70 | 0.9667 | 0.0483 | 0.0921 | 0.9298 |

## Recommendations

Based on this evaluation, **alpha=0.50** provides the best balance:

- Recall@10: 0.9667
- Precision@10: 0.0967
- MRR: 0.9190

This configuration provides **balanced hybrid search**, combining lexical and semantic strengths.