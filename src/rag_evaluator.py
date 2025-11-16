#!/usr/bin/env python3
"""
RAG Evaluation Metrics Module for RAGDOC

Implements standard retrieval evaluation metrics:
- Recall@K: Proportion of relevant documents retrieved
- Precision@K: Proportion of retrieved documents that are relevant
- MRR (Mean Reciprocal Rank): Rank of first relevant result
- NDCG@K: Normalized Discounted Cumulative Gain (ranking quality)
- F1@K: Harmonic mean of Precision and Recall

Usage:
    from rag_evaluator import RAGEvaluator

    evaluator = RAGEvaluator()

    # Single query evaluation
    metrics = evaluator.evaluate_query(
        retrieved_ids=['doc1', 'doc2', 'doc3'],
        relevant_ids={'doc2', 'doc5'},
        k_values=[5, 10]
    )

    # Dataset evaluation
    results = evaluator.evaluate_dataset(dataset, search_function)
    report = evaluator.generate_report(results)
"""

import numpy as np
from typing import List, Set, Dict, Tuple, Callable, Optional, Union
import json
import csv
from pathlib import Path
from datetime import datetime


class RAGEvaluator:
    """
    RAG evaluation metrics calculator.

    Provides standard information retrieval metrics for evaluating
    Retrieval-Augmented Generation systems.
    """

    def __init__(self):
        """Initialize RAG evaluator."""
        self.results_history = []

    # ==================== Core Metrics ====================

    @staticmethod
    def recall_at_k(
        retrieved_ids: List[str],
        relevant_ids: Set[str],
        k: int
    ) -> float:
        """
        Calculate Recall@K.

        Recall@K measures what proportion of relevant documents were retrieved
        in the top-K results.

        Formula: |relevant ∩ retrieved@K| / |relevant|

        Args:
            retrieved_ids: List of retrieved document IDs (ordered by rank)
            relevant_ids: Set of relevant document IDs (ground truth)
            k: Number of top results to consider

        Returns:
            Recall@K score (0.0 to 1.0)

        Example:
            >>> retrieved = ['doc1', 'doc2', 'doc3', 'doc4', 'doc5']
            >>> relevant = {'doc2', 'doc4', 'doc7'}
            >>> recall_at_k(retrieved, relevant, k=5)
            0.6667  # Found 2 out of 3 relevant docs
        """
        if not relevant_ids:
            return 0.0

        top_k = set(retrieved_ids[:k])
        relevant_retrieved = top_k & relevant_ids

        return len(relevant_retrieved) / len(relevant_ids)

    @staticmethod
    def precision_at_k(
        retrieved_ids: List[str],
        relevant_ids: Set[str],
        k: int
    ) -> float:
        """
        Calculate Precision@K.

        Precision@K measures what proportion of the top-K retrieved documents
        are relevant.

        Formula: |relevant ∩ retrieved@K| / K

        Args:
            retrieved_ids: List of retrieved document IDs (ordered by rank)
            relevant_ids: Set of relevant document IDs (ground truth)
            k: Number of top results to consider

        Returns:
            Precision@K score (0.0 to 1.0)

        Example:
            >>> retrieved = ['doc1', 'doc2', 'doc3', 'doc4', 'doc5']
            >>> relevant = {'doc2', 'doc4', 'doc7'}
            >>> precision_at_k(retrieved, relevant, k=5)
            0.4  # 2 out of 5 retrieved docs are relevant
        """
        if k <= 0:
            return 0.0

        top_k = set(retrieved_ids[:k])
        relevant_retrieved = top_k & relevant_ids

        return len(relevant_retrieved) / k

    @staticmethod
    def mean_reciprocal_rank(
        retrieved_ids: List[str],
        relevant_ids: Set[str]
    ) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).

        MRR measures how early the first relevant document appears in the
        ranked results. Returns 1/rank of first relevant doc.

        Formula: 1 / rank(first_relevant_doc)

        Args:
            retrieved_ids: List of retrieved document IDs (ordered by rank)
            relevant_ids: Set of relevant document IDs (ground truth)

        Returns:
            MRR score (0.0 to 1.0)
            - 1.0 if first result is relevant
            - 0.5 if second result is relevant
            - 0.0 if no relevant results found

        Example:
            >>> retrieved = ['doc1', 'doc2', 'doc3', 'doc4']
            >>> relevant = {'doc3', 'doc7'}
            >>> mean_reciprocal_rank(retrieved, relevant)
            0.3333  # First relevant doc at position 3
        """
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_ids:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def dcg_at_k(relevance_scores: List[float], k: int) -> float:
        """
        Calculate Discounted Cumulative Gain (DCG) at K.

        DCG is a measure of ranking quality that uses graded relevance scores.
        Higher relevance scores earlier in the ranking produce higher DCG.

        Formula: DCG@K = Σ(i=1 to K) rel_i / log2(i + 1)

        Args:
            relevance_scores: List of relevance scores (in retrieval order)
            k: Number of top results to consider

        Returns:
            DCG@K score
        """
        relevance_scores = np.array(relevance_scores[:k], dtype=np.float64)
        if relevance_scores.size == 0:
            return 0.0

        # Positions: [1, 2, 3, ..., k]
        positions = np.arange(1, len(relevance_scores) + 1)

        # Discounts: log2(position + 1)
        discounts = np.log2(positions + 1)

        # DCG = sum(rel_i / log2(i + 1))
        dcg = np.sum(relevance_scores / discounts)

        return float(dcg)

    def ndcg_at_k(
        self,
        retrieved_ids: List[str],
        relevance_dict: Dict[str, float],
        k: int
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG) at K.

        NDCG normalizes DCG by the ideal DCG (IDCG) to get a score between 0 and 1.
        It measures how close the ranking is to the ideal ranking.

        Formula: NDCG@K = DCG@K / IDCG@K

        Args:
            retrieved_ids: List of retrieved document IDs (ordered by rank)
            relevance_dict: Dictionary mapping document IDs to relevance scores
                           (e.g., 3=highly relevant, 2=relevant, 1=marginally, 0=not)
            k: Number of top results to consider

        Returns:
            NDCG@K score (0.0 to 1.0)
            - 1.0 = perfect ranking
            - 0.0 = no relevant documents or worst possible ranking

        Example:
            >>> retrieved = ['doc1', 'doc2', 'doc3', 'doc4']
            >>> relevance = {'doc1': 0, 'doc2': 3, 'doc3': 2, 'doc4': 1}
            >>> ndcg_at_k(retrieved, relevance, k=4)
            0.8463  # Good but not perfect ranking
        """
        # Get relevance scores for retrieved documents (in order)
        retrieved_relevances = [
            relevance_dict.get(doc_id, 0.0)
            for doc_id in retrieved_ids[:k]
        ]

        # Calculate DCG@K for retrieved ranking
        dcg = self.dcg_at_k(retrieved_relevances, k)

        # Calculate ideal DCG (sort all relevances in descending order)
        ideal_relevances = sorted(relevance_dict.values(), reverse=True)[:k]
        idcg = self.dcg_at_k(ideal_relevances, k)

        # Normalize
        if idcg == 0.0:
            return 0.0

        return dcg / idcg

    @staticmethod
    def f1_at_k(
        retrieved_ids: List[str],
        relevant_ids: Set[str],
        k: int
    ) -> float:
        """
        Calculate F1@K score.

        F1@K is the harmonic mean of Precision@K and Recall@K.
        Useful when you want a single metric balancing both.

        Formula: 2 * (Precision@K * Recall@K) / (Precision@K + Recall@K)

        Args:
            retrieved_ids: List of retrieved document IDs (ordered by rank)
            relevant_ids: Set of relevant document IDs (ground truth)
            k: Number of top results to consider

        Returns:
            F1@K score (0.0 to 1.0)
        """
        precision = RAGEvaluator.precision_at_k(retrieved_ids, relevant_ids, k)
        recall = RAGEvaluator.recall_at_k(retrieved_ids, relevant_ids, k)

        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)

    # ==================== Batch Evaluation ====================

    def evaluate_query(
        self,
        retrieved_ids: List[str],
        ground_truth: Dict[str, Union[Set[str], float]],
        k_values: List[int] = [5, 10, 20],
        use_graded_relevance: bool = False
    ) -> Dict[str, Dict[int, float]]:
        """
        Evaluate a single query across multiple K values.

        Args:
            retrieved_ids: List of retrieved document IDs (ordered by rank)
            ground_truth: Either:
                - Set of relevant document IDs (binary relevance)
                - Dict mapping doc IDs to relevance scores (graded relevance)
            k_values: List of K values to evaluate
            use_graded_relevance: If True, expects relevance scores in ground_truth

        Returns:
            Dictionary of metrics with structure:
            {
                'recall': {5: 0.6, 10: 0.8, 20: 0.9},
                'precision': {5: 0.4, 10: 0.3, 20: 0.2},
                'mrr': 0.5,
                'ndcg': {5: 0.65, 10: 0.72, 20: 0.78},
                'f1': {5: 0.48, 10: 0.43, 20: 0.32}
            }
        """
        results = {
            'recall': {},
            'precision': {},
            'f1': {},
            'ndcg': {}
        }

        # Determine relevant IDs
        if use_graded_relevance:
            relevance_dict = ground_truth
            relevant_ids = {doc_id for doc_id, score in relevance_dict.items() if score > 0}
        else:
            relevant_ids = ground_truth
            # Create binary relevance dict for NDCG
            relevance_dict = {doc_id: 1.0 for doc_id in relevant_ids}

        # Calculate metrics for each K
        for k in k_values:
            results['recall'][k] = self.recall_at_k(retrieved_ids, relevant_ids, k)
            results['precision'][k] = self.precision_at_k(retrieved_ids, relevant_ids, k)
            results['f1'][k] = self.f1_at_k(retrieved_ids, relevant_ids, k)
            results['ndcg'][k] = self.ndcg_at_k(retrieved_ids, relevance_dict, k)

        # MRR is independent of K
        results['mrr'] = self.mean_reciprocal_rank(retrieved_ids, relevant_ids)

        return results

    def evaluate_dataset(
        self,
        dataset: List[Dict],
        search_function: Callable[[str], List[str]],
        k_values: List[int] = [5, 10, 20],
        use_graded_relevance: bool = False
    ) -> Dict:
        """
        Evaluate a complete dataset of queries.

        Args:
            dataset: List of test cases, each containing:
                {
                    'query': str,
                    'relevant_ids': Set[str] or Dict[str, float],
                    'query_id': str (optional)
                }
            search_function: Function that takes a query string and returns
                           a list of document IDs (ordered by relevance)
            k_values: List of K values to evaluate
            use_graded_relevance: If True, expects graded relevance scores

        Returns:
            Dictionary containing:
            - 'aggregate': Average metrics across all queries
            - 'per_query': Metrics for each individual query
            - 'metadata': Evaluation metadata
        """
        per_query_results = []

        for i, test_case in enumerate(dataset):
            query = test_case['query']
            ground_truth = test_case['relevant_ids']
            query_id = test_case.get('query_id', f'query_{i:03d}')

            # Run search
            retrieved_ids = search_function(query)

            # Evaluate
            metrics = self.evaluate_query(
                retrieved_ids,
                ground_truth,
                k_values,
                use_graded_relevance
            )

            per_query_results.append({
                'query_id': query_id,
                'query': query,
                'metrics': metrics,
                'num_retrieved': len(retrieved_ids),
                'num_relevant': len(ground_truth) if isinstance(ground_truth, set)
                               else sum(1 for score in ground_truth.values() if score > 0)
            })

        # Calculate aggregate metrics (mean across queries)
        aggregate = self._aggregate_metrics(per_query_results, k_values)

        # Compile results
        results = {
            'aggregate': aggregate,
            'per_query': per_query_results,
            'metadata': {
                'num_queries': len(dataset),
                'k_values': k_values,
                'use_graded_relevance': use_graded_relevance,
                'evaluation_date': datetime.now().isoformat()
            }
        }

        # Store in history
        self.results_history.append(results)

        return results

    @staticmethod
    def _aggregate_metrics(
        per_query_results: List[Dict],
        k_values: List[int]
    ) -> Dict:
        """Calculate mean metrics across all queries."""
        aggregate = {
            'recall': {k: [] for k in k_values},
            'precision': {k: [] for k in k_values},
            'f1': {k: [] for k in k_values},
            'ndcg': {k: [] for k in k_values},
            'mrr': []
        }

        # Collect all values
        for result in per_query_results:
            metrics = result['metrics']

            for k in k_values:
                aggregate['recall'][k].append(metrics['recall'][k])
                aggregate['precision'][k].append(metrics['precision'][k])
                aggregate['f1'][k].append(metrics['f1'][k])
                aggregate['ndcg'][k].append(metrics['ndcg'][k])

            aggregate['mrr'].append(metrics['mrr'])

        # Calculate means
        aggregated = {
            'recall': {k: np.mean(vals) for k, vals in aggregate['recall'].items()},
            'precision': {k: np.mean(vals) for k, vals in aggregate['precision'].items()},
            'f1': {k: np.mean(vals) for k, vals in aggregate['f1'].items()},
            'ndcg': {k: np.mean(vals) for k, vals in aggregate['ndcg'].items()},
            'mrr': np.mean(aggregate['mrr'])
        }

        return aggregated

    # ==================== Reporting ====================

    @staticmethod
    def generate_report(
        results: Dict,
        title: str = "RAG Evaluation Report"
    ) -> str:
        """
        Generate a Markdown report from evaluation results.

        Args:
            results: Results dictionary from evaluate_dataset()
            title: Report title

        Returns:
            Markdown-formatted report string
        """
        aggregate = results['aggregate']
        metadata = results['metadata']
        k_values = metadata['k_values']

        # Build Markdown report
        lines = []
        lines.append(f"# {title}\n")
        lines.append(f"**Evaluation Date:** {metadata['evaluation_date']}\n")
        lines.append(f"**Number of Queries:** {metadata['num_queries']}\n")
        lines.append(f"**Graded Relevance:** {metadata['use_graded_relevance']}\n")
        lines.append("")

        # Aggregate metrics table
        lines.append("## Aggregate Metrics\n")
        lines.append("| Metric | " + " | ".join([f"@{k}" for k in k_values]) + " |")
        lines.append("|--------|" + "|".join(["-------"] * len(k_values)) + "|")

        # Recall row
        recall_vals = " | ".join([f"{aggregate['recall'][k]:.4f}" for k in k_values])
        lines.append(f"| **Recall** | {recall_vals} |")

        # Precision row
        precision_vals = " | ".join([f"{aggregate['precision'][k]:.4f}" for k in k_values])
        lines.append(f"| **Precision** | {precision_vals} |")

        # F1 row
        f1_vals = " | ".join([f"{aggregate['f1'][k]:.4f}" for k in k_values])
        lines.append(f"| **F1** | {f1_vals} |")

        # NDCG row
        ndcg_vals = " | ".join([f"{aggregate['ndcg'][k]:.4f}" for k in k_values])
        lines.append(f"| **NDCG** | {ndcg_vals} |")

        lines.append("")
        lines.append(f"**MRR:** {aggregate['mrr']:.4f}\n")

        # Quality assessment
        lines.append("## Quality Assessment\n")

        # Use K=10 as reference
        k_ref = 10 if 10 in k_values else k_values[0]

        recall_10 = aggregate['recall'][k_ref]
        precision_10 = aggregate['precision'][k_ref]
        mrr = aggregate['mrr']

        lines.append(f"- **Recall@{k_ref}**: {recall_10:.4f} - " +
                    ("✅ Good" if recall_10 >= 0.70 else "⚠️ Needs improvement"))
        lines.append(f"- **Precision@{k_ref}**: {precision_10:.4f} - " +
                    ("✅ Good" if precision_10 >= 0.40 else "⚠️ Needs improvement"))
        lines.append(f"- **MRR**: {mrr:.4f} - " +
                    ("✅ Good" if mrr >= 0.50 else "⚠️ Needs improvement"))

        return "\n".join(lines)

    @staticmethod
    def export_to_csv(
        results: Dict,
        filepath: Union[str, Path]
    ) -> None:
        """
        Export evaluation results to CSV file.

        Args:
            results: Results dictionary from evaluate_dataset()
            filepath: Path to output CSV file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        k_values = results['metadata']['k_values']

        # Write CSV
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            # Create header
            fieldnames = ['query_id', 'query']
            for k in k_values:
                fieldnames.extend([f'recall@{k}', f'precision@{k}', f'f1@{k}', f'ndcg@{k}'])
            fieldnames.append('mrr')

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            # Write per-query results
            for result in results['per_query']:
                row = {
                    'query_id': result['query_id'],
                    'query': result['query']
                }

                metrics = result['metrics']
                for k in k_values:
                    row[f'recall@{k}'] = f"{metrics['recall'][k]:.4f}"
                    row[f'precision@{k}'] = f"{metrics['precision'][k]:.4f}"
                    row[f'f1@{k}'] = f"{metrics['f1'][k]:.4f}"
                    row[f'ndcg@{k}'] = f"{metrics['ndcg'][k]:.4f}"

                row['mrr'] = f"{metrics['mrr']:.4f}"

                writer.writerow(row)

    @staticmethod
    def export_to_json(
        results: Dict,
        filepath: Union[str, Path],
        indent: int = 2
    ) -> None:
        """
        Export evaluation results to JSON file.

        Args:
            results: Results dictionary from evaluate_dataset()
            filepath: Path to output JSON file
            indent: JSON indentation level
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=indent, ensure_ascii=False)


# ==================== Convenience Functions ====================

def quick_evaluate(
    queries: List[str],
    relevant_docs: List[Set[str]],
    search_function: Callable[[str], List[str]],
    k: int = 10
) -> Dict:
    """
    Quick evaluation for simple use cases.

    Args:
        queries: List of query strings
        relevant_docs: List of relevant document ID sets (parallel to queries)
        search_function: Function that takes query and returns doc IDs
        k: K value for metrics

    Returns:
        Dictionary with average metrics at K
    """
    evaluator = RAGEvaluator()

    dataset = [
        {'query': q, 'relevant_ids': rel}
        for q, rel in zip(queries, relevant_docs)
    ]

    results = evaluator.evaluate_dataset(
        dataset,
        search_function,
        k_values=[k],
        use_graded_relevance=False
    )

    return results['aggregate']
