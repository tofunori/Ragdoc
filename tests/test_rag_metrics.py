#!/usr/bin/env python3
"""
Unit tests for RAG evaluation metrics.

Tests all core metrics with known ground truth examples and edge cases.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import unittest
from rag_evaluator import RAGEvaluator, quick_evaluate


class TestRAGMetrics(unittest.TestCase):
    """Test suite for RAG evaluation metrics."""

    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = RAGEvaluator()

    # ==================== Test Recall@K ====================

    def test_recall_at_k_perfect(self):
        """Test Recall@K with perfect retrieval."""
        retrieved = ['doc1', 'doc2', 'doc3', 'doc4', 'doc5']
        relevant = {'doc1', 'doc2', 'doc3'}

        recall = self.evaluator.recall_at_k(retrieved, relevant, k=5)

        # All 3 relevant docs are in top-5
        self.assertEqual(recall, 1.0)

    def test_recall_at_k_partial(self):
        """Test Recall@K with partial retrieval."""
        retrieved = ['doc1', 'doc2', 'doc3', 'doc4', 'doc5']
        relevant = {'doc2', 'doc4', 'doc7'}

        recall = self.evaluator.recall_at_k(retrieved, relevant, k=5)

        # 2 out of 3 relevant docs retrieved
        self.assertAlmostEqual(recall, 2/3, places=4)

    def test_recall_at_k_zero(self):
        """Test Recall@K with no relevant docs retrieved."""
        retrieved = ['doc1', 'doc2', 'doc3']
        relevant = {'doc5', 'doc6', 'doc7'}

        recall = self.evaluator.recall_at_k(retrieved, relevant, k=3)

        self.assertEqual(recall, 0.0)

    def test_recall_at_k_empty_relevant(self):
        """Test Recall@K with empty relevant set."""
        retrieved = ['doc1', 'doc2', 'doc3']
        relevant = set()

        recall = self.evaluator.recall_at_k(retrieved, relevant, k=3)

        self.assertEqual(recall, 0.0)

    # ==================== Test Precision@K ====================

    def test_precision_at_k_perfect(self):
        """Test Precision@K with perfect precision."""
        retrieved = ['doc1', 'doc2', 'doc3', 'doc4', 'doc5']
        relevant = {'doc1', 'doc2', 'doc3', 'doc4', 'doc5'}

        precision = self.evaluator.precision_at_k(retrieved, relevant, k=5)

        # All 5 retrieved are relevant
        self.assertEqual(precision, 1.0)

    def test_precision_at_k_partial(self):
        """Test Precision@K with partial precision."""
        retrieved = ['doc1', 'doc2', 'doc3', 'doc4', 'doc5']
        relevant = {'doc2', 'doc4'}

        precision = self.evaluator.precision_at_k(retrieved, relevant, k=5)

        # 2 out of 5 retrieved are relevant
        self.assertAlmostEqual(precision, 0.4, places=4)

    def test_precision_at_k_zero(self):
        """Test Precision@K with no relevant docs."""
        retrieved = ['doc1', 'doc2', 'doc3']
        relevant = {'doc5', 'doc6'}

        precision = self.evaluator.precision_at_k(retrieved, relevant, k=3)

        self.assertEqual(precision, 0.0)

    def test_precision_at_k_invalid_k(self):
        """Test Precision@K with k=0."""
        retrieved = ['doc1', 'doc2']
        relevant = {'doc1'}

        precision = self.evaluator.precision_at_k(retrieved, relevant, k=0)

        self.assertEqual(precision, 0.0)

    # ==================== Test MRR ====================

    def test_mrr_first_position(self):
        """Test MRR when first result is relevant."""
        retrieved = ['doc1', 'doc2', 'doc3', 'doc4']
        relevant = {'doc1', 'doc5'}

        mrr = self.evaluator.mean_reciprocal_rank(retrieved, relevant)

        # First doc is relevant: 1/1
        self.assertEqual(mrr, 1.0)

    def test_mrr_third_position(self):
        """Test MRR when third result is relevant."""
        retrieved = ['doc1', 'doc2', 'doc3', 'doc4']
        relevant = {'doc3', 'doc7'}

        mrr = self.evaluator.mean_reciprocal_rank(retrieved, relevant)

        # First relevant at position 3: 1/3
        self.assertAlmostEqual(mrr, 1/3, places=4)

    def test_mrr_no_relevant(self):
        """Test MRR with no relevant docs retrieved."""
        retrieved = ['doc1', 'doc2', 'doc3']
        relevant = {'doc5', 'doc6'}

        mrr = self.evaluator.mean_reciprocal_rank(retrieved, relevant)

        self.assertEqual(mrr, 0.0)

    # ==================== Test DCG and NDCG ====================

    def test_dcg_simple(self):
        """Test DCG calculation with simple example."""
        relevances = [3, 2, 1, 0]
        k = 4

        dcg = self.evaluator.dcg_at_k(relevances, k)

        # DCG = 3/log2(2) + 2/log2(3) + 1/log2(4) + 0/log2(5)
        #     = 3/1 + 2/1.585 + 1/2 + 0
        #     ≈ 3.0 + 1.262 + 0.5 + 0 = 4.762
        self.assertAlmostEqual(dcg, 4.762, places=2)

    def test_dcg_empty(self):
        """Test DCG with empty relevances."""
        relevances = []
        dcg = self.evaluator.dcg_at_k(relevances, k=5)

        self.assertEqual(dcg, 0.0)

    def test_ndcg_perfect(self):
        """Test NDCG with perfect ranking."""
        retrieved = ['doc1', 'doc2', 'doc3', 'doc4']
        relevance = {'doc1': 3, 'doc2': 2, 'doc3': 1, 'doc4': 0}

        ndcg = self.evaluator.ndcg_at_k(retrieved, relevance, k=4)

        # Perfect ranking → NDCG = 1.0
        self.assertAlmostEqual(ndcg, 1.0, places=4)

    def test_ndcg_imperfect(self):
        """Test NDCG with imperfect ranking."""
        retrieved = ['doc1', 'doc2', 'doc3', 'doc4']
        relevance = {'doc1': 0, 'doc2': 3, 'doc3': 2, 'doc4': 1}

        ndcg = self.evaluator.ndcg_at_k(retrieved, relevance, k=4)

        # Imperfect ranking → NDCG < 1.0
        self.assertLess(ndcg, 1.0)
        self.assertGreater(ndcg, 0.0)

    def test_ndcg_no_relevance(self):
        """Test NDCG with no relevant docs."""
        retrieved = ['doc1', 'doc2', 'doc3']
        relevance = {'doc1': 0, 'doc2': 0, 'doc3': 0}

        ndcg = self.evaluator.ndcg_at_k(retrieved, relevance, k=3)

        self.assertEqual(ndcg, 0.0)

    def test_ndcg_missing_relevance(self):
        """Test NDCG when retrieved docs not in relevance dict."""
        retrieved = ['doc1', 'doc2', 'doc3']
        relevance = {'doc5': 3, 'doc6': 2}  # Retrieved docs not in dict

        ndcg = self.evaluator.ndcg_at_k(retrieved, relevance, k=3)

        # Should handle gracefully (treat unknown docs as 0 relevance)
        self.assertEqual(ndcg, 0.0)

    # ==================== Test F1@K ====================

    def test_f1_at_k_balanced(self):
        """Test F1@K with balanced precision and recall."""
        retrieved = ['doc1', 'doc2', 'doc3', 'doc4']
        relevant = {'doc1', 'doc2', 'doc5', 'doc6'}

        f1 = self.evaluator.f1_at_k(retrieved, relevant, k=4)

        # Precision = 2/4 = 0.5
        # Recall = 2/4 = 0.5
        # F1 = 2 * 0.5 * 0.5 / (0.5 + 0.5) = 0.5
        self.assertAlmostEqual(f1, 0.5, places=4)

    def test_f1_at_k_zero(self):
        """Test F1@K when no relevant docs retrieved."""
        retrieved = ['doc1', 'doc2', 'doc3']
        relevant = {'doc5', 'doc6'}

        f1 = self.evaluator.f1_at_k(retrieved, relevant, k=3)

        self.assertEqual(f1, 0.0)

    # ==================== Test evaluate_query ====================

    def test_evaluate_query_binary(self):
        """Test evaluate_query with binary relevance."""
        retrieved = ['doc1', 'doc2', 'doc3', 'doc4', 'doc5']
        relevant = {'doc2', 'doc3', 'doc5'}

        metrics = self.evaluator.evaluate_query(
            retrieved,
            relevant,
            k_values=[3, 5],
            use_graded_relevance=False
        )

        # Check structure
        self.assertIn('recall', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('mrr', metrics)
        self.assertIn('ndcg', metrics)
        self.assertIn('f1', metrics)

        # Check K=3 metrics
        self.assertAlmostEqual(metrics['recall'][3], 2/3, places=4)  # 2 out of 3 relevant
        self.assertAlmostEqual(metrics['precision'][3], 2/3, places=4)  # 2 out of 3 retrieved

        # Check K=5 metrics
        self.assertEqual(metrics['recall'][5], 1.0)  # All 3 relevant found
        self.assertAlmostEqual(metrics['precision'][5], 3/5, places=4)  # 3 out of 5 relevant

        # Check MRR
        self.assertAlmostEqual(metrics['mrr'], 0.5, places=4)  # First relevant at position 2

    def test_evaluate_query_graded(self):
        """Test evaluate_query with graded relevance."""
        retrieved = ['doc1', 'doc2', 'doc3', 'doc4']
        relevance = {'doc1': 0, 'doc2': 3, 'doc3': 2, 'doc4': 1}

        metrics = self.evaluator.evaluate_query(
            retrieved,
            relevance,
            k_values=[3],
            use_graded_relevance=True
        )

        # NDCG should use graded relevance
        self.assertGreater(metrics['ndcg'][3], 0.0)
        self.assertLess(metrics['ndcg'][3], 1.0)

    # ==================== Test evaluate_dataset ====================

    def test_evaluate_dataset_simple(self):
        """Test evaluate_dataset with simple search function."""
        # Mock search function
        def mock_search(query):
            """Return predefined results based on query."""
            if 'query1' in query:
                return ['doc1', 'doc2', 'doc3']
            elif 'query2' in query:
                return ['doc4', 'doc5', 'doc6']
            else:
                return []

        # Test dataset
        dataset = [
            {
                'query': 'test query1',
                'relevant_ids': {'doc1', 'doc2'},
                'query_id': 'q1'
            },
            {
                'query': 'test query2',
                'relevant_ids': {'doc4', 'doc7'},
                'query_id': 'q2'
            }
        ]

        results = self.evaluator.evaluate_dataset(
            dataset,
            mock_search,
            k_values=[3],
            use_graded_relevance=False
        )

        # Check structure
        self.assertIn('aggregate', results)
        self.assertIn('per_query', results)
        self.assertIn('metadata', results)

        # Check metadata
        self.assertEqual(results['metadata']['num_queries'], 2)

        # Check aggregate metrics exist
        self.assertIn('recall', results['aggregate'])
        self.assertIn('precision', results['aggregate'])
        self.assertIn('mrr', results['aggregate'])

        # Check per-query results
        self.assertEqual(len(results['per_query']), 2)
        self.assertEqual(results['per_query'][0]['query_id'], 'q1')
        self.assertEqual(results['per_query'][1]['query_id'], 'q2')

    # ==================== Test Reporting ====================

    def test_generate_report(self):
        """Test Markdown report generation."""
        # Create mock results
        results = {
            'aggregate': {
                'recall': {10: 0.75, 20: 0.85},
                'precision': {10: 0.50, 20: 0.40},
                'f1': {10: 0.60, 20: 0.55},
                'ndcg': {10: 0.70, 20: 0.75},
                'mrr': 0.65
            },
            'per_query': [],
            'metadata': {
                'num_queries': 30,
                'k_values': [10, 20],
                'use_graded_relevance': False,
                'evaluation_date': '2025-11-15T10:00:00'
            }
        }

        report = self.evaluator.generate_report(results, title="Test Report")

        # Check report contains key elements
        self.assertIn("Test Report", report)
        self.assertIn("Aggregate Metrics", report)
        self.assertIn("0.7500", report)  # Recall@10
        self.assertIn("0.6500", report)  # MRR
        self.assertIn("Quality Assessment", report)

    # ==================== Test Quick Evaluate ====================

    def test_quick_evaluate(self):
        """Test convenience function quick_evaluate."""
        queries = ['query1', 'query2']
        relevant = [{'doc1', 'doc2'}, {'doc3', 'doc4'}]

        def mock_search(query):
            if 'query1' in query:
                return ['doc1', 'doc3', 'doc5']
            else:
                return ['doc3', 'doc4', 'doc6']

        results = quick_evaluate(queries, relevant, mock_search, k=3)

        # Check results structure
        self.assertIn('recall', results)
        self.assertIn('precision', results)
        self.assertIn('mrr', results)

        # Check values are in expected range
        self.assertGreaterEqual(results['recall'][3], 0.0)
        self.assertLessEqual(results['recall'][3], 1.0)

    # ==================== Edge Cases ====================

    def test_empty_retrieved(self):
        """Test metrics with empty retrieved list."""
        retrieved = []
        relevant = {'doc1', 'doc2'}

        recall = self.evaluator.recall_at_k(retrieved, relevant, k=10)
        precision = self.evaluator.precision_at_k(retrieved, relevant, k=10)
        mrr = self.evaluator.mean_reciprocal_rank(retrieved, relevant)

        self.assertEqual(recall, 0.0)
        self.assertEqual(precision, 0.0)
        self.assertEqual(mrr, 0.0)

    def test_large_k(self):
        """Test metrics when K is larger than retrieved list."""
        retrieved = ['doc1', 'doc2', 'doc3']
        relevant = {'doc1', 'doc2', 'doc4'}

        # K=10 but only 3 docs retrieved
        recall = self.evaluator.recall_at_k(retrieved, relevant, k=10)
        precision = self.evaluator.precision_at_k(retrieved, relevant, k=10)

        # Recall should still work (2 out of 3 relevant found)
        self.assertAlmostEqual(recall, 2/3, places=4)

        # Precision uses K=10 in denominator
        self.assertAlmostEqual(precision, 2/10, places=4)


def run_validation_tests():
    """Run validation tests and print summary."""
    print("=" * 80)
    print("RAGDOC METRICS VALIDATION TESTS")
    print("=" * 80)
    print("\nRunning comprehensive test suite for RAG evaluation metrics...\n")

    # Run tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRAGMetrics)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED - Metrics implementation validated!")
    else:
        print("\n❌ SOME TESTS FAILED - Please review errors above")

    print("=" * 80 + "\n")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_validation_tests()
    sys.exit(0 if success else 1)
