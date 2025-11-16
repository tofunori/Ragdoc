#!/usr/bin/env python3
"""
Compare Simple vs Advanced Tokenization on Real Corpus.

Runs RAGDOC evaluation twice:
1. With simple tokenization (v1.4.0 baseline)
2. With advanced tokenization (v1.5.0)

Generates comparative report showing real improvement metrics.

Usage:
    python scripts/compare_tokenization.py
    python scripts/compare_tokenization.py --dataset tests/test_datasets/custom.json
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))

import argparse
import json
import time
from datetime import datetime
from typing import List, Dict

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import RAGDOC components
import server
from rag_evaluator import RAGEvaluator


def load_test_dataset(dataset_path: str) -> Dict:
    """Load test dataset from JSON file."""
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    print(f"[OK] Loaded dataset: {dataset['name']}")
    print(f"  Queries: {dataset['num_queries']}")

    return dataset


def create_search_function(alpha: float, use_advanced_tokenizer: bool):
    """
    Create search function with specified tokenization mode.

    Args:
        alpha: Semantic weight (0.0-1.0)
        use_advanced_tokenizer: True for advanced, False for simple

    Returns:
        Search function
    """
    def search(query: str) -> List[str]:
        """Execute hybrid search."""
        # Temporarily set tokenizer mode
        original_tokenizer = server.hybrid_retriever.tokenizer

        if use_advanced_tokenizer:
            # Ensure advanced tokenizer is enabled
            if not server.hybrid_retriever.tokenizer:
                from bm25_tokenizers import AdvancedTokenizer
                server.hybrid_retriever.tokenizer = AdvancedTokenizer()
        else:
            # Disable advanced tokenizer for baseline
            server.hybrid_retriever.tokenizer = None

        try:
            # Run search
            results = server.hybrid_retriever.search(
                query=query,
                top_k=50,
                alpha=alpha,
                bm25_top_n=100,
                semantic_top_n=100
            )

            # Extract IDs
            retrieved_ids = [r['id'] for r in results]

            return retrieved_ids

        finally:
            # Restore original tokenizer
            server.hybrid_retriever.tokenizer = original_tokenizer

    return search


def run_tokenization_evaluation(
    dataset: Dict,
    use_advanced_tokenizer: bool,
    alpha: float = 0.5,
    k_values: List[int] = [5, 10, 20]
) -> Dict:
    """
    Run evaluation with specified tokenization mode.

    Args:
        dataset: Test dataset
        use_advanced_tokenizer: True for advanced, False for simple
        alpha: Semantic weight
        k_values: K values for metrics

    Returns:
        Evaluation results
    """
    mode_name = "Advanced Tokenization (v1.5.0)" if use_advanced_tokenizer else "Simple Tokenization (v1.4.0)"
    print(f"\n{'='*80}")
    print(f"EVALUATING: {mode_name}")
    print(f"{'='*80}")

    # Create evaluator
    evaluator = RAGEvaluator()

    # Create search function
    search_fn = create_search_function(alpha, use_advanced_tokenizer)

    # Prepare dataset
    eval_dataset = []
    for query_data in dataset['queries']:
        eval_dataset.append({
            'query': query_data['query'],
            'relevant_ids': set(query_data['relevant_chunks']),
            'query_id': query_data['id']
        })

    # Run evaluation
    print(f"\n[*] Running evaluation on {len(eval_dataset)} queries...")
    start_time = time.time()

    results = evaluator.evaluate_dataset(
        dataset=eval_dataset,
        search_function=search_fn,
        k_values=k_values,
        use_graded_relevance=False
    )

    evaluation_time = time.time() - start_time

    # Add configuration
    results['config'] = {
        'tokenization': 'advanced' if use_advanced_tokenizer else 'simple',
        'alpha': alpha,
        'evaluation_time': evaluation_time
    }

    # Print summary
    print(f"\n[OK] Evaluation complete in {evaluation_time:.2f}s")
    print(f"\nResults:")
    print(f"  Recall@10:    {results['aggregate']['recall'][10]:.4f}")
    print(f"  Precision@10: {results['aggregate']['precision'][10]:.4f}")
    print(f"  F1@10:        {results['aggregate']['f1'][10]:.4f}")
    print(f"  MRR:          {results['aggregate']['mrr']:.4f}")
    print(f"  NDCG@10:      {results['aggregate']['ndcg'][10]:.4f}")

    return results


def generate_comparison_report(
    simple_results: Dict,
    advanced_results: Dict,
    k_values: List[int]
) -> str:
    """
    Generate comparison report.

    Args:
        simple_results: Baseline results
        advanced_results: Advanced tokenization results
        k_values: K values

    Returns:
        Markdown report
    """
    lines = []
    lines.append("# Tokenization Comparison Report - RAGDOC v1.5.0\n")
    lines.append(f"**Evaluation Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"**Number of Queries:** {simple_results['metadata']['num_queries']}\n")

    # Overall comparison
    lines.append("## Overall Comparison\n")
    lines.append("| Metric       | Simple (v1.4) | Advanced (v1.5) | Improvement |")
    lines.append("|--------------|---------------|-----------------|-------------|")

    simple_agg = simple_results['aggregate']
    advanced_agg = advanced_results['aggregate']

    metrics = [
        ('Recall@10', 'recall', 10),
        ('Precision@10', 'precision', 10),
        ('F1@10', 'f1', 10),
        ('MRR', 'mrr', None),
        ('NDCG@10', 'ndcg', 10),
    ]

    for label, metric_key, k in metrics:
        if k is not None:
            simple_val = simple_agg[metric_key][k]
            advanced_val = advanced_agg[metric_key][k]
        else:
            simple_val = simple_agg[metric_key]
            advanced_val = advanced_agg[metric_key]

        improvement = ((advanced_val - simple_val) / simple_val * 100) if simple_val > 0 else 0
        improvement_str = f"+{improvement:.1f}%" if improvement > 0 else f"{improvement:.1f}%"

        # Add emoji for significant improvements
        if improvement > 10:
            improvement_str += " [OK]"
        elif improvement > 5:
            improvement_str += " [+]"
        elif improvement < -5:
            improvement_str += " [WARNING]"

        lines.append(f"| {label:<12} | {simple_val:.4f} | {advanced_val:.4f} | {improvement_str} |")

    lines.append("")

    # Performance time
    simple_time = simple_results['config']['evaluation_time']
    advanced_time = advanced_results['config']['evaluation_time']
    time_overhead = ((advanced_time - simple_time) / simple_time * 100) if simple_time > 0 else 0

    lines.append("## Performance\n")
    lines.append(f"- **Simple Tokenization:** {simple_time:.2f}s")
    lines.append(f"- **Advanced Tokenization:** {advanced_time:.2f}s")
    lines.append(f"- **Time Overhead:** +{time_overhead:.1f}%\n")

    # Detailed metrics by K
    lines.append("## Detailed Metrics by K\n")

    for k in k_values:
        lines.append(f"### K = {k}\n")
        lines.append("| Metric    | Simple | Advanced | Improvement |")
        lines.append("|-----------|--------|----------|-------------|")

        for metric_key in ['recall', 'precision', 'f1', 'ndcg']:
            simple_val = simple_agg[metric_key][k]
            advanced_val = advanced_agg[metric_key][k]
            improvement = ((advanced_val - simple_val) / simple_val * 100) if simple_val > 0 else 0
            improvement_str = f"+{improvement:.1f}%" if improvement >= 0 else f"{improvement:.1f}%"

            lines.append(f"| {metric_key.capitalize():<9} | {simple_val:.4f} | {advanced_val:.4f} | {improvement_str} |")

        lines.append("")

    # Analysis
    lines.append("## Analysis\n")

    recall_improvement = ((advanced_agg['recall'][10] - simple_agg['recall'][10]) /
                          simple_agg['recall'][10] * 100) if simple_agg['recall'][10] > 0 else 0
    precision_improvement = ((advanced_agg['precision'][10] - simple_agg['precision'][10]) /
                             simple_agg['precision'][10] * 100) if simple_agg['precision'][10] > 0 else 0

    if recall_improvement >= 10:
        lines.append(f"[OK] **Significant recall improvement** (+{recall_improvement:.1f}%): "
                    "Advanced tokenization successfully finds more relevant documents through stemming.")
    elif recall_improvement >= 5:
        lines.append(f"[+] **Moderate recall improvement** (+{recall_improvement:.1f}%): "
                    "Stemming helps match word variations.")
    elif recall_improvement > 0:
        lines.append(f"[+] **Modest recall improvement** (+{recall_improvement:.1f}%): "
                    "Some benefit from advanced tokenization.")
    else:
        lines.append(f"[WARNING] **No recall improvement** ({recall_improvement:.1f}%): "
                    "Advanced tokenization did not improve recall on this dataset.")

    lines.append("")

    if precision_improvement >= 5:
        lines.append(f"[OK] **Precision improved** (+{precision_improvement:.1f}%): "
                    "Compound terms reduce false positives.")
    elif precision_improvement > 0:
        lines.append(f"[+] **Slight precision improvement** (+{precision_improvement:.1f}%)")
    else:
        lines.append(f"[INFO] Precision: {precision_improvement:.1f}%")

    lines.append("")

    # Recommendations
    lines.append("## Recommendation\n")

    if recall_improvement >= 5 and time_overhead < 50:
        lines.append("[OK] **Deploy advanced tokenization**")
        lines.append(f"\nReasons:")
        lines.append(f"- Recall improvement: +{recall_improvement:.1f}%")
        lines.append(f"- Precision improvement: +{precision_improvement:.1f}%")
        lines.append(f"- Acceptable time overhead: +{time_overhead:.1f}%")
        lines.append(f"\nAdvanced tokenization provides measurable quality improvement "
                    "with acceptable performance cost.")
    elif recall_improvement > 0:
        lines.append("[+] **Consider deploying advanced tokenization**")
        lines.append(f"\nModerate improvement (+{recall_improvement:.1f}% recall) "
                    f"with {time_overhead:.1f}% time overhead. "
                    "Test on larger query sample before full deployment.")
    else:
        lines.append("[INFO] **Re-evaluate**")
        lines.append(f"\nAdvanced tokenization showed minimal improvement on this test set. "
                    "Consider testing with a larger, more diverse query set or investigating "
                    "why stemming/compounds didn't help.")

    return "\n".join(lines)


def save_comparison_results(
    simple_results: Dict,
    advanced_results: Dict,
    k_values: List[int],
    output_dir: str = "tests/results"
):
    """Save comparison results to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save comparison report
    report = generate_comparison_report(simple_results, advanced_results, k_values)
    report_path = output_dir / f"tokenization_comparison_{timestamp}.md"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n[OK] Comparison report saved: {report_path}")

    # Save detailed JSON
    comparison_data = {
        'timestamp': timestamp,
        'simple_tokenization': simple_results,
        'advanced_tokenization': advanced_results,
    }

    json_path = output_dir / f"tokenization_comparison_{timestamp}.json"

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, indent=2, ensure_ascii=False)

    print(f"[OK] Detailed results saved: {json_path}")

    # Copy to latest
    try:
        import shutil
        latest_report = output_dir / "tokenization_comparison_latest.md"
        latest_json = output_dir / "tokenization_comparison_latest.json"

        shutil.copy(report_path, latest_report)
        shutil.copy(json_path, latest_json)

        print(f"[OK] Latest results copied")
    except Exception as e:
        print(f"[WARNING] Could not create latest files: {e}")

    return report_path


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compare simple vs advanced tokenization on real corpus"
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default='tests/test_datasets/synthetic_ragdoc_qa.json',
        help='Path to test dataset JSON file'
    )

    parser.add_argument(
        '--alpha',
        type=float,
        default=0.5,
        help='Alpha value for hybrid search (default: 0.5)'
    )

    parser.add_argument(
        '--k_values',
        type=int,
        nargs='+',
        default=[5, 10, 20],
        help='K values for metrics (space-separated)'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='tests/results',
        help='Output directory for results'
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("RAGDOC TOKENIZATION COMPARISON")
    print("=" * 80)
    print(f"\nComparing Simple (v1.4) vs Advanced (v1.5) Tokenization")
    print(f"Dataset: {args.dataset}")
    print(f"Alpha: {args.alpha}")
    print(f"K values: {args.k_values}\n")

    # Load dataset
    dataset = load_test_dataset(args.dataset)

    # Initialize server
    print("\n[*] Initializing RAGDOC components...")
    server.init_clients()
    print("[OK] Server initialized\n")

    # Run baseline evaluation (simple tokenization)
    simple_results = run_tokenization_evaluation(
        dataset=dataset,
        use_advanced_tokenizer=False,
        alpha=args.alpha,
        k_values=args.k_values
    )

    # Run advanced tokenization evaluation
    advanced_results = run_tokenization_evaluation(
        dataset=dataset,
        use_advanced_tokenizer=True,
        alpha=args.alpha,
        k_values=args.k_values
    )

    # Save comparison
    report_path = save_comparison_results(
        simple_results,
        advanced_results,
        args.k_values,
        args.output_dir
    )

    # Print summary
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)

    simple_agg = simple_results['aggregate']
    advanced_agg = advanced_results['aggregate']

    recall_improvement = ((advanced_agg['recall'][10] - simple_agg['recall'][10]) /
                          simple_agg['recall'][10] * 100) if simple_agg['recall'][10] > 0 else 0
    precision_improvement = ((advanced_agg['precision'][10] - simple_agg['precision'][10]) /
                             simple_agg['precision'][10] * 100) if simple_agg['precision'][10] > 0 else 0

    print(f"\nRecall@10 improvement:    {recall_improvement:+.1f}%")
    print(f"Precision@10 improvement: {precision_improvement:+.1f}%")
    print(f"\nFull report: {report_path}")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
