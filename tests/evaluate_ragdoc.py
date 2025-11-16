#!/usr/bin/env python3
"""
RAGDOC Evaluation Runner

Comprehensive evaluation of RAGDOC retrieval quality using synthetic test sets.

Evaluates multiple configurations:
- Different alpha values (BM25 vs Semantic weight)
- Different K values (top-5, top-10, top-20)
- Pure BM25 vs Pure Semantic vs Hybrid

Outputs:
- Markdown summary reports
- Detailed CSV results
- JSON export for further analysis

Usage:
    python tests/evaluate_ragdoc.py
    python tests/evaluate_ragdoc.py --dataset tests/test_datasets/custom.json
    python tests/evaluate_ragdoc.py --alpha 0.3 0.5 0.7 1.0
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import json
import time
from datetime import datetime
from typing import List, Dict, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import RAGDOC components
import server
from rag_evaluator import RAGEvaluator


def load_test_dataset(dataset_path: str) -> Dict:
    """
    Load test dataset from JSON file.

    Args:
        dataset_path: Path to dataset JSON file

    Returns:
        Dataset dictionary
    """
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    print(f"[OK] Loaded dataset: {dataset['name']}")
    print(f"  Queries: {dataset['num_queries']}")
    print(f"  Strategy: {dataset['strategy']}")

    return dataset


def create_search_function(alpha: float):
    """
    Create a search function with specified alpha value.

    Args:
        alpha: Semantic weight (0.0-1.0)

    Returns:
        Search function that takes query string and returns doc IDs
    """
    def search(query: str) -> List[str]:
        """
        Execute hybrid search with specified alpha.

        Args:
            query: Query string

        Returns:
            List of retrieved chunk IDs (ordered by relevance)
        """
        # Use hybrid retriever
        results = server.hybrid_retriever.search(
            query=query,
            top_k=50,  # Get more results for evaluation
            alpha=alpha,
            bm25_top_n=100,
            semantic_top_n=100
        )

        # Extract IDs from results
        retrieved_ids = [r['id'] for r in results]

        return retrieved_ids

    return search


def run_single_evaluation(
    dataset: Dict,
    alpha: float,
    k_values: List[int],
    evaluator: RAGEvaluator
) -> Dict:
    """
    Run evaluation for a single alpha configuration.

    Args:
        dataset: Test dataset
        alpha: Semantic weight
        k_values: List of K values to evaluate
        evaluator: RAG evaluator instance

    Returns:
        Evaluation results dictionary
    """
    print(f"\n  Evaluating alpha={alpha:.2f}...")

    # Create search function for this alpha
    search_fn = create_search_function(alpha)

    # Prepare dataset for evaluator
    # Convert from dataset format to evaluator format
    eval_dataset = []
    for query_data in dataset['queries']:
        eval_dataset.append({
            'query': query_data['query'],
            'relevant_ids': set(query_data['relevant_chunks']),  # Use chunk IDs as relevant
            'query_id': query_data['id']
        })

    # Run evaluation
    start_time = time.time()

    results = evaluator.evaluate_dataset(
        dataset=eval_dataset,
        search_function=search_fn,
        k_values=k_values,
        use_graded_relevance=False
    )

    evaluation_time = time.time() - start_time

    # Add configuration info
    results['config'] = {
        'alpha': alpha,
        'search_mode': 'pure_bm25' if alpha == 0.0 else 'pure_semantic' if alpha == 1.0 else 'hybrid',
        'evaluation_time': evaluation_time
    }

    print(f"    Completed in {evaluation_time:.2f}s")
    print(f"    Recall@10: {results['aggregate']['recall'][10]:.4f}")
    print(f"    Precision@10: {results['aggregate']['precision'][10]:.4f}")
    print(f"    MRR: {results['aggregate']['mrr']:.4f}")

    return results


def run_multi_alpha_evaluation(
    dataset: Dict,
    alpha_values: List[float],
    k_values: List[int] = [5, 10, 20]
) -> List[Dict]:
    """
    Run evaluation across multiple alpha values.

    Args:
        dataset: Test dataset
        alpha_values: List of alpha values to test
        k_values: List of K values to evaluate

    Returns:
        List of evaluation results (one per alpha)
    """
    print("=" * 80)
    print("RAGDOC MULTI-CONFIGURATION EVALUATION")
    print("=" * 80)
    print(f"\nDataset: {dataset['name']}")
    print(f"Queries: {dataset['num_queries']}")
    print(f"Alpha values: {alpha_values}")
    print(f"K values: {k_values}\n")

    # Initialize server
    print("[*] Initializing RAGDOC components...")
    server.init_clients()
    print("[OK] Server initialized\n")

    # Create evaluator
    evaluator = RAGEvaluator()

    # Run evaluations
    all_results = []

    for alpha in alpha_values:
        results = run_single_evaluation(
            dataset,
            alpha,
            k_values,
            evaluator
        )
        all_results.append(results)

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80 + "\n")

    return all_results


def compare_configurations(all_results: List[Dict], k_values: List[int]) -> str:
    """
    Generate comparative analysis across configurations.

    Args:
        all_results: List of evaluation results
        k_values: K values that were evaluated

    Returns:
        Markdown-formatted comparison report
    """
    lines = []
    lines.append("# RAGDOC Configuration Comparison\n")
    lines.append(f"**Evaluation Date:** {datetime.now().isoformat()}\n")
    lines.append(f"**Number of Queries:** {all_results[0]['metadata']['num_queries']}\n")
    lines.append("")

    # Configuration comparison table
    lines.append("## Performance by Alpha Value\n")
    lines.append("| Alpha | Mode | Recall@10 | Precision@10 | MRR | NDCG@10 | F1@10 | Time(s) |")
    lines.append("|-------|------|-----------|--------------|-----|---------|-------|---------|")

    for result in all_results:
        alpha = result['config']['alpha']
        mode = result['config']['search_mode']
        agg = result['aggregate']
        time_taken = result['config']['evaluation_time']

        recall_10 = agg['recall'][10]
        precision_10 = agg['precision'][10]
        mrr = agg['mrr']
        ndcg_10 = agg['ndcg'][10]
        f1_10 = agg['f1'][10]

        lines.append(
            f"| {alpha:.2f} | {mode:13s} | {recall_10:.4f} | {precision_10:.4f} | "
            f"{mrr:.4f} | {ndcg_10:.4f} | {f1_10:.4f} | {time_taken:.2f} |"
        )

    lines.append("")

    # Find best configuration
    best_recall_idx = max(range(len(all_results)),
                          key=lambda i: all_results[i]['aggregate']['recall'][10])
    best_precision_idx = max(range(len(all_results)),
                             key=lambda i: all_results[i]['aggregate']['precision'][10])
    best_mrr_idx = max(range(len(all_results)),
                      key=lambda i: all_results[i]['aggregate']['mrr'])

    lines.append("## Best Configurations\n")
    lines.append(f"- **Best Recall@10:** alpha={all_results[best_recall_idx]['config']['alpha']:.2f} "
                f"({all_results[best_recall_idx]['aggregate']['recall'][10]:.4f})")
    lines.append(f"- **Best Precision@10:** alpha={all_results[best_precision_idx]['config']['alpha']:.2f} "
                f"({all_results[best_precision_idx]['aggregate']['precision'][10]:.4f})")
    lines.append(f"- **Best MRR:** alpha={all_results[best_mrr_idx]['config']['alpha']:.2f} "
                f"({all_results[best_mrr_idx]['aggregate']['mrr']:.4f})\n")

    # Detailed metrics by K
    for k in k_values:
        lines.append(f"## Metrics @ K={k}\n")
        lines.append("| Alpha | Recall | Precision | F1 | NDCG |")
        lines.append("|-------|--------|-----------|-----|------|")

        for result in all_results:
            alpha = result['config']['alpha']
            agg = result['aggregate']

            lines.append(
                f"| {alpha:.2f} | {agg['recall'][k]:.4f} | {agg['precision'][k]:.4f} | "
                f"{agg['f1'][k]:.4f} | {agg['ndcg'][k]:.4f} |"
            )

        lines.append("")

    # Recommendations
    lines.append("## Recommendations\n")

    hybrid_results = [r for r in all_results if 0.3 <= r['config']['alpha'] <= 0.9]
    if hybrid_results:
        best_hybrid = max(hybrid_results, key=lambda r: r['aggregate']['recall'][10])
        best_alpha = best_hybrid['config']['alpha']

        lines.append(f"Based on this evaluation, **alpha={best_alpha:.2f}** provides the best balance:\n")
        lines.append(f"- Recall@10: {best_hybrid['aggregate']['recall'][10]:.4f}")
        lines.append(f"- Precision@10: {best_hybrid['aggregate']['precision'][10]:.4f}")
        lines.append(f"- MRR: {best_hybrid['aggregate']['mrr']:.4f}\n")

        if best_alpha < 0.5:
            lines.append("This configuration favors **BM25 (lexical matching)**, "
                        "suggesting queries benefit from exact term matching.")
        elif best_alpha > 0.7:
            lines.append("This configuration favors **semantic search**, "
                        "suggesting queries benefit from conceptual understanding.")
        else:
            lines.append("This configuration provides **balanced hybrid search**, "
                        "combining lexical and semantic strengths.")

    return "\n".join(lines)


def save_results(
    all_results: List[Dict],
    output_dir: str = "tests/results"
):
    """
    Save evaluation results to files.

    Args:
        all_results: List of evaluation results
        output_dir: Output directory path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Save comparison report (Markdown)
    comparison = compare_configurations(all_results, all_results[0]['metadata']['k_values'])
    report_path = output_dir / f"evaluation_report_{timestamp}.md"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(comparison)

    print(f"[OK] Comparison report saved: {report_path}")

    # 2. Save detailed results (JSON)
    json_path = output_dir / f"evaluation_detailed_{timestamp}.json"

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"[OK] Detailed results saved: {json_path}")

    # 3. Save aggregate metrics CSV
    csv_path = output_dir / f"evaluation_aggregate_{timestamp}.csv"

    import csv
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        k_values = all_results[0]['metadata']['k_values']

        fieldnames = ['alpha', 'mode']
        for k in k_values:
            fieldnames.extend([f'recall@{k}', f'precision@{k}', f'f1@{k}', f'ndcg@{k}'])
        fieldnames.extend(['mrr', 'time_seconds'])

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in all_results:
            row = {
                'alpha': result['config']['alpha'],
                'mode': result['config']['search_mode'],
                'mrr': f"{result['aggregate']['mrr']:.4f}",
                'time_seconds': f"{result['config']['evaluation_time']:.2f}"
            }

            agg = result['aggregate']
            for k in k_values:
                row[f'recall@{k}'] = f"{agg['recall'][k]:.4f}"
                row[f'precision@{k}'] = f"{agg['precision'][k]:.4f}"
                row[f'f1@{k}'] = f"{agg['f1'][k]:.4f}"
                row[f'ndcg@{k}'] = f"{agg['ndcg'][k]:.4f}"

            writer.writerow(row)

    print(f"[OK] Aggregate CSV saved: {csv_path}")

    # Create latest symlinks (for convenience)
    latest_report = output_dir / "evaluation_report_latest.md"
    latest_json = output_dir / "evaluation_detailed_latest.json"
    latest_csv = output_dir / "evaluation_aggregate_latest.csv"

    try:
        # On Windows, copy instead of symlink
        import shutil
        shutil.copy(report_path, latest_report)
        shutil.copy(json_path, latest_json)
        shutil.copy(csv_path, latest_csv)
        print(f"[OK] Latest results copied to *_latest.* files")
    except Exception as e:
        print(f"[WARNING] Could not create latest files: {e}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate RAGDOC retrieval quality"
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
        nargs='+',
        default=[0.3, 0.5, 0.7, 1.0],
        help='Alpha values to test (space-separated)'
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

    # Load dataset
    dataset = load_test_dataset(args.dataset)

    # Run evaluations
    all_results = run_multi_alpha_evaluation(
        dataset,
        alpha_values=args.alpha,
        k_values=args.k_values
    )

    # Save results
    save_results(all_results, args.output_dir)

    # Print quick summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nEvaluated {len(all_results)} configurations on {dataset['num_queries']} queries")
    print(f"\nResults saved to: {args.output_dir}/")
    print(f"  - evaluation_report_latest.md   (comparison report)")
    print(f"  - evaluation_detailed_latest.json (full results)")
    print(f"  - evaluation_aggregate_latest.csv (metrics table)")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
