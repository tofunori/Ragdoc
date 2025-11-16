#!/usr/bin/env python3
"""
Generate synthetic test dataset for RAGDOC evaluation.

Approach: "Self-Retrieval" Testing
- Sample diverse chunks from ChromaDB
- Use chunk text as query
- Ground truth: source document should be retrieved in top-K

This creates a challenging but realistic test set where the system
must retrieve the correct source document given a passage from it.

Usage:
    python scripts/generate_test_dataset.py --n_queries 30 --output test_datasets/synthetic.json
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import json
import random
from datetime import datetime
from typing import List, Dict, Set
import chromadb
from collections import defaultdict, Counter


def initialize_chromadb(db_path: str, collection_name: str) -> chromadb.Collection:
    """
    Initialize connection to ChromaDB.

    Args:
        db_path: Path to ChromaDB directory
        collection_name: Name of the collection

    Returns:
        ChromaDB collection object
    """
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(name=collection_name)

    print(f"[OK] Connected to ChromaDB: {collection_name}")
    print(f"  Total chunks: {collection.count()}")

    return collection


def sample_diverse_chunks(
    collection: chromadb.Collection,
    n_samples: int = 50,
    min_chunk_length: int = 200,
    max_chunk_length: int = 800,
    diversity_mode: str = "balanced"
) -> List[Dict]:
    """
    Sample diverse chunks from ChromaDB collection.

    Args:
        collection: ChromaDB collection
        n_samples: Number of chunks to sample
        min_chunk_length: Minimum chunk text length
        max_chunk_length: Maximum chunk text length
        diversity_mode: Sampling strategy
            - "balanced": Equal chunks per document
            - "random": Pure random sampling
            - "mixed": Combination of both

    Returns:
        List of sampled chunks with metadata
    """
    print(f"\n[*] Sampling {n_samples} diverse chunks...")

    # Get all data
    all_data = collection.get(include=["documents", "metadatas"])

    # Filter by chunk length
    valid_chunks = []
    for i, (doc, metadata) in enumerate(zip(all_data['documents'], all_data['metadatas'])):
        chunk_length = len(doc)
        if min_chunk_length <= chunk_length <= max_chunk_length:
            valid_chunks.append({
                'index': i,
                'id': all_data['ids'][i],
                'text': doc,
                'metadata': metadata,
                'length': chunk_length
            })

    print(f"  Valid chunks (length {min_chunk_length}-{max_chunk_length}): {len(valid_chunks)}")

    # Group by source document
    chunks_by_source = defaultdict(list)
    for chunk in valid_chunks:
        source = chunk['metadata']['source']
        chunks_by_source[source].append(chunk)

    print(f"  Documents represented: {len(chunks_by_source)}")

    # Sample based on strategy
    sampled_chunks = []

    if diversity_mode == "balanced":
        # Ensure each document contributes equally
        docs = list(chunks_by_source.keys())
        chunks_per_doc = max(1, n_samples // len(docs))

        for doc in docs:
            available = chunks_by_source[doc]
            sample_size = min(chunks_per_doc, len(available))
            samples = random.sample(available, sample_size)
            sampled_chunks.extend(samples)

            if len(sampled_chunks) >= n_samples:
                break

        # Fill remaining if needed
        if len(sampled_chunks) < n_samples:
            remaining = n_samples - len(sampled_chunks)
            all_remaining = [c for c in valid_chunks if c not in sampled_chunks]
            sampled_chunks.extend(random.sample(all_remaining, min(remaining, len(all_remaining))))

    elif diversity_mode == "random":
        # Pure random sampling
        sampled_chunks = random.sample(valid_chunks, min(n_samples, len(valid_chunks)))

    else:  # mixed
        # 70% balanced, 30% random
        balanced_count = int(n_samples * 0.7)
        random_count = n_samples - balanced_count

        # Balanced part
        docs = list(chunks_by_source.keys())
        chunks_per_doc = max(1, balanced_count // len(docs))

        for doc in docs:
            available = chunks_by_source[doc]
            sample_size = min(chunks_per_doc, len(available))
            samples = random.sample(available, sample_size)
            sampled_chunks.extend(samples)

            if len(sampled_chunks) >= balanced_count:
                break

        # Random part
        remaining_chunks = [c for c in valid_chunks if c not in sampled_chunks]
        sampled_chunks.extend(random.sample(remaining_chunks, min(random_count, len(remaining_chunks))))

    # Trim to exact count
    sampled_chunks = sampled_chunks[:n_samples]

    print(f"  [OK] Sampled {len(sampled_chunks)} chunks")

    # Print distribution
    source_counts = Counter(c['metadata']['source'] for c in sampled_chunks)
    print(f"\n  Distribution across documents:")
    for source, count in source_counts.most_common(10):
        print(f"    {source[:50]:50s} : {count:2d} chunks")
    if len(source_counts) > 10:
        print(f"    ... and {len(source_counts) - 10} more documents")

    return sampled_chunks


def categorize_chunk(chunk_text: str, metadata: Dict) -> str:
    """
    Categorize chunk based on content for better test organization.

    Args:
        chunk_text: Text content of chunk
        metadata: Chunk metadata

    Returns:
        Category label
    """
    text_lower = chunk_text.lower()

    # Simple keyword-based categorization
    if any(word in text_lower for word in ['albedo', 'reflectance', 'optical']):
        return 'albedo'
    elif any(word in text_lower for word in ['glacier', 'ice', 'snow']):
        return 'glaciology'
    elif any(word in text_lower for word in ['black carbon', 'impurities', 'aerosol']):
        return 'impurities'
    elif any(word in text_lower for word in ['remote sensing', 'satellite', 'modis']):
        return 'remote_sensing'
    elif any(word in text_lower for word in ['temperature', 'climate', 'warming']):
        return 'climate'
    elif any(word in text_lower for word in ['method', 'measurement', 'technique']):
        return 'methodology'
    else:
        return 'general'


def assess_difficulty(
    chunk_text: str,
    chunk_index: int,
    total_chunks: int
) -> str:
    """
    Assess query difficulty based on chunk characteristics.

    Args:
        chunk_text: Text content
        chunk_index: Position in document
        total_chunks: Total chunks in document

    Returns:
        Difficulty level: 'easy', 'medium', 'hard'
    """
    # Easy: Chunks with distinctive content (title, abstract, unique terms)
    if chunk_index < 5:  # Early chunks (title, abstract)
        return 'easy'

    # Medium: Mid-document chunks with specific content
    if chunk_index < total_chunks * 0.7:
        return 'medium'

    # Hard: Later chunks, references, generic text
    return 'hard'


def create_test_case(
    chunk: Dict,
    query_id: str
) -> Dict:
    """
    Create a test case from a sampled chunk.

    Args:
        chunk: Chunk dictionary with text and metadata
        query_id: Unique query identifier

    Returns:
        Test case dictionary
    """
    metadata = chunk['metadata']
    chunk_text = chunk['text']

    # Use chunk text as query (truncate if too long)
    query_text = chunk_text[:500] if len(chunk_text) > 500 else chunk_text

    # Ground truth: The source document should be retrieved
    source_doc = metadata['source']
    chunk_id = chunk['id']

    # For self-retrieval, the exact chunk should ideally be top-1
    # But we're generous: source doc in top-10 is success

    test_case = {
        'id': query_id,
        'query': query_text,
        'relevant_chunks': [chunk_id],  # Exact chunk
        'relevant_docs': [source_doc],  # Source document
        'relevance_scores': {
            chunk_id: 3  # Highest relevance (the chunk itself)
        },
        'category': categorize_chunk(chunk_text, metadata),
        'difficulty': assess_difficulty(
            chunk_text,
            metadata.get('chunk_index', 0),
            metadata.get('total_chunks', 100)
        ),
        'metadata': {
            'source': source_doc,
            'chunk_index': metadata.get('chunk_index', 0),
            'total_chunks': metadata.get('total_chunks', 0),
            'chunk_length': len(chunk_text)
        }
    }

    return test_case


def generate_test_dataset(
    collection: chromadb.Collection,
    n_queries: int = 30,
    diversity_mode: str = "balanced",
    output_path: str = None
) -> Dict:
    """
    Generate complete test dataset.

    Args:
        collection: ChromaDB collection
        n_queries: Number of test queries to generate
        diversity_mode: Sampling strategy
        output_path: Path to save JSON file (optional)

    Returns:
        Complete dataset dictionary
    """
    print("=" * 80)
    print("RAGDOC SYNTHETIC TEST DATASET GENERATOR")
    print("=" * 80)

    # Sample diverse chunks
    sampled_chunks = sample_diverse_chunks(
        collection,
        n_samples=n_queries,
        diversity_mode=diversity_mode
    )

    # Create test cases
    print(f"\n[*] Creating test cases...")
    test_cases = []

    for i, chunk in enumerate(sampled_chunks):
        query_id = f"q{i+1:03d}"
        test_case = create_test_case(chunk, query_id)
        test_cases.append(test_case)

    # Compile dataset
    dataset = {
        'name': 'RAGDOC Synthetic Evaluation Dataset',
        'version': '1.0',
        'description': 'Self-retrieval test set for RAGDOC evaluation',
        'created': datetime.now().isoformat(),
        'num_queries': len(test_cases),
        'strategy': 'self_retrieval',
        'diversity_mode': diversity_mode,
        'queries': test_cases
    }

    # Print statistics
    print(f"\n[*] Dataset Statistics:")
    print(f"  Total queries: {len(test_cases)}")

    category_counts = Counter(tc['category'] for tc in test_cases)
    print(f"\n  Categories:")
    for category, count in category_counts.most_common():
        print(f"    {category:20s} : {count:2d} queries")

    difficulty_counts = Counter(tc['difficulty'] for tc in test_cases)
    print(f"\n  Difficulty:")
    for difficulty, count in difficulty_counts.most_common():
        print(f"    {difficulty:20s} : {count:2d} queries")

    # Save to file if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

        print(f"\n[OK] Dataset saved to: {output_path}")

    print("=" * 80)
    print("[OK] TEST DATASET GENERATION COMPLETE")
    print("=" * 80 + "\n")

    return dataset


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic test dataset for RAGDOC evaluation"
    )

    parser.add_argument(
        '--n_queries',
        type=int,
        default=30,
        help='Number of test queries to generate (default: 30)'
    )

    parser.add_argument(
        '--db_path',
        type=str,
        default='D:/Claude Code/ragdoc-mcp/chroma_db_new',
        help='Path to ChromaDB directory'
    )

    parser.add_argument(
        '--collection',
        type=str,
        default='zotero_research_context_hybrid_v3',
        help='ChromaDB collection name'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='tests/test_datasets/synthetic_ragdoc_qa.json',
        help='Output JSON file path'
    )

    parser.add_argument(
        '--diversity_mode',
        choices=['balanced', 'random', 'mixed'],
        default='balanced',
        help='Sampling diversity strategy'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Initialize ChromaDB
    collection = initialize_chromadb(args.db_path, args.collection)

    # Generate dataset
    dataset = generate_test_dataset(
        collection,
        n_queries=args.n_queries,
        diversity_mode=args.diversity_mode,
        output_path=args.output
    )

    print(f"\n[OK] Successfully generated {len(dataset['queries'])} test queries")
    print(f"  Output file: {args.output}")
    print(f"\nNext steps:")
    print(f"  1. Review the generated dataset")
    print(f"  2. Run evaluation: python tests/evaluate_ragdoc.py")
    print()


if __name__ == "__main__":
    main()
