#!/usr/bin/env python3
"""Evaluate the actual MCP evidence tools only against reviewed corpus judgments.

Validation is offline. Running a benchmark calls configured embedding/rerank APIs.
No retrieval score is interpreted as a confidence in scientific truth.
"""
import argparse
import asyncio
import json
import sys
import re
import hashlib
from importlib.metadata import version
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def validate(dataset: dict) -> dict:
    questions = dataset.get('questions', [])
    if not questions:
        raise ValueError('No questions')
    ids = [q['id'] for q in questions]
    if len(set(ids)) != len(ids) or any(not q.get('query', '').strip() for q in questions):
        raise ValueError('Duplicate IDs or empty questions')
    judgments = dataset.get('judgments', {})
    if set(judgments) - set(ids):
        raise ValueError('Judgments refer to unknown questions')
    reviewed = 0
    for qid in ids:
        judgment = judgments.get(qid, {})
        if judgment.get('reviewed') is not True:
            continue
        if type(judgment.get('answerable')) is not bool:
            raise ValueError(f'{qid}: answerable must be a reviewed boolean')
        for field in ('sources', 'chunks'):
            values = judgment.get(field)
            if not isinstance(values, list) or any(not isinstance(v, str) or not v for v in values):
                raise ValueError(f'{qid}: {field} must be a list of identifiers')
            if judgment['answerable'] != bool(values):
                raise ValueError(f'{qid}: answerability conflicts with {field}')
        hashes = judgment.get('content_sha256_by_source', {})
        if (not isinstance(hashes, dict) or set(hashes) != set(judgment['sources']) or
                any(not isinstance(v, str) or not re.fullmatch(r'[0-9a-f]{64}', v) for v in hashes.values())):
            raise ValueError(f'{qid}: pin exactly the annotated sources with valid SHA-256 hashes')
        reviewed += 1
    return {'questions': len(ids), 'reviewed': reviewed, 'ready_to_score': reviewed == len(ids)}


def score_hits(hits: list[dict], judgment: dict) -> dict:
    sources = {h['provenance']['source'] for h in hits}
    chunks = {h['chunk_id'] for h in hits}
    if not judgment['answerable']:
        return {'answerable': False, 'returned_candidates': bool(hits),
                'note': 'Candidates for an unanswerable question are not evidence of an answer.'}
    relevant_sources, relevant_chunks = set(judgment['sources']), set(judgment['chunks'])
    return {'answerable': True, 'article_recall': len(sources & relevant_sources) / len(relevant_sources),
            'passage_recall': len(chunks & relevant_chunks) / len(relevant_chunks),
            'distinct_articles': len(sources),
            'reciprocal_rank': next((1 / (i + 1) for i, h in enumerate(hits)
                                     if h['chunk_id'] in relevant_chunks), 0.0)}


async def evaluate(dataset: dict, top_k: int) -> dict:
    if not validate(dataset)['ready_to_score']:
        raise ValueError('All questions require reviewed judgments before scoring; draft is not a benchmark result')
    from fastmcp import Client
    from src.server import mcp
    async with Client(mcp) as client:
        before = (await client.call_tool('audit_library', {})).structured_content
        if before['write_state'] != 'ready' or before['repairing']:
            raise ValueError('Benchmark requires a ready, versioned index')
        # Verify every annotation, even when the corresponding passage is not retrieved.
        for judgment in dataset['judgments'].values():
            annotated_sources = set()
            for chunk in judgment['chunks']:
                passage = (await client.call_tool('get_passage', {'chunk_id': chunk})).structured_content
                p = passage['provenance']
                if not passage['canonical_verified'] or p['source'] not in judgment['sources']:
                    raise ValueError(f'Unverified or out-of-scope annotated passage {chunk}')
                if p['content_sha256'] != judgment['content_sha256_by_source'].get(p['source']):
                    raise ValueError(f'Stale corpus judgment for {chunk}')
                annotated_sources.add(p['source'])
            if annotated_sources != set(judgment['sources']):
                raise ValueError('Each annotated source needs at least one verified relevant passage')
        rows = []
        for question in dataset['questions']:
            result = (await client.call_tool('search_evidence',
                      {'query': question['query'], 'top_k': top_k})).structured_content
            if result['index_revision'] != before['index_revision']:
                raise ValueError('Corpus changed during benchmark')
            checks = []
            for hit in result['hits']:
                passage = await client.call_tool('get_passage', {'chunk_id': hit['chunk_id'],
                    'expected_content_sha256': hit['provenance']['content_sha256']})
                checks.append(passage.structured_content['canonical_verified'])
            rows.append({'id': question['id'], **score_hits(result['hits'], dataset['judgments'][question['id']]),
                         'canonical_verified_passages': sum(checks), 'passages_checked': len(checks),
                         'retrieval': result['retrieval'], 'reranking': result['reranking'], 'warnings': result['warnings']})
        after = (await client.call_tool('audit_library', {})).structured_content
        if after['index_revision'] != before['index_revision'] or after['write_state'] != 'ready':
            raise ValueError('Corpus changed during benchmark')
    root = Path(__file__).resolve().parent.parent
    code_hash = hashlib.sha256()
    for path in sorted(list((root / 'src').rglob('*.py')) + list((root / 'scripts').glob('*.py'))):
        code_hash.update(str(path.relative_to(root)).encode())
        code_hash.update(path.read_bytes())
    return {'dataset': dataset['name'], 'at': datetime.now(timezone.utc).isoformat(),
            'dataset_sha256': hashlib.sha256(json.dumps(dataset, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            'code_sha256': code_hash.hexdigest(),
            'runtime_versions': {name: version(name) for name in ('fastmcp', 'chromadb', 'voyageai', 'cohere', 'rank-bm25')},
            'index_revision': before['index_revision'], 'top_k': top_k, 'rows': rows,
            'scope': 'MCP retrieval and canonical location checks, not generated-answer factuality'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    dataset = json.loads(args.dataset.read_text(encoding='utf-8'))
    if args.validate_only:
        print(json.dumps(validate(dataset), indent=2))
        return
    if not args.output:
        parser.error('--output is required for a benchmark run')
    result = asyncio.run(evaluate(dataset, args.top_k))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
