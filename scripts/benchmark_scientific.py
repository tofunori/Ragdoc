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
import statistics
from importlib.metadata import version
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def validate(dataset: dict, allow_assistant_draft: bool = False) -> dict:
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
    assistant_adjudicated = 0
    for qid in ids:
        judgment = judgments.get(qid, {})
        human_reviewed = judgment.get('reviewed') is True
        assistant_ready = (allow_assistant_draft and judgment.get('assistant_adjudicated') is True)
        if human_reviewed:
            if judgment.get('assistant_adjudicated') is True:
                raise ValueError(f'{qid}: assistant adjudication cannot be marked human-reviewed')
            if judgment.get('review_provenance') != 'human':
                raise ValueError(f'{qid}: reviewed judgments require review_provenance=human')
            if not isinstance(judgment.get('reviewer'), str) or not judgment['reviewer'].strip():
                raise ValueError(f'{qid}: reviewed judgments require a nonempty reviewer')
        if not human_reviewed and not assistant_ready:
            continue
        if type(judgment.get('answerable')) is not bool:
            raise ValueError(f'{qid}: answerable must be a reviewed boolean')
        canonical_available = judgment.get('canonical_evidence_available', judgment['answerable'])
        if type(canonical_available) is not bool or canonical_available != judgment['answerable']:
            raise ValueError(f'{qid}: answerable is the compatibility alias for canonical_evidence_available')
        if judgment.get('judgment_scope', 'canonical_evidence') != 'canonical_evidence':
            raise ValueError(f'{qid}: only canonical_evidence judgments are supported')
        corpus_status = judgment.get('corpus_status')
        if corpus_status is not None and corpus_status not in {
                'answerable', 'present_but_unverifiable', 'not_established_after_targeted_search'}:
            raise ValueError(f'{qid}: invalid corpus_status')
        if corpus_status is not None:
            if judgment['answerable'] != (corpus_status == 'answerable'):
                raise ValueError(f'{qid}: corpus_status conflicts with canonical evidence availability')
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
        if human_reviewed:
            reviewed += 1
        else:
            assistant_adjudicated += 1
    return {'questions': len(ids), 'reviewed': reviewed,
            'assistant_adjudicated': assistant_adjudicated,
            'ready_to_score': reviewed == len(ids),
            'ready_for_provisional_score': assistant_adjudicated == len(ids)}


def score_hits(hits: list[dict], judgment: dict) -> dict:
    sources = {h['provenance']['source'] for h in hits}
    chunks = {h['chunk_id'] for h in hits}
    if not judgment['answerable']:
        return {'answerable': False, 'canonical_evidence_available': False,
                'corpus_status': judgment.get('corpus_status', 'unspecified'),
                'returned_candidates': bool(hits),
                'note': 'Candidates do not establish canonical evidence or scientific answerability.'}
    relevant_sources, relevant_chunks = set(judgment['sources']), set(judgment['chunks'])
    return {'answerable': True, 'canonical_evidence_available': True,
            'corpus_status': judgment.get('corpus_status', 'answerable'),
            'article_recall': len(sources & relevant_sources) / len(relevant_sources),
            'passage_recall': len(chunks & relevant_chunks) / len(relevant_chunks),
            'distinct_articles': len(sources),
            'reciprocal_rank': next((1 / (i + 1) for i, h in enumerate(hits)
                                     if h['chunk_id'] in relevant_chunks), 0.0)}


def summarize_rows(rows: list[dict]) -> dict:
    """Summarize answerable and boundary questions without mixing their meanings."""
    positives = [row for row in rows if row['answerable']]
    negatives = [row for row in rows if not row['answerable']]
    summary = {
        'questions': len(rows),
        'canonical_evidence_questions': len(positives),
        'no_canonical_evidence_questions': len(negatives),
    }
    for metric in ('article_recall', 'passage_recall', 'reciprocal_rank'):
        summary[f'mean_{metric}'] = (statistics.fmean(row[metric] for row in positives)
                                     if positives else None)
    summary['candidate_rate_without_canonical_evidence'] = (
        statistics.fmean(bool(row['returned_candidates']) for row in negatives)
        if negatives else None)
    summary['canonical_verification_rate'] = (
        sum(row['canonical_verified_passages'] for row in rows) /
        sum(row['passages_checked'] for row in rows)
        if sum(row['passages_checked'] for row in rows) else None)
    return summary


async def evaluate(dataset: dict, top_k: int, mcp_url: str | None = None,
                   search_options: dict | None = None, split: str | None = None,
                   provisional_assistant: bool = False) -> dict:
    validation = validate(dataset, allow_assistant_draft=provisional_assistant)
    if not validation['ready_to_score']:
        if not provisional_assistant or not validation['ready_for_provisional_score']:
            raise ValueError('All questions require human-reviewed judgments before official scoring; '
                             'use --provisional-assistant only for an explicitly provisional diagnostic')
    from fastmcp import Client
    if mcp_url:
        target = mcp_url
    else:
        from src.server import mcp
        target = mcp
    async with Client(target) as client:
        before = (await client.call_tool('audit_library', {})).structured_content
        server_runtime_status = (await client.call_tool('get_runtime_status', {})).structured_content
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
        selected_questions = [question for question in dataset['questions']
                              if split is None or question.get('split') == split]
        if not selected_questions:
            raise ValueError(f'No questions found for split {split!r}')
        search_options = dict(search_options or {})
        use_lexical_query = search_options.pop('use_lexical_query', False)
        for question in selected_questions:
            search_request = {'query': question['query'], 'top_k': top_k, **search_options}
            if use_lexical_query and question.get('lexical_query'):
                search_request['subqueries'] = [question['lexical_query']]
            result = (await client.call_tool('search_evidence',
                      search_request)).structured_content
            if result['index_revision'] != before['index_revision']:
                raise ValueError('Corpus changed during benchmark')
            checks = []
            for hit in result['hits']:
                passage = await client.call_tool('get_passage', {'chunk_id': hit['chunk_id'],
                    'expected_content_sha256': hit['provenance']['content_sha256']})
                checks.append(passage.structured_content['canonical_verified'])
            rows.append({'id': question['id'], 'split': question.get('split', 'unspecified'),
                         **score_hits(result['hits'], dataset['judgments'][question['id']]),
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
    splits = {name: summarize_rows([row for row in rows if row['split'] == name])
              for name in sorted({row['split'] for row in rows})}
    return {'dataset': dataset['name'], 'at': datetime.now(timezone.utc).isoformat(),
            'dataset_sha256': hashlib.sha256(json.dumps(dataset, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            'runner_code_sha256': code_hash.hexdigest(),
            'runner_runtime_versions': {name: version(name) for name in ('fastmcp', 'chromadb', 'voyageai', 'cohere', 'rank-bm25')},
            'server_runtime_status': server_runtime_status,
            'server_code_sha256': None,
            'server_code_note': 'The MCP does not currently expose a server source hash.',
            'index_revision': before['index_revision'], 'top_k': top_k,
            'mcp_transport': 'remote' if mcp_url else 'in_process',
            'result_status': ('provisional_assistant_adjudicated'
                              if provisional_assistant else 'human_reviewed'),
            'evaluated_split': split or 'all',
            'search_config': {**(search_options or {}),
                              'use_lexical_query': use_lexical_query},
            'summary': summarize_rows(rows), 'splits': splits, 'rows': rows,
            'scope': 'MCP retrieval and canonical location checks, not generated-answer factuality'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--provisional-assistant', action='store_true',
                        help='Score assistant-adjudicated judgments as a clearly labelled diagnostic.')
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--url', help='Remote MCP URL. Omit to use the in-process server (tests/local data).')
    parser.add_argument('--split', choices=('development', 'heldout'))
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--multi-query', action='store_true')
    parser.add_argument('--use-lexical-query', action='store_true',
                        help='Add each annotated English lexical query as a recorded subquery.')
    parser.add_argument('--retrieval-strategy', choices=('passages', 'articles_then_passages'),
                        default='passages')
    parser.add_argument('--article-limit', type=int, default=8)
    parser.add_argument('--max-per-document', type=int, default=2)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    dataset = json.loads(args.dataset.read_text(encoding='utf-8'))
    if args.validate_only:
        print(json.dumps(validate(dataset, allow_assistant_draft=args.provisional_assistant), indent=2))
        return
    if not args.output:
        parser.error('--output is required for a benchmark run')
    if not 0 <= args.alpha <= 1:
        parser.error('--alpha must be in [0, 1]')
    if args.multi_query and args.use_lexical_query:
        parser.error('--multi-query and --use-lexical-query are alternative expansion modes')
    search_options = {
        'alpha': args.alpha,
        'multi_query': args.multi_query,
        'retrieval_strategy': args.retrieval_strategy,
        'article_limit': args.article_limit,
        'max_per_document': args.max_per_document,
        'use_lexical_query': args.use_lexical_query,
    }
    result = asyncio.run(evaluate(dataset, args.top_k, args.url, search_options, args.split,
                                  args.provisional_assistant))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
