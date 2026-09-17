#!/usr/bin/env python3
"""Build auditable evidence packets for manual scientific benchmark annotation.

The script deliberately does not decide answerability or write gold judgments.
It searches through several retrieval channels, resolves candidate chunks against
their canonical sources, and saves enough context for a reviewer to adjudicate.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from fastmcp import Client


def _dataset_hash(dataset: dict) -> str:
    payload = json.dumps(dataset, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _packet_signature(question: dict, index_revision: str, top_k: int,
                      max_passages: int) -> str:
    payload = {
        'question': question,
        'index_revision': index_revision,
        'top_k': top_k,
        'max_passages': max_passages,
        'requests': _search_requests(question, top_k),
        'format_version': 2,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _search_requests(question: dict, top_k: int) -> list[tuple[str, dict]]:
    query = question['query']
    lexical_query = question.get('lexical_query', query)
    return [
        ('hybrid', {
            'query': query, 'top_k': top_k, 'alpha': 0.5, 'multi_query': True,
            'max_per_document': 2, 'preview_chars': 1600,
        }),
        ('semantic_articles', {
            'query': query, 'top_k': top_k, 'alpha': 1.0, 'multi_query': True,
            'max_per_document': 2, 'preview_chars': 1600,
            'retrieval_strategy': 'articles_then_passages', 'article_limit': 10,
        }),
        ('lexical', {
            'query': lexical_query, 'top_k': top_k, 'alpha': 0.0,
            'multi_query': False, 'max_per_document': 2, 'preview_chars': 1600,
        }),
    ]


async def _prepare_question(client: Client, question: dict, index_revision: str,
                            top_k: int, max_passages: int) -> dict:
    channels = {}
    candidates: dict[str, dict] = {}
    for name, request in _search_requests(question, top_k):
        response = (await client.call_tool('search_evidence', request)).structured_content
        if response['index_revision'] != index_revision:
            raise RuntimeError('Corpus changed while preparing annotation packets')
        channels[name] = {
            'request': request,
            'queries': response['queries'],
            'retrieval': response['retrieval'],
            'reranking': response['reranking'],
            'warnings': response['warnings'],
            'selected_articles': response['selected_articles'],
            'hit_ids': [hit['chunk_id'] for hit in response['hits']],
        }
        for rank, hit in enumerate(response['hits'], 1):
            candidate = candidates.setdefault(hit['chunk_id'], {
                'chunk_id': hit['chunk_id'],
                'provenance': hit['provenance'],
                'excerpt': hit['excerpt'],
                'channels': [],
            })
            candidate['channels'].append({
                'name': name, 'rank': rank, 'scores': hit['scores'],
                'matched_queries': hit['matched_queries'],
            })

    # Prefer passages supported by several independent retrieval configurations,
    # then by their best rank. This ordering is for review efficiency only.
    ordered = sorted(candidates.values(), key=lambda item: (
        -len(item['channels']), min(channel['rank'] for channel in item['channels'])))
    verified = []
    for candidate in ordered[:max_passages]:
        passage = (await client.call_tool('get_passage', {
            'chunk_id': candidate['chunk_id'],
            'expected_content_sha256': candidate['provenance']['content_sha256'],
            'context': 'section',
            'max_context_chars': 12000,
        })).structured_content
        candidate['canonical_verified'] = passage['canonical_verified']
        candidate['passage_text'] = passage['text']
        candidate['context'] = passage['context']
        candidate['passage_warnings'] = passage['warnings']
        verified.append(candidate)

    return {
        'id': question['id'],
        'split': question.get('split', 'unspecified'),
        'query': question['query'],
        'lexical_query': question.get('lexical_query'),
        'index_revision': index_revision,
        'packet_signature': _packet_signature(
            question, index_revision, top_k, max_passages),
        'channels': channels,
        'candidates': verified,
        'review': {
            'answerable': None,
            'relevant_sources': [],
            'relevant_chunks': [],
            'rationale': '',
            'human_reviewed': False,
        },
    }


async def prepare(dataset: dict, url: str, output_dir: Path, top_k: int,
                  max_passages: int, resume: bool) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    async with Client(url) as client:
        audit = (await client.call_tool('audit_library', {})).structured_content
        if audit['write_state'] != 'ready' or audit['repairing']:
            raise RuntimeError('Annotation requires a ready, stable index')
        revision = audit['index_revision']
        completed = []
        for position, question in enumerate(dataset['questions'], 1):
            destination = output_dir / f"{question['id']}.json"
            expected_signature = _packet_signature(
                question, revision, top_k, max_passages)
            if resume and destination.exists():
                previous = json.loads(destination.read_text(encoding='utf-8'))
                if previous.get('packet_signature') == expected_signature:
                    completed.append(question['id'])
                    print(f"[{position:02}/{len(dataset['questions'])}] {question['id']} reused")
                    continue
            print(f"[{position:02}/{len(dataset['questions'])}] {question['id']} searching")
            packet = await _prepare_question(client, question, revision, top_k, max_passages)
            destination.write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding='utf-8')
            completed.append(question['id'])
        final_audit = (await client.call_tool('audit_library', {})).structured_content
        if final_audit['index_revision'] != revision or final_audit['write_state'] != 'ready':
            raise RuntimeError('Corpus changed while preparing annotation packets')

    manifest = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'dataset': dataset['name'],
        'dataset_sha256': _dataset_hash(dataset),
        'index_revision': revision,
        'questions': len(dataset['questions']),
        'packets': completed,
        'retrieval_channels': ['hybrid', 'semantic_articles', 'lexical'],
        'judgment_status': 'candidate evidence only; no answerability decision',
    }
    (output_dir / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--url', required=True, help='Ragdoc MCP URL')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--top-k', type=int, default=12)
    parser.add_argument('--max-passages', type=int, default=12)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.top_k <= 100 or not 1 <= args.max_passages <= 100:
        parser.error('--top-k and --max-passages must be in [1, 100]')
    dataset = json.loads(args.dataset.read_text(encoding='utf-8'))
    manifest = asyncio.run(prepare(
        dataset, args.url, args.output_dir, args.top_k, args.max_passages, args.resume))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
