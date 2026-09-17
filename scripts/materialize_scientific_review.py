#!/usr/bin/env python3
"""Verify proposed benchmark evidence and create a human review document.

This tool never marks a judgment as human-reviewed. It turns an assistant's
selection of chunk IDs into a pinned, canonical draft plus a readable checklist.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from fastmcp import Client


def _excerpt(text: str, limit: int = 900) -> str:
    compact = ' '.join(text.split())
    return compact if len(compact) <= limit else compact[:limit].rstrip() + '…'


async def materialize(dataset: dict, selections: dict, url: str) -> tuple[dict, str]:
    question_ids = [question['id'] for question in dataset['questions']]
    if set(selections) != set(question_ids):
        missing = sorted(set(question_ids) - set(selections))
        extra = sorted(set(selections) - set(question_ids))
        raise ValueError(f'Selections must cover every question; missing={missing}, extra={extra}')

    judgments = {}
    evidence_by_question = {}
    async with Client(url) as client:
        audit = (await client.call_tool('audit_library', {})).structured_content
        if audit['write_state'] != 'ready' or audit['repairing']:
            raise RuntimeError('Review materialization requires a ready index')
        revision = audit['index_revision']
        for question in dataset['questions']:
            qid = question['id']
            selection = selections[qid]
            answerable = selection.get('answerable')
            chunks = selection.get('chunks', [])
            corpus_status = selection.get(
                'corpus_status', 'answerable' if answerable else 'not_established_after_targeted_search')
            if corpus_status not in {
                'answerable', 'present_but_unverifiable', 'not_established_after_targeted_search'
            }:
                raise ValueError(f'{qid}: unknown corpus_status {corpus_status!r}')
            if type(answerable) is not bool:
                raise ValueError(f'{qid}: answerable must be boolean')
            if answerable != bool(chunks):
                raise ValueError(f'{qid}: answerable must agree with selected chunks')
            sources = []
            hashes = {}
            evidence = []
            for chunk_id in chunks:
                passage = (await client.call_tool('get_passage', {
                    'chunk_id': chunk_id, 'context': 'paragraphs',
                    'paragraphs_before': 1, 'paragraphs_after': 1,
                    'max_context_chars': 6000,
                })).structured_content
                if not passage['canonical_verified']:
                    raise ValueError(f'{qid}: non-canonical passage {chunk_id}')
                provenance = passage['provenance']
                source = provenance['source']
                digest = provenance['content_sha256']
                if source not in sources:
                    sources.append(source)
                hashes[source] = digest
                evidence.append({
                    'chunk_id': chunk_id,
                    'source': source,
                    'content_sha256': digest,
                    'bibliography': provenance['bibliography'],
                    'location': provenance['location'],
                    'text': passage['text'],
                    'context': passage['context'],
                })
            judgments[qid] = {
                'reviewed': False,
                'answerable': answerable,
                'judgment_scope': 'canonical_evidence',
                'canonical_evidence_available': answerable,
                'corpus_status': corpus_status,
                'sources': sources,
                'chunks': chunks,
                'content_sha256_by_source': hashes,
                'assistant_rationale': selection.get('rationale', '').strip(),
                'assistant_adjudicated': True,
            }
            evidence_by_question[qid] = evidence
        final_audit = (await client.call_tool('audit_library', {})).structured_content
        if final_audit['index_revision'] != revision or final_audit['write_state'] != 'ready':
            raise RuntimeError('Corpus changed while materializing the review')

    draft = dict(dataset)
    draft['name'] = dataset['name'].replace('annotation in progress', 'assistant draft')
    draft['status'] = 'assistant_adjudicated_pending_human_review'
    draft['index_revision'] = revision
    draft['assistant_selection_sha256'] = hashlib.sha256(
        json.dumps(selections, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    draft['judgments'] = judgments

    lines = [
        '# Révision humaine — benchmark scientifique Ragdoc v1', '',
        f'Index épinglé : `{revision}`', '',
        ('Chaque décision ci-dessous est une proposition de Codex. Cocher une question signifie que '
         'le statut de la preuve canonique et les passages sélectionnés ont été vérifiés humainement.'), '',
        'Politique : `answerable` est conservé comme alias technique de « preuve canonique disponible '
        'dans cette bibliothèque ». Une absence signifie « preuve canonique insuffisante dans cet '
        'index », pas « affirmation fausse dans la littérature ».', '',
    ]
    for question in dataset['questions']:
        qid = question['id']
        judgment = judgments[qid]
        label = ('preuve canonique disponible' if judgment['canonical_evidence_available']
                 else 'preuve canonique insuffisante')
        lines.extend([
            f"## [ ] {qid} · {question.get('split', 'unspecified')} · {label}", '',
            question['query'], '',
            f"**Raison proposée.** {judgment['assistant_rationale']}", '',
            f"**Statut du corpus proposé.** `{judgment['corpus_status']}`", '',
            f"[Ouvrir le dossier complet des candidats]({qid}.json)", '',
        ])
        for number, evidence in enumerate(evidence_by_question[qid], 1):
            bibliography = evidence['bibliography']
            title = bibliography.get('title') or evidence['source']
            year = bibliography.get('year') or 'année inconnue'
            location = evidence['location']
            locator = location.get('section_path') or location.get('section') or 'section inconnue'
            lines.extend([
                f"**Preuve {number}.** {title} ({year})", '',
                f"- Source : `{evidence['source']}`",
                f"- Passage : `{evidence['chunk_id']}`",
                f"- Section : {locator}",
                f"- Extrait : {_excerpt(evidence['text'])}", '',
                '<details>',
                '<summary>Afficher le passage canonique complet</summary>', '',
                evidence['text'], '',
                '</details>', '',
            ])
        if not evidence_by_question[qid]:
            lines.extend([
                'Aucun passage canonique retenu. Une recherche indépendante ciblée doit confirmer '
                'ce statut avant validation humaine.', ''
            ])
    return draft, '\n'.join(lines).rstrip() + '\n'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('selections', type=Path)
    parser.add_argument('--url', required=True)
    parser.add_argument('--dataset-output', type=Path, required=True)
    parser.add_argument('--review-output', type=Path, required=True)
    args = parser.parse_args()
    dataset = json.loads(args.dataset.read_text(encoding='utf-8'))
    selections = json.loads(args.selections.read_text(encoding='utf-8'))
    draft, review = asyncio.run(materialize(dataset, selections, args.url))
    args.dataset_output.parent.mkdir(parents=True, exist_ok=True)
    args.review_output.parent.mkdir(parents=True, exist_ok=True)
    args.dataset_output.write_text(json.dumps(draft, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    args.review_output.write_text(review, encoding='utf-8')
    print(json.dumps({
        'questions': len(draft['questions']),
        'canonical_evidence_available': sum(j['answerable'] for j in draft['judgments'].values()),
        'canonical_evidence_unavailable': sum(not j['answerable'] for j in draft['judgments'].values()),
        'human_reviewed': sum(j['reviewed'] for j in draft['judgments'].values()),
        'dataset_output': str(args.dataset_output),
        'review_output': str(args.review_output),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
