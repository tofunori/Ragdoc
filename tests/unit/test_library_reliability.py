"""Hermetic safety, provenance and MCP tests. No production collection or paid API."""
import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastmcp import Client

from src import server
from src.hybrid_retriever import HybridRetriever
from src.index_safety import replace_document, IndexRepairRequired
from src.library import Library, document_metadata, locate_chunks, sha256, read_sidecar


class Collection:
    def __init__(self):
        self.rows = {}
        self.metadata = {"ragdoc_revision": "initial"}
        self.upsert_calls = 0
        self.fail_on = set()
        self.fail_delete = False

    def get(self, ids=None, where=None, include=None):
        rows = [(i, r) for i, r in self.rows.items() if (ids is None or i in ids)
                and (where is None or r['metadatas'].get('source') == where['source'])]
        return {"ids": [i for i, _ in rows],
                **{key: [copy.deepcopy(r[key]) for _, r in rows]
                   for key in ('documents', 'metadatas', 'embeddings')}}

    def upsert(self, ids, **data):
        self.upsert_calls += 1
        for j, i in enumerate(ids):
            self.rows[i] = {key: copy.deepcopy(values[j]) for key, values in data.items()}
            if self.upsert_calls in self.fail_on:
                raise RuntimeError("simulated partial storage write")

    def delete(self, ids):
        for i in ids:
            self.rows.pop(i, None)
        if self.fail_delete:
            self.fail_delete = False
            raise RuntimeError('simulated delete failure')

    def modify(self, metadata):
        self.metadata = metadata

    def count(self):
        return len(self.rows)


def payload(texts, ids=None, source='paper.md', **extra):
    return {"ids": ids or [f'c{i}' for i in range(len(texts))], "documents": texts,
            "embeddings": [[1.0, 0.0] for _ in texts],
            "metadatas": [{"source": source, "chunk_index": i, "total_chunks": len(texts), **extra}
                          for i, _ in enumerate(texts)]}


@pytest.fixture
def collection():
    c = Collection()
    c.upsert(**payload(['old text', 'old second']))
    return c


def test_replacement_preserves_old_on_partial_write(collection):
    original = copy.deepcopy(collection.rows)
    collection.fail_on = {2}
    with pytest.raises(RuntimeError):
        replace_document(collection, 'paper.md', payload(['new', 'added'], ids=['c0', 'new']))
    assert collection.rows == original
    assert collection.metadata['ragdoc_write_state'] == 'ready'


def test_replacement_rolls_back_failed_cleanup(collection):
    original = copy.deepcopy(collection.rows)
    collection.fail_delete = True
    with pytest.raises(RuntimeError):
        replace_document(collection, 'paper.md', payload(['new'], ids=['c0']))
    assert collection.rows == original


def test_failed_rollback_requires_repair(collection):
    collection.fail_on = {2, 3}
    with pytest.raises(IndexRepairRequired):
        replace_document(collection, 'paper.md', payload(['new']))
    assert collection.metadata['ragdoc_write_state'] == 'writing'
    with pytest.raises(IndexRepairRequired):
        replace_document(collection, 'other.md', payload(['other'], ['other'], source='other.md'))
    assert collection.metadata['ragdoc_write_state'] == 'writing'


def test_successful_replacement_removes_only_obsolete(collection):
    collection.upsert(**payload(['other'], ids=['other'], source='other.md'))
    replace_document(collection, 'paper.md', payload(['new'], ids=['new']))
    assert set(collection.rows) == {'new', 'other'}


def test_invalid_replacement_makes_no_write(collection):
    before = copy.deepcopy(collection.rows)
    with pytest.raises(ValueError):
        replace_document(collection, 'paper.md', payload([]))
    assert collection.rows == before
    assert collection.metadata['ragdoc_revision'] == 'initial'


def test_immutable_snapshot_survives_source_change(tmp_path):
    library = Library(tmp_path)
    digest = library.snapshot('A\r\nB\n')
    library.snapshot('changed')
    assert library.read(digest) == 'A\r\nB\n'
    (tmp_path / 'snapshots' / f'{digest}.md').write_text('corruption')
    with pytest.raises(ValueError, match='checksum'):
        library.read(digest)
    with pytest.raises(ValueError):
        library.read('../../outside')


def test_metadata_does_not_infer_bibliography(tmp_path):
    metadata = document_metadata(Path('Painter_2009.md'), 'text', {})
    assert metadata['title_verified'] is False
    assert json.loads(metadata['bibliography_json']) == {}
    source = tmp_path / 'paper.md'
    source.with_suffix('.metadata.json').write_text('{"year":"2009"}')
    with pytest.raises(ValueError):
        read_sidecar(source)


def test_locators_use_exact_offsets_and_bound_page_mapping():
    text = '# Methods\nA result.\nAnother result.'
    sidecar = {'content_sha256': sha256(text), 'page_spans': [{'start': 0, 'end': len(text), 'page': 4}]}
    locations = locate_chunks(text, ['A result.', 'changed text'], sidecar)
    assert locations[0]['section'] == 'Methods'
    assert text[locations[0]['char_start']:locations[0]['char_end']] == 'A result.'
    assert locations[0]['page_start'] == 4
    assert locations[1] == {'locator_status': 'unresolved'}
    assert locate_chunks('Repeated.\nRepeated.', ['Repeated.'], {}) == [{'locator_status': 'unresolved'}]
    sidecar['content_sha256'] = 'outdated'
    assert 'page_start' not in locate_chunks(text, ['A result.'], sidecar)[0]
    assert 'page_start' not in locate_chunks(text, [text], {'content_sha256': sha256(text),
        'page_spans': [{'start': 0, 'end': 3, 'page': 1}]})[0]


def test_ingestion_events_keep_failure_visible(tmp_path):
    library = Library(tmp_path)
    library.record('a.md', 'ready', 'old')
    library.record('a.md', 'failed', error='embedding failed')
    assert library.latest_events()[0]['status'] == 'failed'


def test_library_relative_path_is_readable(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    library = Library(Path('relative_library'))
    library.record('a.md', 'ready')
    assert library.latest_events()[0]['source'] == 'a.md'


def test_pure_lexical_is_local_and_drops_unmatched(collection):
    embed = Mock(side_effect=AssertionError('must not embed'))
    retriever = HybridRetriever(collection, embed, use_advanced_tokenizer=False)
    assert retriever.search('absent_token', alpha=0) == []
    assert retriever.search('old', alpha=0)  # zero/negative BM25 scores still match
    embed.assert_not_called()


def test_embedding_failure_falls_back_to_lexical(collection):
    retriever = HybridRetriever(collection, Mock(side_effect=RuntimeError('offline')), use_advanced_tokenizer=False)
    assert retriever.search('old', alpha=1)
    assert retriever.last_status['mode'] == 'lexical'
    assert 'semantic_unavailable' in retriever.last_status['warnings']


def test_revision_refreshes_same_size_corpus(collection):
    retriever = HybridRetriever(collection, use_advanced_tokenizer=False)
    assert retriever.search('old', alpha=0)
    replace_document(collection, 'paper.md', payload(['fresh', 'fresh second']))
    assert retriever.search('old', alpha=0) == []
    assert retriever.search('fresh', alpha=0)
    collection.metadata['ragdoc_write_state'] = 'writing'
    with pytest.raises(RuntimeError, match='incomplete'):
        retriever.search('fresh', alpha=0)


def test_filters_agree_for_year_bounds(collection):
    collection.upsert(**payload(['snow'], ['year'], year=2020))
    r = HybridRetriever(collection, use_advanced_tokenizer=False)
    assert r.search('snow', alpha=0, where={'year': {'$gte': 2020}})
    assert not r.search('snow', alpha=0, where={'year': {'$lt': 2020}})


@pytest.fixture
def mcp_library(tmp_path, monkeypatch):
    c = Collection()
    text = '# Methods\nSnow albedo decreases.\n\n# Results\nSnow albedo remains variable.'
    chunks = ['Snow albedo decreases.', 'Snow albedo remains variable.']
    digest = Library(tmp_path).snapshot(text)
    metadata = document_metadata(Path('paper.md'), text, {'title': 'Snow study', 'year': 2020})
    data = payload(chunks, canonical_sha256=digest, **{k: v for k, v in metadata.items() if k != 'canonical_sha256'})
    for meta, locator in zip(data['metadatas'], locate_chunks(text, chunks, {})):
        meta.update(locator)
    c.upsert(**data)
    # Return independent metadata snapshots like Chroma, so revision checks are meaningful.
    def fresh_collection(**kwargs):
        current = copy.copy(c)
        current.metadata = copy.deepcopy(c.metadata)
        return current
    monkeypatch.setattr(server, 'chroma_client', SimpleNamespace(get_collection=fresh_collection))
    monkeypatch.setattr(server, 'LIBRARY_PATH', tmp_path)
    monkeypatch.setattr(server, 'hybrid_retriever', HybridRetriever(c, use_advanced_tokenizer=False))
    monkeypatch.setattr(server, 'init_cohere_client', lambda: None)
    return c, text, digest


def test_mcp_structured_evidence_canonical_read_and_version_pin(mcp_library):
    _, text, digest = mcp_library
    async def check():
        async with Client(server.mcp) as client:
            tools = await client.list_tools()
            schema = next(t.outputSchema for t in tools if t.name == 'search_evidence')
            assert 'hits' in schema['properties']
            result = await client.call_tool('search_evidence', {'query': 'Snow', 'alpha': 0})
            data = result.structured_content
            assert len(data['hits']) == 2
            assert data['reranking'] == 'unavailable'
            hit = data['hits'][0]
            assert hit['scores']['rerank'] is None
            assert hit['provenance']['bibliography']['doi'] is None
            passage = await client.call_tool('get_passage', {'chunk_id': hit['chunk_id'], 'expected_content_sha256': digest})
            assert passage.structured_content['canonical_verified']
            read = await client.call_tool('read_document', {'source': 'paper.md', 'limit': 10})
            data = read.structured_content
            assert data['text'] == text[:10]
            rest = await client.call_tool('read_document', {'source': 'paper.md', 'offset': data['next_offset'],
                                                          'expected_content_sha256': digest})
            assert data['text'] + rest.structured_content['text'] == text
            wrong = await client.call_tool('get_passage', {'chunk_id': hit['chunk_id'],
                                                         'expected_content_sha256': 'wrong'}, raise_on_error=False)
            assert wrong.is_error
            empty = await client.call_tool('search_evidence', {'query': 'absent', 'alpha': 0})
            assert empty.structured_content['hits'] == []
            assert empty.structured_content['index_revision'] == 'initial'
            invalid = await client.call_tool('search_evidence', {'query': ''}, raise_on_error=False)
            assert invalid.is_error
    asyncio.run(check())


def test_legacy_read_warns_instead_of_claiming_canonical(mcp_library):
    c, _, _ = mcp_library
    for row in c.rows.values():
        row['metadatas'].pop('canonical_sha256')
    result = server.get_document_content.fn('paper.md', format='text')
    assert 'Legacy index' in result


def test_document_content_uses_snapshot_not_overlapping_chunks(mcp_library):
    _, text, _ = mcp_library
    assert server.get_document_content.fn('paper.md', format='text') == text


def test_audit_reports_missing_snapshot_and_catalogue_unknowns(mcp_library):
    c, _, _ = mcp_library
    c.upsert(**payload(['legacy'], ['legacy'], source='legacy.md'))
    audit = server.audit_library.fn()
    assert any(f['source'] == 'legacy.md' and f['issue'] == 'legacy_no_canonical_snapshot' for f in audit['findings'])
    catalogue = server.search_documents.fn(query='Snow', year_from=2020)
    assert catalogue['total'] == 1


def test_corrupt_snapshot_cannot_be_presented_as_verified(mcp_library, tmp_path):
    _, _, digest = mcp_library
    (tmp_path / 'snapshots' / f'{digest}.md').write_text('broken')
    passage = server.get_passage.fn('c0')
    assert not passage.canonical_verified
    assert 'canonical_snapshot_unavailable_or_corrupt' in passage.warnings
    with pytest.raises(Exception, match='snapshot unavailable or corrupt'):
        server.read_document.fn('paper.md')


def test_source_diversity_and_rerank_service_failure(mcp_library, monkeypatch):
    c, _, _ = mcp_library
    c.upsert(**payload(['Snow impurity observation.'], ['other'], source='other.md'))
    monkeypatch.setattr(server, 'init_cohere_client', lambda: SimpleNamespace(
        rerank=Mock(side_effect=RuntimeError('service outage'))))
    result = server.search_evidence.fn('Snow', alpha=0, max_per_document=1)
    assert len(result.hits) == 2
    assert len({h.provenance.source for h in result.hits}) == 2
    assert result.reranking == 'unavailable'


def test_benchmark_refuses_unreviewed_or_unpinned_judgments():
    from scripts.benchmark_scientific import validate, evaluate, score_hits
    draft = {'questions': [{'id': 'q1', 'query': 'snow?'}], 'judgments': {}}
    assert validate(draft)['ready_to_score'] is False
    with pytest.raises(ValueError, match='reviewed'):
        asyncio.run(evaluate(draft, 10))
    draft['judgments']['q1'] = {'reviewed': True, 'answerable': True,
        'sources': ['paper.md'], 'chunks': ['c0'], 'content_sha256_by_source': {'other.md': 'a' * 64}}
    with pytest.raises(ValueError, match='pin exactly'):
        validate(draft)
    draft['judgments']['q1']['content_sha256_by_source'] = {'paper.md': 'a' * 64}
    assert validate(draft)['ready_to_score']
    score = score_hits([{'chunk_id': 'c0', 'provenance': {'source': 'paper.md'}}], draft['judgments']['q1'])
    assert score['article_recall'] == score['passage_recall'] == 1
    negative = score_hits([], {'answerable': False, 'sources': [], 'chunks': []})
    assert negative['returned_candidates'] is False
    assert 'article_recall' not in negative


def test_ndcg_cannot_gain_from_repeated_passage():
    from src.rag_evaluator import RAGEvaluator
    assert RAGEvaluator().ndcg_at_k(['A', 'A'], {'A': 1}, 2) == 1.0


def test_benchmark_runs_actual_mcp_with_positive_and_no_answer_cases(mcp_library):
    from scripts.benchmark_scientific import evaluate
    c, _, digest = mcp_library
    c.metadata['ragdoc_write_state'] = 'ready'
    dataset = {'name': 'synthetic integration check', 'questions': [
        {'id': 'positive', 'query': 'Snow'}, {'id': 'negative', 'query': 'nonexistent_token'}],
        'judgments': {
            'positive': {'reviewed': True, 'answerable': True, 'sources': ['paper.md'], 'chunks': ['c0'],
                         'content_sha256_by_source': {'paper.md': digest}},
            'negative': {'reviewed': True, 'answerable': False, 'sources': [], 'chunks': [],
                         'content_sha256_by_source': {}}}}
    result = asyncio.run(evaluate(dataset, 10))
    assert len(result['dataset_sha256']) == len(result['code_sha256']) == 64
    assert result['rows'][0]['passage_recall'] == 1.0
    assert result['rows'][0]['canonical_verified_passages'] == 2
    assert result['rows'][1]['returned_candidates'] is False


@pytest.mark.parametrize('failure', ['embeddings', 'empty_chunks', 'rollback'])
def test_indexer_counts_errors_preserves_previous_and_stops_on_failed_rollback(tmp_path, monkeypatch, collection, failure):
    from scripts import index_incremental as indexer
    path = tmp_path / 'paper.md'
    path.write_text('new contents')
    library = tmp_path / 'library'
    monkeypatch.setattr(indexer, 'MARKDOWN_DIR', tmp_path)
    monkeypatch.setattr(indexer, 'LIBRARY_PATH', library)
    monkeypatch.setattr(indexer, 'CHROMA_DB_PATH', tmp_path / 'db')
    monkeypatch.setattr(indexer, 'VOYAGE_API_KEY', 'test-placeholder')
    monkeypatch.setattr(indexer.voyageai, 'Client', lambda **k: Mock())
    monkeypatch.setattr(indexer.chromadb, 'HttpClient', lambda **k: SimpleNamespace(
        heartbeat=lambda: None, get_or_create_collection=lambda **kw: collection))
    chunk = SimpleNamespace(text='new contents', token_count=2)
    chunker = SimpleNamespace(chunk=lambda text: [] if failure == 'empty_chunks' else [chunk], refine=lambda c: c)
    for name in ('TokenChunker', 'SemanticChunker', 'OverlapRefinery'):
        monkeypatch.setattr(indexer, name, lambda **kw: chunker)
    monkeypatch.setattr(indexer, 'process_embeddings_with_limit_check',
                        lambda *a: [] if failure == 'embeddings' else [[1.0, 0.0]])
    previous = copy.deepcopy(collection.rows)
    if failure == 'rollback':
        collection.fail_on = {2, 3}
        with pytest.raises(IndexRepairRequired):
            indexer.index_incremental()
        assert collection.metadata['ragdoc_write_state'] == 'writing'
    else:
        stats = indexer.index_incremental()
        assert stats['errors'] == 1
        assert collection.rows == previous
        assert collection.upsert_calls == 1
    assert Library(library).latest_events()[0]['status'] == 'failed'


def test_real_chroma_replacement_and_query(tmp_path):
    import chromadb
    from chromadb.config import Settings
    client = chromadb.PersistentClient(path=str(tmp_path / 'chroma'), settings=Settings(anonymized_telemetry=False))
    c = client.create_collection('reliability_test', embedding_function=None)
    replace_document(c, 'paper.md', payload(['snow albedo'], ['v1']))
    r = HybridRetriever(c, lambda texts: [[1.0, 0.0] for _ in texts], use_advanced_tokenizer=False,
                        revision_provider=lambda: client.get_collection('reliability_test').metadata)
    assert r.search('snow')[0]['id'] == 'v1'
    replace_document(c, 'paper.md', payload(['fire aerosol'], ['v2']))
    assert r.search('snow', alpha=0) == []
    assert r.search('fire', alpha=0)[0]['id'] == 'v2'
