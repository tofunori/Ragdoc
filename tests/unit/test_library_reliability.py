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
from src.index_safety import replace_document, bump_revision, IndexRepairRequired
from src.library import Library, document_metadata, locate_chunks, sha256, read_sidecar


class Collection:
    def __init__(self):
        self.rows = {}
        self.metadata = {
            "ragdoc_revision": "initial",
            "embedding_model": "voyage-context-4",
        }
        self.upsert_calls = 0
        self.fail_on = set()
        self.fail_delete = False

    def get(self, ids=None, where=None, include=None, limit=None, offset=0):
        rows = [(i, r) for i, r in self.rows.items() if (ids is None or i in ids)
                and (where is None or r['metadatas'].get('source') == where['source'])]
        rows = rows[offset:offset + limit] if limit is not None else rows[offset:]
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


def test_revision_update_omits_immutable_chroma_settings():
    c = Collection()
    c.metadata.update({"hnsw:space": "cosine", "hnsw:M": 64, "pipeline": "legacy"})
    original_modify = c.modify

    def reject_hnsw(metadata):
        if any(key.startswith("hnsw:") for key in metadata):
            raise ValueError("immutable HNSW setting was resubmitted")
        original_modify(metadata)

    c.modify = reject_hnsw
    bump_revision(c, "writing")
    assert c.metadata["pipeline"] == "legacy"
    assert c.metadata["ragdoc_write_state"] == "writing"
    assert "hnsw:space" not in c.metadata
    assert json.loads(c.metadata["ragdoc_preserved_hnsw_json"]) == {
        "hnsw:M": 64,
        "hnsw:space": "cosine",
    }

    bump_revision(c)
    assert json.loads(c.metadata["ragdoc_preserved_hnsw_json"])["hnsw:space"] == "cosine"


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


def test_paginated_reads_cover_large_filtered_corpus_without_unbounded_calls():
    from src.chroma_reads import read_collection
    c = Collection()
    c.upsert(**payload([f'text {i}' for i in range(1501)]))
    c.upsert(**payload(['excluded'], ['other'], source='other.md'))
    original = c.get
    calls = []
    def bounded(**kwargs):
        assert 1 <= kwargs['limit'] <= 500
        calls.append(kwargs['offset'])
        return original(**kwargs)
    c.get = bounded
    rows = read_collection(c, include=['documents', 'metadatas', 'embeddings'], where={'source':'paper.md'})
    assert len(rows['ids']) == len(rows['documents']) == len(rows['embeddings']) == 1501
    assert rows['documents'][-1] == 'text 1500'
    assert calls == [0, 500, 1000, 1500, 1501]
    assert 'other' not in rows['ids']


def test_paginated_reads_reject_duplicates_and_incomplete_rows():
    from src.chroma_reads import read_collection
    repeated = SimpleNamespace(get=lambda **kwargs: {'ids':['same'], 'documents':['text']})
    with pytest.raises(RuntimeError, match='changed'):
        read_collection(repeated, ['documents'])
    broken = SimpleNamespace(get=lambda **kwargs: {'ids':['same'], 'documents':None})
    with pytest.raises(RuntimeError, match='Incomplete'):
        read_collection(broken, ['documents'])


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


def test_sidecar_rejects_overlapping_or_reordered_page_spans(tmp_path):
    source = tmp_path / 'paper.md'
    sidecar = source.with_suffix('.metadata.json')
    sidecar.write_text(json.dumps({'page_spans': [
        {'page': 1, 'start': 0, 'end': 10},
        {'page': 2, 'start': 9, 'end': 20},
    ]}))
    with pytest.raises(ValueError, match='ordered and non-overlapping'):
        read_sidecar(source)
    sidecar.write_text(json.dumps({'page_spans': [
        {'page': 1, 'start': 0, 'end': 10},
        {'page': 1, 'start': 10, 'end': 20},
    ]}))
    assert len(read_sidecar(source)['page_spans']) == 2
    sidecar.write_text(json.dumps({'page_spans': [
        {'page': 2, 'start': 0, 'end': 10},
        {'page': 1, 'start': 10, 'end': 20},
    ]}))
    with pytest.raises(ValueError, match='ordered and non-overlapping'):
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
            schema = next(t.output_schema for t in tools if t.name == 'search_evidence')
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


def test_section_search_context_subqueries_and_article_first(mcp_library):
    _, text, digest = mcp_library
    strict = server.search_evidence("Snow", alpha=0, section_types=["results"])
    assert [hit.provenance.location.section for hit in strict.hits] == ["Results"]
    assert "strict_section_filter" in " ".join(strict.warnings)

    passage = server.get_passage(
        strict.hits[0].chunk_id, expected_content_sha256=digest,
        context="section", max_context_chars=500,
    )
    assert passage.context is not None
    assert passage.context.text == text[text.index("# Results"):]
    assert passage.context.passage_start == strict.hits[0].provenance.location.char_start

    decomposed = server.search_evidence(
        "Snow", alpha=0, subqueries=["remains"],
        retrieval_strategy="articles_then_passages", article_limit=2,
    )
    assert decomposed.retrieval_strategy == "articles_then_passages"
    assert decomposed.selected_articles == ["paper.md"]
    variable_hit = next(hit for hit in decomposed.hits if "variable" in hit.excerpt)
    assert variable_hit.matched_queries == ["Snow", "remains"]


def test_section_metadata_migration_preserves_ids_text_and_vectors(tmp_path, monkeypatch):
    from scripts import enrich_section_metadata as migration

    collection = Collection()
    text = "# Methods\nMeasured snow.\n\n# Results\nObserved albedo."
    digest = Library(tmp_path / "library").snapshot(text)
    chunks = ["Measured snow.", "Observed albedo."]
    data = payload(chunks, canonical_sha256=digest)
    for metadata, locator in zip(data["metadatas"], locate_chunks(text, chunks, {})):
        metadata.update({key: value for key, value in locator.items()
                         if not key.startswith("section") and key != "structure_version"})
    collection.upsert(**data)
    before = copy.deepcopy(collection.rows)
    synced = {}

    monkeypatch.setattr(migration, "CHROMA_DB_PATH", tmp_path / "chroma")
    monkeypatch.setattr(migration, "LIBRARY_PATH", tmp_path / "library")
    monkeypatch.setattr(migration, "LEXICAL_INDEX_PATH", tmp_path / "lexical.sqlite3")
    monkeypatch.setattr(migration, "acquire_lock", lambda path: object())
    monkeypatch.setattr(migration, "release_lock", lambda path, lock: None)
    monkeypatch.setattr(migration, "open_chroma_client", lambda module, path: (
        SimpleNamespace(get_collection=lambda **kwargs: collection), "test"
    ))

    class Lexical:
        def __init__(self, path):
            self.path = path

        def sync_sources(self, current, sources, revision, allow_repairing=False):
            synced.update(sources=sources, revision=revision, allow_repairing=allow_repairing)
            return {"ready": True}

        def rebuild(self, current, allow_repairing=False):
            synced.update(rebuilt=True, allow_repairing=allow_repairing)
            return {"ready": True}

    monkeypatch.setattr(migration, "PersistentLexicalIndex", Lexical)
    result = migration.enrich(apply=True)

    assert result["changed_sources"] == 1
    assert set(collection.rows) == set(before)
    for chunk_id, row in collection.rows.items():
        assert row["documents"] == before[chunk_id]["documents"]
        assert row["embeddings"] == before[chunk_id]["embeddings"]
        assert row["metadatas"]["structure_version"] == "markdown-sections-v1"
    assert collection.metadata["ragdoc_repairing"] is False
    assert synced["sources"] == {"paper.md"}
    assert synced["allow_repairing"] is True

    collection.metadata["ragdoc_repairing"] = True
    synced.clear()
    resumed = migration.enrich(apply=True)
    assert resumed["changed_sources"] == 0
    assert synced == {"rebuilt": True, "allow_repairing": True}
    assert collection.metadata["ragdoc_repairing"] is False


def test_section_metadata_migration_resume_rebuilds_after_additional_changes(tmp_path, monkeypatch):
    from scripts import enrich_section_metadata as migration

    collection = Collection()
    text = "# Results\nObserved albedo."
    digest = Library(tmp_path / "library").snapshot(text)
    data = payload(["Observed albedo."], canonical_sha256=digest)
    data["metadatas"][0].update({
        key: value for key, value in locate_chunks(text, data["documents"], {})[0].items()
        if not key.startswith("section") and key != "structure_version"
    })
    collection.upsert(**data)
    collection.metadata["ragdoc_repairing"] = True
    calls = {}

    monkeypatch.setattr(migration, "CHROMA_DB_PATH", tmp_path / "chroma")
    monkeypatch.setattr(migration, "LIBRARY_PATH", tmp_path / "library")
    monkeypatch.setattr(migration, "LEXICAL_INDEX_PATH", tmp_path / "lexical.sqlite3")
    monkeypatch.setattr(migration, "acquire_lock", lambda path: object())
    monkeypatch.setattr(migration, "release_lock", lambda path, lock: None)
    monkeypatch.setattr(migration, "open_chroma_client", lambda module, path: (
        SimpleNamespace(get_collection=lambda **kwargs: collection), "test"
    ))

    class Lexical:
        def __init__(self, path):
            pass

        def sync_sources(self, *args, **kwargs):
            calls["synced"] = True

        def rebuild(self, current, allow_repairing=False):
            calls.update(rebuilt=True, allow_repairing=allow_repairing)

    monkeypatch.setattr(migration, "PersistentLexicalIndex", Lexical)
    result = migration.enrich(apply=True)

    assert result["changed_sources"] == 1
    assert calls == {"rebuilt": True, "allow_repairing": True}
    assert collection.metadata["ragdoc_repairing"] is False


def test_section_migration_accepts_real_chroma_embedding_arrays(tmp_path):
    import chromadb
    from scripts.enrich_section_metadata import _same_rows, _vectors

    collection = chromadb.PersistentClient(path=str(tmp_path)).create_collection("migration-vectors")
    collection.add(
        ids=["chunk"], documents=["text"], embeddings=[[1.0, 0.0]],
        metadatas=[{"section_is_results": True}],
    )
    stored = collection.get(ids=["chunk"], include=["embeddings"])["embeddings"]
    assert _vectors(stored) == [[1.0, 0.0]]
    assert _same_rows(
        {"chunk": ("text", {"source": "paper.md"}, [0.123456791, 0.0])},
        {"chunk": ("text", {"source": "paper.md"}, [0.123456789, 0.0])},
    )
    assert not _same_rows(
        {"chunk": ("different", {"source": "paper.md"}, [0.123456791, 0.0])},
        {"chunk": ("text", {"source": "paper.md"}, [0.123456789, 0.0])},
    )
    filtered = collection.query(
        query_embeddings=[[1.0, 0.0]], n_results=1,
        where=server._section_filter(["results"]),
    )
    assert filtered["ids"] == [["chunk"]]


def test_preferred_sections_group_final_results_after_reranking(mcp_library, monkeypatch):
    def rerank(**kwargs):
        methods_first = sorted(
            range(len(kwargs["documents"])),
            key=lambda index: "Section: Methods" not in kwargs["documents"][index],
        )
        return SimpleNamespace(results=[
            SimpleNamespace(index=index, relevance_score=1.0 - position * 0.1)
            for position, index in enumerate(methods_first)
        ])

    monkeypatch.setattr(server, "init_cohere_client", lambda: SimpleNamespace(rerank=rerank))
    result = server.search_evidence(
        "Snow", alpha=0, section_types=["results"], section_mode="prefer"
    )
    assert result.hits[0].provenance.location.section == "Results"
    assert "section_preference_heuristic" in " ".join(result.warnings)


def test_legacy_read_warns_instead_of_claiming_canonical(mcp_library):
    c, _, _ = mcp_library
    for row in c.rows.values():
        row['metadatas'].pop('canonical_sha256')
    result = server.get_document_content('paper.md', format='text')
    assert 'Legacy index' in result


def test_document_content_uses_snapshot_not_overlapping_chunks(mcp_library):
    _, text, _ = mcp_library
    assert server.get_document_content('paper.md', format='text') == text


def test_audit_reports_missing_snapshot_and_catalogue_unknowns(mcp_library):
    c, _, _ = mcp_library
    c.upsert(**payload(['legacy'], ['legacy'], source='legacy.md'))
    audit = server.audit_library()
    assert audit['finding_count'] == 3
    assert audit['findings_by_issue'] == {
        'bibliography_incomplete': 2,
        'legacy_no_canonical_snapshot': 1,
    }
    assert 'items' not in audit
    details = server.audit_library(view='findings', limit=1)
    assert details['total'] == 3
    assert len(details['items']) == 1
    assert details['next_offset'] == 1
    remaining = server.audit_library(view='findings', offset=1, limit=10)
    assert any(f['source'] == 'legacy.md' and f['issue'] == 'legacy_no_canonical_snapshot'
               for f in details['items'] + remaining['items'])
    catalogue = server.search_documents(query='Snow', year_from=2020)
    assert catalogue['total'] == 1


def test_audit_summary_stays_bounded_and_detail_views_validate(mcp_library):
    c, _, _ = mcp_library
    for index in range(150):
        c.upsert(**payload([f'legacy {index}'], [f'legacy-{index}'], source=f'legacy-{index}.md'))
    summary = server.audit_library()
    assert summary['documents'] == 151
    assert summary['finding_count'] == 301
    assert len(json.dumps(summary)) < 2000
    with pytest.raises(Exception, match='view must be'):
        server.audit_library(view='everything')
    with pytest.raises(Exception, match='limit'):
        server.audit_library(view='findings', limit=101)


def test_duplicate_detail_paginates_pairs_not_unbounded_source_groups(mcp_library):
    c, _, digest = mcp_library
    for index in range(150):
        c.upsert(**payload(
            [f'duplicate {index}'], [f'duplicate-{index}'],
            source=f'duplicate-{index}.md', canonical_sha256=digest,
        ))
    page = server.audit_library(view='duplicates', limit=1)
    assert page['candidate_duplicate_group_count'] == 1
    assert page['candidate_duplicate_pair_count'] == 150
    assert page['total'] == 150
    assert len(page['items']) == 1
    assert len(page['items'][0]['sources']) == 2
    assert page['items'][0]['group_size'] == 151


def test_citation_readiness_requires_canonical_exact_page_locator(mcp_library):
    c, _, _ = mcp_library
    library = Library(server.LIBRARY_PATH)

    page_text = 'Page located evidence.'
    page_digest = library.snapshot(page_text)
    c.upsert(**payload(
        [page_text], ['page-ready'], source='page-ready.md', canonical_sha256=page_digest,
        locator_status='exact', char_start=0, char_end=len(page_text), page_start=4, page_end=4,
    ))

    partial_text = 'Located.\nText only.'
    partial_digest = library.snapshot(partial_text)
    partial = payload(
        ['Located.', 'Text only.'], ['partial-0', 'partial-1'], source='partial.md',
        canonical_sha256=partial_digest,
    )
    partial['metadatas'][0].update(
        locator_status='exact', char_start=0, char_end=len('Located.'), page_start=2, page_end=2,
    )
    partial['metadatas'][1].update(
        locator_status='exact', char_start=partial_text.index('Text only.'),
        char_end=len(partial_text),
    )
    c.upsert(**partial)
    c.upsert(**payload(['legacy'], ['citation-legacy'], source='citation-legacy.md'))

    audit = server.audit_citation_readiness(limit=2)
    assert audit['documents'] == 4
    assert audit['status_counts'] == {
        'canonical_text_only': 1,
        'fully_page_verifiable': 1,
        'partially_page_verifiable': 1,
        'unverified_legacy': 1,
    }
    assert audit['criteria']['doi_alone_is_sufficient'] is False
    assert audit['total'] == 4
    assert audit['next_offset'] == 2

    filtered = server.audit_citation_readiness(status='fully_page_verifiable')
    assert filtered['total'] == 1
    assert filtered['items'][0]['source'] == 'page-ready.md'
    assert filtered['items'][0]['page_coverage'] == 1.0
    with pytest.raises(Exception, match='unknown citation readiness status'):
        server.audit_citation_readiness(status='ready')


def test_citation_readiness_does_not_trust_page_metadata_without_exact_offsets(mcp_library):
    c, _, _ = mcp_library
    text = 'Evidence with unverified offsets.'
    digest = Library(server.LIBRARY_PATH).snapshot(text)
    c.upsert(**payload(
        [text], ['false-page'], source='false-page.md', canonical_sha256=digest,
        locator_status='exact', page_start=7, page_end=7,
    ))
    row = server.audit_citation_readiness(status='canonical_text_only')['items'][0]
    assert row['source'] == 'false-page.md'
    assert row['page_located_chunks'] == 1
    assert row['exact_locator_chunks'] == 0
    assert row['page_verified_chunks'] == 0


@pytest.mark.parametrize('tool', ['audit_library', 'audit_citation_readiness', 'get_indexation_status'])
def test_diagnostics_reject_revision_change_during_scan(mcp_library, monkeypatch, tool):
    c, _, _ = mcp_library
    original = server.read_collection
    def changing(*args, **kwargs):
        result = original(*args, **kwargs)
        c.metadata['ragdoc_revision'] = 'updated_during_scan'
        return result
    monkeypatch.setattr(server, 'read_collection', changing)
    with pytest.raises(Exception, match='Index changed'):
        getattr(server, tool)()


def test_corrupt_snapshot_cannot_be_presented_as_verified(mcp_library, tmp_path):
    _, _, digest = mcp_library
    (tmp_path / 'snapshots' / f'{digest}.md').write_text('broken')
    passage = server.get_passage('c0')
    assert not passage.canonical_verified
    assert 'canonical_snapshot_unavailable_or_corrupt' in passage.warnings
    with pytest.raises(Exception, match='snapshot unavailable or corrupt'):
        server.read_document('paper.md')


def test_source_diversity_and_rerank_service_failure(mcp_library, monkeypatch):
    c, _, _ = mcp_library
    c.upsert(**payload(['Snow impurity observation.'], ['other'], source='other.md'))
    monkeypatch.setattr(server, 'init_cohere_client', lambda: SimpleNamespace(
        rerank=Mock(side_effect=RuntimeError('service outage'))))
    result = server.search_evidence('Snow', alpha=0, max_per_document=1)
    assert len(result.hits) == 2
    assert len({h.provenance.source for h in result.hits}) == 2
    assert result.reranking == 'unavailable'


@pytest.mark.parametrize('term', ['authors', 'null', 'year'])
def test_catalogue_does_not_search_json_keys(mcp_library, term):
    assert server.search_documents(query=term)['total'] == 0


def test_catalogue_cache_reuses_scan_and_refreshes_revision(mcp_library, monkeypatch):
    c, _, _ = mcp_library
    scan = Mock(wraps=server.read_collection)
    monkeypatch.setattr(server, 'read_collection', scan)
    assert server.search_documents(query='Snow')['total'] == 1
    server.search_documents(offset=0, limit=1)
    assert scan.call_count == 1
    c.upsert(**payload(['new'], ['new'], source='new.md'))
    c.metadata['ragdoc_revision'] = 'new-revision'
    assert server.search_documents()['total'] == 2
    assert scan.call_count == 2
    c.metadata['ragdoc_write_state'] = 'writing'
    with pytest.raises(Exception, match='incomplete'):
        server.search_documents()


def test_catalogue_never_caches_unversioned_collection(mcp_library, monkeypatch):
    c, _, _ = mcp_library
    c.metadata.pop('ragdoc_revision')
    scan = Mock(wraps=server.read_collection)
    monkeypatch.setattr(server, 'read_collection', scan)
    server.search_documents()
    server.search_documents()
    assert scan.call_count == 2


def test_catalogue_rejects_revision_change_before_cache_publish(mcp_library, monkeypatch):
    c, _, _ = mcp_library
    original = server.read_collection
    def changing(*args, **kwargs):
        result = original(*args, **kwargs)
        c.metadata['ragdoc_revision'] = 'changed'
        return result
    monkeypatch.setattr(server, 'read_collection', changing)
    with pytest.raises(Exception, match='Index changed'):
        server.search_documents()


def test_search_can_return_one_hundred_hits(mcp_library, monkeypatch):
    hits = [{'id': f'h{i}', 'text': 'snow', 'metadata': {'source': f'p{i}.md'},
             'score': 1 / (i + 1)} for i in range(120)]
    retrieve = Mock(side_effect=lambda **kw: hits[:kw['top_k']])
    monkeypatch.setattr(server, 'hybrid_retriever', SimpleNamespace(search=retrieve, last_status={}))
    result = server.search_evidence('snow', top_k=100, alpha=0)
    assert len(result.hits) == 100


def test_search_expands_candidates_for_source_diversity(mcp_library, monkeypatch):
    hits = [{'id': f'h{i}', 'text': 'snow',
             'metadata': {'source': 'dominant.md' if i < 180 else f'p{i}.md'},
             'score': 1 / (i + 1)} for i in range(210)]
    retrieve = Mock(side_effect=lambda **kw: hits[:kw['top_k']])
    rerank = Mock(side_effect=RuntimeError('offline'))
    monkeypatch.setattr(server, 'hybrid_retriever', SimpleNamespace(search=retrieve, last_status={}))
    monkeypatch.setattr(server, 'init_cohere_client', lambda: SimpleNamespace(rerank=rerank))
    result = server.search_evidence('snow', top_k=10, alpha=0, max_per_document=2)
    assert len(result.hits) == 10
    assert sum(h.provenance.source == 'dominant.md' for h in result.hits) <= 2
    assert retrieve.call_args.kwargs['top_k'] > 100
    assert len(rerank.call_args.kwargs['documents']) <= 100
    assert rerank.call_args.kwargs['documents'][0].startswith('Source: dominant.md\n')


def test_runtime_diagnostics_are_offline(monkeypatch):
    monkeypatch.setattr(server, 'init_chroma_client', Mock(side_effect=AssertionError('must not connect')))
    status = server.get_runtime_status()
    assert status['offline_check_only']
    assert 'versions' in status


def test_french_query_expansion_handles_accents():
    assert any('remote sensing' in q for q in server._generate_query_variants('télédétection de la neige', 5))


def test_benchmark_refuses_unreviewed_or_unpinned_judgments():
    from scripts.benchmark_scientific import validate, evaluate, score_hits, summarize_rows
    draft = {'questions': [{'id': 'q1', 'query': 'snow?'}], 'judgments': {}}
    assert validate(draft)['ready_to_score'] is False
    with pytest.raises(ValueError, match='reviewed'):
        asyncio.run(evaluate(draft, 10))
    draft['judgments']['q1'] = {'reviewed': True, 'answerable': True,
        'review_provenance': 'human', 'reviewer': 'Test Reviewer',
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
    summary = summarize_rows([
        {'answerable': True, 'article_recall': 1.0, 'passage_recall': 0.5,
         'reciprocal_rank': 1.0, 'canonical_verified_passages': 2, 'passages_checked': 2},
        {'answerable': False, 'returned_candidates': True,
         'canonical_verified_passages': 1, 'passages_checked': 2},
    ])
    assert summary['mean_passage_recall'] == 0.5
    assert summary['candidate_rate_without_canonical_evidence'] == 1.0
    assert summary['canonical_verification_rate'] == 0.75

    assistant_draft = {'questions': [{'id': 'q1', 'query': 'snow?'}], 'judgments': {
        'q1': {'reviewed': False, 'assistant_adjudicated': True, 'answerable': False,
               'judgment_scope': 'canonical_evidence',
               'canonical_evidence_available': False,
               'corpus_status': 'not_established_after_targeted_search',
               'sources': [], 'chunks': [], 'content_sha256_by_source': {}}}}
    assert validate(assistant_draft)['ready_for_provisional_score'] is False
    provisional = validate(assistant_draft, allow_assistant_draft=True)
    assert provisional['ready_to_score'] is False
    assert provisional['ready_for_provisional_score'] is True


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
                         'review_provenance': 'human', 'reviewer': 'Test Reviewer',
                         'content_sha256_by_source': {'paper.md': digest}},
            'negative': {'reviewed': True, 'answerable': False, 'sources': [], 'chunks': [],
                         'review_provenance': 'human', 'reviewer': 'Test Reviewer',
                         'content_sha256_by_source': {}}}}
    result = asyncio.run(evaluate(dataset, 10))
    assert len(result['dataset_sha256']) == len(result['runner_code_sha256']) == 64
    assert result['server_code_sha256'] is None
    assert result['result_status'] == 'human_reviewed'
    assert result['rows'][0]['passage_recall'] == 1.0
    assert result['rows'][0]['canonical_verified_passages'] == 2
    assert result['rows'][1]['returned_candidates'] is False
    assert result['summary']['questions'] == 2
    assert result['splits']['unspecified']['canonical_evidence_questions'] == 1
    assert result['mcp_transport'] == 'in_process'


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
    chunker = SimpleNamespace(chunk=lambda text: [] if failure == 'empty_chunks' else [chunk])
    monkeypatch.setattr(indexer, 'TokenChunker', lambda **kw: chunker)
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


def test_real_chroma_preserves_legacy_hnsw_configuration_after_reopen(tmp_path):
    import chromadb
    from chromadb.config import Settings
    path = tmp_path / 'chroma_hnsw'
    settings = Settings(anonymized_telemetry=False)
    client = chromadb.PersistentClient(path=str(path), settings=settings)
    c = client.create_collection(
        'hnsw_migration_test',
        embedding_function=None,
        metadata={'hnsw:space': 'cosine', 'hnsw:M': 32, 'pipeline': 'legacy'},
    )
    data = payload(['same direction', 'orthogonal'], ['parallel', 'orthogonal'])
    data['embeddings'] = [[10.0, 0.0], [0.0, 10.0]]
    replace_document(c, 'paper.md', data)

    reopened = chromadb.PersistentClient(path=str(path), settings=settings).get_collection(
        'hnsw_migration_test', embedding_function=None
    )
    assert reopened.configuration['hnsw']['space'] == 'cosine'
    assert reopened.configuration['hnsw']['max_neighbors'] == 32
    assert json.loads(reopened.metadata['ragdoc_preserved_hnsw_json']) == {
        'hnsw:M': 32,
        'hnsw:space': 'cosine',
    }
    result = reopened.query(query_embeddings=[[1.0, 0.0]], n_results=1, include=['distances'])
    assert result['ids'][0] == ['parallel']
    assert result['distances'][0][0] == pytest.approx(0.0, abs=1e-6)
