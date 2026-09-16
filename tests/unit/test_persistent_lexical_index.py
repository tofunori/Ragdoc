import copy

import pytest

from src.hybrid_retriever import HybridRetriever
from src.lexical_index import PersistentLexicalIndex


class Collection:
    def __init__(self):
        self.metadata = {
            "ragdoc_revision": "r1",
            "ragdoc_write_state": "ready",
            "ragdoc_repairing": False,
        }
        self.rows = [
            ("c1", "The RAQDPSFW system incorporates near-real-time wildfire emissions.",
             {"source": "fire.md", "doi": "10.1000/fire"}),
            ("c2", "Glacier albedo decreases when light absorbing particles accumulate.",
             {"source": "ice.md", "doi": "10.1000/ice"}),
        ]

    def count(self):
        return len(self.rows)

    def get(self, ids=None, include=None, where=None, limit=None, offset=0):
        rows = [row for row in self.rows
                if (ids is None or row[0] in ids)
                and (where is None or row[2].get("source") == where.get("source"))]
        if ids is None:
            rows = rows[offset:offset + limit] if limit is not None else rows[offset:]
        return {
            "ids": [row[0] for row in rows],
            "documents": [row[1] for row in rows],
            "metadatas": [copy.deepcopy(row[2]) for row in rows],
        }


def test_persistent_lexical_index_is_revision_pinned_and_used_without_ram_corpus(tmp_path):
    collection = Collection()
    index = PersistentLexicalIndex(tmp_path / "lexical.sqlite3")

    status = index.rebuild(collection)
    assert status["ready"]
    assert status["schema_version"] == "3"
    assert status["ranking"] == "fielded_bm25"
    assert status["max_chunks_per_source"] == 10
    results, payload = index.search(
        "RAQDPSFW wildfire", top_n=10, revision="r1", chunk_count=2
    )
    assert results[0][0] == "c1"
    assert payload["c1"][1]["source"] == "fire.md"
    doi_results, _ = index.search(
        "10.1000/fire", top_n=10, revision="r1", chunk_count=2
    )
    assert doi_results[0][0] == "c1"

    retriever = HybridRetriever(collection, use_advanced_tokenizer=False, lexical_index=index)
    hit = retriever.search("RAQDPSFW", alpha=0, top_k=1)[0]
    assert hit["id"] == "c1"
    assert retriever.last_status["mode"] == "lexical"
    assert retriever.last_status["persistent_lexical_ready"]
    assert retriever.docs == []

    with pytest.raises(RuntimeError, match="revision_or_count_mismatch"):
        index.search("RAQDPSFW", top_n=10, revision="stale", chunk_count=2)


def test_targeted_sync_replaces_changed_source_and_advances_revision(tmp_path):
    collection = Collection()
    index = PersistentLexicalIndex(tmp_path / "lexical.sqlite3")
    index.rebuild(collection)

    collection.rows[0] = (
        "c3",
        "The updated system tracks CFFEPS plume emissions.",
        {"source": "fire.md", "doi": "10.1000/fire"},
    )
    collection.metadata["ragdoc_revision"] = "r2"
    status = index.sync_sources(collection, {"fire.md"}, previous_revision="r1")

    assert status["ready"]
    assert status["revision"] == "r2"
    results, _ = index.search("CFFEPS", top_n=10, revision="r2", chunk_count=2)
    assert [row[0] for row in results] == ["c3"]
    old, _ = index.search("RAQDPSFW", top_n=10, revision="r2", chunk_count=2)
    assert old == []


def test_fielded_bm25_prefers_catalogue_metadata_over_incidental_body_mentions(tmp_path):
    collection = Collection()
    collection.rows = [
        (
            "target",
            "This is the canonical article.",
            {
                "source": "target.md",
                "title": "Snow Albedo Retrieval from MODIS",
                "doi": "10.1234/target",
                "bibliography_json": '{"authors":["Christophe Kinnard"]}',
            },
        ),
        (
            "citation",
            "Snow Albedo Retrieval from MODIS cites 10.1234/target repeatedly: "
            "10.1234/target 10.1234/target.",
            {"source": "citation.md", "title": "A review article"},
        ),
    ]
    index = PersistentLexicalIndex(tmp_path / "lexical.sqlite3")
    index.rebuild(collection)

    title, _ = index.search(
        "Snow Albedo Retrieval from MODIS", top_n=10, revision="r1", chunk_count=2
    )
    doi, _ = index.search(
        "10.1234/target", top_n=10, revision="r1", chunk_count=2
    )
    author, _ = index.search(
        "Christophe Kinnard", top_n=10, revision="r1", chunk_count=2
    )

    assert title[0][0] == "target"
    assert doi[0][0] == "target"
    assert author[0][0] == "target"


def test_lexical_candidates_are_diverse_and_skip_macos_sidecars(tmp_path):
    collection = Collection()
    collection.rows = [
        (f"dominant-{index}", "glacier", {"source": "dominant.md", "title": "Glacier"})
        for index in range(30)
    ]
    collection.rows.extend(
        (f"other-{index}", "glacier", {"source": f"other-{index}.md", "title": "Glacier"})
        for index in range(15)
    )
    collection.rows.append(
        ("sidecar", "glacier", {"source": "._Finder-sidecar.md", "title": "Glacier"})
    )
    index = PersistentLexicalIndex(tmp_path / "lexical.sqlite3")
    index.rebuild(collection)

    results, payload = index.search(
        "glacier", top_n=20, revision="r1", chunk_count=46
    )
    sources = [payload[chunk_id][1]["source"] for chunk_id, _, _ in results]

    assert len(results) == 20
    assert sources.count("dominant.md") <= 10
    assert len(set(sources)) >= 11
    assert "._Finder-sidecar.md" not in sources
