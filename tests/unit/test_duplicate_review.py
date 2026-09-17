import asyncio
from types import SimpleNamespace

import pytest

from scripts.export_duplicate_review import _duplicates, exact_rows, fuzzy_rows


def document(source, title=None, authors=None, year=None, doi=None):
    return {
        "source": source,
        "bibliography": {"title": title, "authors": authors, "year": year, "doi": doi},
    }


def test_exact_duplicate_prefers_complete_zotero_record():
    documents = [
        document("Legacy_2020_Title.md"),
        document("zotero_A_B.md", "A useful title", ["A. Author"], 2020, "10.1/example"),
    ]
    rows, pairs = exact_rows(documents, [{
        "key": "doi:10.1/example", "sources": [item["source"] for item in documents]
    }])
    assert rows[0]["preferred_source"] == "zotero_A_B.md"
    assert rows[0]["confidence"] == "high"
    assert len(pairs) == 1


def test_dot_underscore_artifact_matches_exact_original_filename():
    documents = [
        document("._Fiddes_2022_TopoCLIM.md"),
        document("Fiddes_2022_TopoCLIM.md"),
    ]
    rows, pairs = exact_rows(documents, [])
    assert rows[0]["preferred_source"] == "Fiddes_2022_TopoCLIM.md"
    assert rows[0]["match_basis"] == "macos_dot_underscore_filename"
    assert rows[0]["confidence"] == "high"
    assert len(pairs) == 1


def test_flattened_duplicate_pairs_reconstruct_group_before_selecting_preferred():
    documents = [
        document("legacy-a.md"),
        document("legacy-b.md", "A title", ["A. Author"], 2020),
        document("zotero-c.md", "A title", ["A. Author"], 2020, "10.1/example"),
    ]
    flattened = [
        {"key": "content:x", "sources": ["legacy-a.md", "legacy-b.md"], "group_size": 3},
        {"key": "content:x", "sources": ["legacy-a.md", "zotero-c.md"], "group_size": 3},
    ]
    rows, _pairs = exact_rows(documents, flattened)
    assert {(row["candidate_source"], row["preferred_source"]) for row in rows} == {
        ("legacy-a.md", "zotero-c.md"),
        ("legacy-b.md", "zotero-c.md"),
    }


def test_fuzzy_duplicate_requires_year_and_explainable_title_overlap():
    documents = [
        document("AubryWake_2022_Fire_and_Ice_Wildfire_Albedo.md"),
        document(
            "zotero_A_B.md",
            "Fire and ice: wildfire affected albedo and glacier melt",
            ["Caroline Aubry-Wake"],
            2022,
            "10.1/example",
        ),
        document("zotero_C_D.md", "Unrelated glacier paper", ["Other Author"], 2023, "10.2/other"),
    ]
    rows = fuzzy_rows(documents, set())
    assert len(rows) == 1
    assert rows[0]["preferred_source"] == "zotero_A_B.md"
    assert rows[0]["confidence"] == "medium"
    assert rows[0]["title_token_overlap"] >= 4


def test_fuzzy_duplicate_marks_close_same_year_candidates_for_review():
    documents = [
        document("Smith_2020_Glacier_Albedo.md"),
        document("zotero_A.md", "Glacier albedo change", ["A. Smith"], 2020, "10.1/a"),
        document("zotero_B.md", "Glacier albedo variability", ["A. Smith"], 2020, "10.1/b"),
    ]
    rows = fuzzy_rows(documents, set())
    assert rows[0]["confidence"] == "review"
    assert rows[0]["ambiguous_candidates"]


def test_duplicate_fetch_does_not_hide_transient_mcp_failure():
    class Client:
        async def call_tool(self, _name, _arguments):
            raise RuntimeError("connection reset")

    with pytest.raises(RuntimeError, match="connection reset"):
        asyncio.run(_duplicates(Client()))


def test_duplicate_fetch_accepts_explicit_legacy_schema_only():
    class Client:
        def __init__(self):
            self.calls = 0

        async def call_tool(self, _name, _arguments):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("unexpected keyword argument 'view'")
            return SimpleNamespace(structured_content={
                "candidate_duplicates": [{"key": "doi:x", "sources": ["a", "b"]}]
            })

    groups = asyncio.run(_duplicates(Client()))
    assert groups == [{"key": "doi:x", "sources": ["a", "b"]}]
