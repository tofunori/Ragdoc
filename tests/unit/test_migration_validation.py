from types import SimpleNamespace

from scripts.migrate_embedding_collection import _embed_group, source_is_identical


class Collection:
    def __init__(self, rows):
        self.rows = rows

    def get(self, **kwargs):
        source = (kwargs.get("where") or {}).get("source")
        selected = [row for row in self.rows if row[2].get("source") == source]
        offset = kwargs.get("offset", 0)
        limit = kwargs.get("limit", len(selected))
        selected = selected[offset:offset + limit]
        return {
            "ids": [row[0] for row in selected],
            "documents": [row[1] for row in selected],
            "metadatas": [row[2] for row in selected],
        }


def test_resume_requires_identical_text_and_provenance():
    source = Collection([
        ("paper-0", "current text", {"source": "paper.md", "doi": "10.1/current", "model": "voyage-context-3"})
    ])
    stale = Collection([
        ("paper-0", "current text", {"source": "paper.md", "doi": "10.1/old", "model": "voyage-context-4"})
    ])
    exact = Collection([
        ("paper-0", "current text", {
            "source": "paper.md",
            "doi": "10.1/current",
            "model": "voyage-context-4",
            "embedding_migrated_at": "later",
            "embedding_source_collection": "source",
        })
    ])

    assert not source_is_identical(source, stale, "paper.md", "voyage-context-4")
    assert source_is_identical(source, exact, "paper.md", "voyage-context-4")


def test_oversize_context_group_is_split_without_losing_vector_order():
    class Client:
        def contextualized_embed(self, *, inputs, **kwargs):
            group = inputs[0]
            if len(group) > 2:
                raise RuntimeError("too many tokens for context window")
            vectors = [[float(text)] for text in group]
            return SimpleNamespace(
                results=[SimpleNamespace(embeddings=vectors)],
                total_tokens=len(group),
            )

    vectors, tokens = _embed_group(Client(), ["1", "2", "3", "4", "5"], "voyage-context-4")
    assert vectors == [[1.0], [2.0], [3.0], [4.0], [5.0]]
    assert tokens == 5
