from pathlib import Path

import pytest

from scripts.index_incremental import open_chroma_client, remove_empty_chunks


class FakeHttp:
    def heartbeat(self):
        raise AssertionError("HTTP must not be tried in persistent mode")


class FakeChroma:
    def __init__(self):
        self.http_calls = 0
        self.persistent_paths = []

    def HttpClient(self, **kwargs):
        self.http_calls += 1
        return FakeHttp()

    def PersistentClient(self, *, path):
        self.persistent_paths.append(path)
        return {"path": path}


def test_persistent_mode_never_probes_http():
    chroma = FakeChroma()
    client, mode = open_chroma_client(chroma, Path("/safe/copy"), "persistent")
    assert mode == "persistent-forced"
    assert client == {"path": "/safe/copy"}
    assert chroma.http_calls == 0


def test_invalid_mode_is_rejected():
    with pytest.raises(RuntimeError, match="auto.*persistent"):
        open_chroma_client(FakeChroma(), Path("/safe/copy"), "remote")


def test_mcp_respects_explicit_local_mode(monkeypatch):
    from src import server
    chroma = FakeChroma()
    monkeypatch.setenv("RAGDOC_CHROMA_MODE", "persistent")
    monkeypatch.setattr(server, 'chroma_client', None)
    monkeypatch.setattr(server, '_get_chromadb', lambda: chroma)
    monkeypatch.setattr(server, 'ACTIVE_DB_PATH', Path('/safe/copy'))
    assert server.init_chroma_client() == {"path": "/safe/copy"}
    assert chroma.http_calls == 0


def test_explicit_http_failure_does_not_open_local_store():
    chroma = FakeChroma()
    with pytest.raises(AssertionError):
        open_chroma_client(chroma, Path('/safe/copy'), 'http')
    assert chroma.persistent_paths == []


def test_mcp_rejects_collection_with_different_embedding_model(monkeypatch):
    from src import server

    class Collection:
        metadata = {"embedding_model": "voyage-context-3"}

    class Client:
        def get_collection(self, *, name):
            return Collection()

    monkeypatch.setattr(server, "hybrid_retriever", None)
    monkeypatch.setattr(server, "chroma_client", Client())
    monkeypatch.setattr(server, "EMBEDDING_MODEL", "voyage-context-4")

    with pytest.raises(RuntimeError, match="Embedding model mismatch"):
        server.init_retriever()


def test_empty_parser_chunks_are_removed_before_embedding():
    class Chunk:
        def __init__(self, text, token_count):
            self.text = text
            self.token_count = token_count

    chunks = [Chunk("scientific text", 2), Chunk("", 1024), Chunk(" \n", 8)]

    filtered = remove_empty_chunks(chunks)

    assert [chunk.text for chunk in filtered] == ["scientific text"]
