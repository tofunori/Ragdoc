from pathlib import Path

import pytest

from scripts.index_incremental import open_chroma_client


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
