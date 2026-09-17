from src.hybrid_retriever import HybridRetriever


class LargeCollection:
    metadata = {"ragdoc_write_state": "ready", "ragdoc_revision": "one"}

    def count(self):
        return 70_522


def test_large_corpus_skips_in_memory_bm25(monkeypatch):
    monkeypatch.setenv("RAGDOC_BM25_MAX_CHUNKS", "50000")
    retriever = HybridRetriever(LargeCollection(), use_advanced_tokenizer=False)

    assert not retriever.ensure_bm25_index(background=False)
    assert retriever.bm25 is None
    assert "70522" in retriever._bm25_disabled_reason
