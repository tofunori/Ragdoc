import json
from pathlib import Path
from unittest.mock import Mock

import scripts.import_zotero_mineru as importer
from scripts.import_zotero_mineru import deduplicate, page_spans_from_content


def test_page_spans_never_search_backward():
    markdown = "First\nSecond\nThird\n"
    items = [
        {"page_idx": 0, "text": "First"},
        {"page_idx": 1, "text": "Third"},
        {"page_idx": 2, "text": "Second"},
    ]
    assert page_spans_from_content(items, markdown) == [
        {"page": 1, "start": 0, "end": 5},
        {"page": 2, "start": 13, "end": 18},
    ]


def test_binary_deduplication_prefers_bibliographic_attachment():
    common = {"md5": "same", "path": "/tmp/a.pdf", "date_added": "2026-01-01"}
    standalone = {**common, "attachment_key": "STANDALONE", "parent_key": None}
    linked = {**common, "attachment_key": "LINKED", "parent_key": "PARENT"}
    chosen, duplicates = deduplicate([standalone, linked])
    assert chosen[0]["attachment_key"] == "LINKED"
    assert duplicates == [{"attachment_key": "STANDALONE", "duplicate_of": "LINKED"}]


def test_batch_submission_uses_current_mineru_options(monkeypatch):
    request = {}

    class Response:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"code": 0, "data": {"batch_id": "batch", "file_urls": ["upload"]}}

    def post(url, **kwargs):
        request.update(url=url, **kwargs)
        return Response()

    monkeypatch.setattr(importer.requests, "post", post)
    batch_id, urls = importer.submit_batch(
        [{"path": "/tmp/paper.pdf", "attachment_key": "ABC123", "text_layer": True}],
        "token",
        "vlm",
    )

    assert (batch_id, urls) == ("batch", ["upload"])
    assert request["json"]["model_version"] == "vlm"
    assert request["json"]["language"] == "en"
    assert request["json"]["enable_formula"] is True
    assert request["json"]["enable_table"] is True


def test_pdf_over_six_hundred_pages_is_split_instead_of_skipped(monkeypatch):
    split = Mock()
    monkeypatch.setattr(
        importer,
        "output_paths",
        lambda _: (Path("/tmp/long.md"), Path("/tmp/long.metadata.json")),
    )
    monkeypatch.setattr(importer, "refresh_existing", lambda _: False)
    monkeypatch.setattr(importer, "import_split_record", split)
    record = {
        "attachment_key": "LONGPDF",
        "path": "/tmp/long.pdf",
        "date_added": "2026-09-16",
        "exists": True,
        "pages": 601,
    }

    importer.import_records([record], limit=None)

    split.assert_called_once()
    assert split.call_args.args[2] == 601
