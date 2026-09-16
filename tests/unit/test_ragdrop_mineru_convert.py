import io
import json
import shutil
import uuid
import zipfile

import scripts.ragdrop_mineru_convert as mineru
from scripts.ragdrop_mineru_convert import build_manifest, extract_archive


def test_manifest_preserves_table_locator_and_image():
    content = [{
        "type": "table",
        "img_path": "images/table.jpg",
        "table_caption": ["Table 2. Model results."],
        "table_body": "<table><tr><td>P1</td></tr></table>",
        "bbox": [10, 20, 300, 400],
        "page_idx": 6,
    }]

    manifest = build_manifest(content, "ren.md", {"table.jpg"})

    assert manifest["source"] == "ren.md"
    assert manifest["artifacts"] == [{
        "artifact_id": manifest["artifacts"][0]["artifact_id"],
        "type": "table",
        "label": "Table 2",
        "page": 7,
        "bbox": [10, 20, 300, 400],
        "caption": "Table 2. Model results.",
        "body": "<table><tr><td>P1</td></tr></table>",
        "image": "assets/table.jpg",
    }]


def test_manifest_keeps_artifact_when_archive_has_no_image():
    manifest = build_manifest(
        [{"type": "chart", "chart_caption": ["Figure 4. Trends"], "page_idx": 2}],
        "paper.md",
        set(),
    )

    assert manifest["artifacts"][0]["label"] == "Figure 4"
    assert manifest["artifacts"][0]["page"] == 3
    assert manifest["artifacts"][0]["image"] is None


def test_archive_keeps_referenced_mineru_image_and_content_list():
    output_name = f"ragdrop_test_{uuid.uuid4().hex}"
    payload = io.BytesIO()
    content = [{
        "type": "table", "img_path": "images/table.jpg",
        "table_caption": ["Table 2. Results"], "table_body": "<table></table>",
        "page_idx": 1, "bbox": [1, 2, 3, 4],
    }]
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("paper.md", "# Paper\n\n![Table 2](images/table.jpg)\n")
        archive.writestr("paper_content_list.json", json.dumps(content))
        archive.writestr("images/table.jpg", b"jpeg-data")

    markdown, bundle, count = extract_archive(payload.getvalue(), output_name)
    try:
        manifest = json.loads((bundle / "manifest.json").read_text())
        assert count == 1
        assert (bundle / "assets/table.jpg").read_bytes() == b"jpeg-data"
        assert (bundle / "content_list.json").is_file()
        assert manifest["artifacts"][0]["image"] == "assets/table.jpg"
        assert "[Figure: Table 2]" in markdown.read_text()
    finally:
        markdown.unlink(missing_ok=True)
        shutil.rmtree(bundle, ignore_errors=True)


def test_submit_uses_current_mineru_v4_request(monkeypatch, tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"pdf")
    request = {}

    class Response:
        def __init__(self, body=None):
            self.body = body

        def raise_for_status(self):
            return None

        def json(self):
            return self.body

    def post(url, **kwargs):
        request.update(url=url, **kwargs)
        return Response({"code": 0, "data": {"batch_id": "batch", "file_urls": ["https://upload"]}})

    monkeypatch.setattr(mineru, "needs_ocr", lambda _: False)
    monkeypatch.setattr(mineru, "extraction_language", lambda _: "en")
    monkeypatch.setattr(mineru.requests, "post", post)
    monkeypatch.setattr(mineru.requests, "put", lambda *args, **kwargs: Response())

    assert mineru.submit(pdf, "token") == "batch"
    assert request["url"].endswith("/api/v4/file-urls/batch")
    assert request["json"]["model_version"] == "vlm"
    assert request["json"]["language"] == "en"
    assert "layout_model" not in request["json"]
