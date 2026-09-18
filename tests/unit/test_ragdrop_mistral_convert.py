import base64
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.ragdrop_mistral_convert as mistral


def response_payload() -> dict:
    pixel = base64.b64encode(b"fake-png").decode("ascii")
    return {
        "model": "mistral-ocr-2512",
        "pages": [
            {
                "index": 0,
                "markdown": "# Methods\n\n$$x = y \\tag{1}$$\n\n![](img-1.png)",
                "blocks": [
                    {"type": "image", "image_id": "img-1.png", "coordinates": [1, 2, 3, 4]},
                    {"type": "caption", "content": "Experimental setup", "coordinates": [1, 5, 3, 6]},
                ],
                "images": [{"id": "img-1.png", "image_base64": pixel}],
                "tables": [],
            },
            {
                "index": 1,
                "markdown": "# Results\n\n[table-1.html](table-1.html)",
                "blocks": [
                    {"type": "table", "table_id": "table-1.html", "coordinates": [2, 3, 8, 9]},
                    {"type": "caption", "content": "Model results", "coordinates": [2, 10, 8, 11]},
                ],
                "images": [],
                "tables": [{"id": "table-1.html", "content": "<table><tr><td>42</td></tr></table>"}],
            },
        ],
    }


def test_materialize_preserves_pages_equations_tables_and_images(tmp_path):
    output, bundle, artifact_count = mistral.materialize(response_payload(), "paper", tmp_path)

    markdown = output.read_text(encoding="utf-8")
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    scrubbed = (bundle / "mistral-response.json").read_text(encoding="utf-8")

    assert "\\tag{1}" in markdown
    assert "<table><tr><td>42</td></tr></table>" in markdown
    assert "[Figure: Experimental setup]" in markdown
    assert manifest["source"] == "paper.md"
    assert manifest["parser"] == "mistral-ocr"
    assert [span["page"] for span in manifest["page_spans"]] == [1, 2]
    assert all(markdown[span["start"]:span["end"]] for span in manifest["page_spans"])
    assert artifact_count == 2
    assert (bundle / manifest["artifacts"][0]["image"]).read_bytes() == b"fake-png"
    assert "fake-png" not in scrubbed
    assert "<saved separately>" in scrubbed


def test_image_caption_with_latex_is_inserted_literally():
    markdown = "![](figure.png)"
    caption = r"Absorption coefficient $\kappa$"

    assert mistral._replace_image_reference(markdown, "figure.png", caption) == (
        r"[Figure: Absorption coefficient $\kappa$]"
    )


def test_convert_uploads_runs_ocr_and_deletes_file(monkeypatch, tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")
    calls = []
    monkeypatch.setattr(mistral, "upload_pdf", lambda path, key: calls.append(("upload", path, key)) or "file-1")
    monkeypatch.setattr(mistral, "signed_url", lambda file_id, key: calls.append(("url", file_id, key)) or "https://signed")
    monkeypatch.setattr(mistral, "run_ocr", lambda url, key: calls.append(("ocr", url, key)) or response_payload())
    monkeypatch.setattr(mistral, "delete_remote_file", lambda file_id, key: calls.append(("delete", file_id, key)))

    assert mistral.convert_pdf(pdf, "secret")["pages"]
    assert [call[0] for call in calls] == ["upload", "url", "ocr", "delete"]


def test_convert_deletes_uploaded_file_when_ocr_fails(monkeypatch, tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")
    deleted = []
    monkeypatch.setattr(mistral, "upload_pdf", lambda *_: "file-1")
    monkeypatch.setattr(mistral, "signed_url", lambda *_: "https://signed")
    monkeypatch.setattr(mistral, "run_ocr", lambda *_: (_ for _ in ()).throw(mistral.MistralOCRError("boom")))
    monkeypatch.setattr(mistral, "delete_remote_file", lambda file_id, key: deleted.append(file_id))

    with pytest.raises(mistral.MistralOCRError, match="boom"):
        mistral.convert_pdf(pdf, "secret")
    assert deleted == ["file-1"]


def test_load_api_key_uses_keychain_without_printing_it(monkeypatch, tmp_path):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.setattr(mistral.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(mistral.sys, "platform", "darwin")
    monkeypatch.setattr(mistral.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="secret-key\n"))

    assert mistral.load_api_key() == "secret-key"


def test_missing_api_key_has_actionable_message(monkeypatch, tmp_path):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.setattr(mistral.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(mistral.sys, "platform", "linux")

    with pytest.raises(mistral.MistralOCRError, match="Settings > Mistral OCR"):
        mistral.load_api_key()


def test_intermittent_invalid_key_response_is_retried(monkeypatch):
    responses = []
    for status, body in ((401, '{"message":"Invalid API Key"}'), (200, '{"data":[]}')):
        response = mistral.requests.Response()
        response.status_code = status
        response._content = body.encode()
        response.headers["Content-Type"] = "application/json"
        responses.append(response)
    calls = []
    monkeypatch.setattr(
        mistral.requests,
        "request",
        lambda *args, **kwargs: calls.append((args, kwargs)) or responses.pop(0),
    )
    monkeypatch.setattr(mistral.time, "sleep", lambda _delay: None)

    response = mistral._request("GET", "https://api.mistral.ai/v1/files", "secret")

    assert response.status_code == 200
    assert len(calls) == 2
