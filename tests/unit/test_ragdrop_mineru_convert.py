import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
import zipfile

import scripts.ragdrop_mineru_convert as mineru
from scripts.ragdrop_mineru_convert import build_manifest, extract_archive, extract_archives


def mineru_archive(markdown: str, content: list[dict], assets: dict[str, bytes] | None = None) -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("paper.md", markdown)
        archive.writestr("paper_content_list.json", json.dumps(content))
        for name, data in (assets or {}).items():
            archive.writestr(name, data)
    return payload.getvalue()


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


def test_multiple_archives_merge_pages_markdown_and_colliding_asset_names():
    output_name = f"ragdrop_test_{uuid.uuid4().hex}"
    first = mineru_archive(
        "# First\n\nAlpha passage.\n\n![Figure](images/figure.jpg)\n",
        [{
            "type": "image", "text": "Alpha passage.", "img_path": "images/figure.jpg",
            "image_caption": ["Figure 1. First"], "page_idx": 0,
        }],
        {"images/figure.jpg": b"first-image"},
    )
    second = mineru_archive(
        "# Second\n\nBeta passage.\n\n![Figure](images/figure.jpg)\n",
        [{
            "type": "image", "text": "Beta passage.", "img_path": "images/figure.jpg",
            "image_caption": ["Figure 2. Second"], "page_idx": 0,
        }],
        {"images/figure.jpg": b"second-image"},
    )

    markdown, bundle, count = extract_archives([(first, 0), (second, 200)], output_name)
    try:
        manifest = json.loads((bundle / "manifest.json").read_text())
        content = json.loads((bundle / "content_list.json").read_text())
        assert count == 2
        assert [artifact["page"] for artifact in manifest["artifacts"]] == [1, 201]
        assert [span["page"] for span in manifest["page_spans"]] == [1, 201]
        assert content[1]["page_idx"] == 200
        assert (bundle / "assets/part-001-figure.jpg").read_bytes() == b"first-image"
        assert (bundle / "assets/part-002-figure.jpg").read_bytes() == b"second-image"
        combined = markdown.read_text()
        assert combined.index("Alpha passage.") < combined.index("Beta passage.")
    finally:
        markdown.unlink(missing_ok=True)
        shutil.rmtree(bundle, ignore_errors=True)


def test_invalid_later_archive_cleans_staged_outputs():
    output_name = f"ragdrop_test_{uuid.uuid4().hex}"
    valid = mineru_archive("# First\n", [])
    output = Path("/tmp") / f"{output_name}.md"
    bundle = Path("/tmp") / f"{output_name}.ragdoc-artifacts"

    try:
        try:
            extract_archives([(valid, 0), (b"not-a-zip", 200)], output_name)
        except zipfile.BadZipFile:
            pass
        else:
            raise AssertionError("An invalid later archive must fail the merge")

        assert not output.exists()
        assert not bundle.exists()
        assert not output.with_name(output.name + ".part").exists()
        assert not bundle.with_name(bundle.name + ".part").exists()
    finally:
        output.unlink(missing_ok=True)
        shutil.rmtree(bundle, ignore_errors=True)


def test_publish_failure_restores_previous_markdown_and_bundle(monkeypatch):
    output_name = f"ragdrop_test_{uuid.uuid4().hex}"
    payload = mineru_archive("# Replacement\n", [])
    output = Path("/tmp") / f"{output_name}.md"
    bundle = Path("/tmp") / f"{output_name}.ragdoc-artifacts"
    output.write_text("# Previous\n", encoding="utf-8")
    bundle.mkdir()
    (bundle / "sentinel.txt").write_text("previous", encoding="utf-8")
    real_replace = mineru.os.replace
    failed = False

    def replace(source, destination):
        nonlocal failed
        source_path = Path(source)
        destination_path = Path(destination)
        if not failed and source_path.name.endswith(".md.part") and destination_path == output:
            failed = True
            raise OSError("simulated publish failure")
        return real_replace(source, destination)

    monkeypatch.setattr(mineru.os, "replace", replace)
    try:
        try:
            extract_archives([(payload, 0)], output_name)
        except OSError as error:
            assert "simulated" in str(error)
        else:
            raise AssertionError("The simulated publication failure must be visible")

        assert output.read_text(encoding="utf-8") == "# Previous\n"
        assert (bundle / "sentinel.txt").read_text(encoding="utf-8") == "previous"
        assert not output.with_name(output.name + ".part").exists()
        assert not bundle.with_name(bundle.name + ".part").exists()
        assert not output.with_name(output.name + ".previous").exists()
        assert not bundle.with_name(bundle.name + ".previous").exists()
    finally:
        output.unlink(missing_ok=True)
        shutil.rmtree(bundle, ignore_errors=True)


def test_large_pdf_is_split_into_bounded_qpdf_ranges(monkeypatch, tmp_path):
    pdf = tmp_path / "large.pdf"
    pdf.write_bytes(b"pdf")
    commands = []

    def run(command, **_kwargs):
        commands.append(command)
        Path(command[-1]).write_bytes(b"part")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(mineru, "pdf_page_count", lambda _pdf: 401)
    monkeypatch.setattr(mineru.shutil, "which", lambda name: "/opt/homebrew/bin/qpdf" if name == "qpdf" else None)
    monkeypatch.setattr(mineru, "_run_cancellable_process", run)

    parts = mineru.split_pdf(pdf, tmp_path / "parts")

    assert [(offset, count) for _path, offset, count in parts] == [(0, 200), (200, 200), (400, 1)]
    assert [command[-3] for command in commands] == ["1-200", "201-400", "401-401"]


def test_pdf_at_api_limit_is_not_split(monkeypatch, tmp_path):
    pdf = tmp_path / "limit.pdf"
    pdf.write_bytes(b"pdf")
    monkeypatch.setattr(mineru, "pdf_page_count", lambda _pdf: 200)

    assert mineru.split_pdf(pdf, tmp_path / "parts") == [(pdf, 0, 200)]


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
    monkeypatch.setattr(mineru, "upload_pdf", lambda *args: None)

    assert mineru.submit(pdf, "token") == "batch"
    assert request["url"].endswith("/api/v4/file-urls/batch")
    assert request["json"]["model_version"] == "vlm"
    assert request["json"]["language"] == "en"
    assert "layout_model" not in request["json"]


def test_upload_has_retries_and_wall_clock_deadline(monkeypatch, tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"pdf")
    observed = {}

    def run(command):
        observed.update(command=command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(mineru.shutil, "which", lambda _: "/usr/bin/curl")
    monkeypatch.setattr(mineru, "_run_upload", run)

    mineru.upload_pdf(pdf, "https://upload.example/signed")

    assert observed["command"][0] == "/usr/bin/curl"
    assert observed["command"][observed["command"].index("--retry") + 1] == "2"
    assert observed["command"][observed["command"].index("--max-time") + 1] == "1800"
    assert observed["command"][observed["command"].index("--speed-time") + 1] == "60"
    assert observed["command"][observed["command"].index("--speed-limit") + 1] == "1024"


def test_stop_process_escalates_when_child_ignores_termination():
    class Process:
        killed = False
        terminated = False
        waits = 0

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            self.waits += 1
            if timeout is not None:
                raise subprocess.TimeoutExpired("curl", timeout)
            return -9

        def kill(self):
            self.killed = True

    process = Process()
    mineru._stop_process(process)

    assert process.terminated
    assert process.killed
    assert process.waits == 2


def test_sigterm_stops_active_upload_child(tmp_path):
    child_pid_file = tmp_path / "child.pid"
    child_code = (
        "from pathlib import Path; import os,time; "
        f"Path({str(child_pid_file)!r}).write_text(str(os.getpid())); time.sleep(60)"
    )
    wrapper_code = (
        "from scripts.ragdrop_mineru_convert import MinerUError,_run_upload; import sys; "
        f"command=[{sys.executable!r},'-c',{child_code!r}]; "
        "\ntry: _run_upload(command)\nexcept MinerUError: raise SystemExit(7)"
    )
    wrapper = subprocess.Popen([sys.executable, "-c", wrapper_code])
    try:
        for _ in range(100):
            if child_pid_file.exists():
                break
            time.sleep(0.02)
        assert child_pid_file.exists()
        child_pid = int(child_pid_file.read_text())
        wrapper.terminate()
        assert wrapper.wait(timeout=5) == 7
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            pass
        else:
            raise AssertionError("Upload child survived cancellation")
    finally:
        if wrapper.poll() is None:
            wrapper.kill()
            wrapper.wait()


def test_polling_reports_permanent_authentication_error(monkeypatch):
    response = type("Response", (), {"status_code": 401})()
    error = mineru.requests.HTTPError(response=response)

    class FailedResponse:
        def raise_for_status(self):
            raise error

    monkeypatch.setattr(mineru.requests, "get", lambda *args, **kwargs: FailedResponse())

    try:
        mineru.wait_for_archive("batch", "token", timeout=1)
    except mineru.MinerUError as caught:
        assert "token" in str(caught)
    else:
        raise AssertionError("Permanent authentication errors must fail immediately")


def test_transient_parsing_failure_is_retried(monkeypatch, tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"pdf")
    batches = iter(["first", "second"])
    waits = []

    monkeypatch.setattr(mineru, "submit", lambda *_args: next(batches))

    def wait(batch_id, _bearer):
        if batch_id == "first":
            raise mineru.MinerUError(
                "MinerU rejected the conversion: parsing failed, please try again later"
            )
        return "https://example/archive.zip"

    monkeypatch.setattr(mineru, "wait_for_archive", wait)
    monkeypatch.setattr(mineru, "download_archive", lambda _url: b"archive")
    monkeypatch.setattr(mineru.time, "sleep", waits.append)

    assert mineru.convert_part(pdf, "token") == b"archive"
    assert waits == [5]


def test_permanent_conversion_failure_is_not_retried(monkeypatch, tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"pdf")
    submissions = []

    def submit(*_args):
        submissions.append(True)
        return "batch"

    monkeypatch.setattr(mineru, "submit", submit)
    monkeypatch.setattr(
        mineru,
        "wait_for_archive",
        lambda *_args: (_ for _ in ()).throw(mineru.MinerUError("invalid document")),
    )

    try:
        mineru.convert_part(pdf, "token")
    except mineru.MinerUError as error:
        assert "invalid document" in str(error)
    else:
        raise AssertionError("Permanent failures must remain visible")
    assert len(submissions) == 1
