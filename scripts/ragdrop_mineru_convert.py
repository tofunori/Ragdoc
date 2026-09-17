#!/usr/bin/env python3
"""Convert one PDF with MinerU and preserve its tables and figures for Ragdrop."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
import zipfile

import requests


MINERU_API = "https://mineru.net/api/v4"
MAX_ARCHIVE_BYTES = 500 * 1024 * 1024
VISUAL_TYPES = {"table", "image", "chart"}
UPLOAD_MAX_SECONDS = 1800
UPLOAD_STALL_SECONDS = 60
UPLOAD_MIN_BYTES_PER_SECOND = 1024
POLL_MAX_SECONDS = 900
MINERU_MAX_PAGES = 200
TRANSIENT_PARSE_RETRY_DELAYS = (5, 15)


class MinerUError(RuntimeError):
    """An actionable conversion failure safe to display in Ragdrop."""


def _stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=0.5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _run_cancellable_process(
    command: list[str],
    *,
    timeout: int,
    cancellation_message: str,
    timeout_message: str,
) -> subprocess.CompletedProcess:
    """Run one child process and forward SIGTERM before unwinding Python."""
    previous_handler = signal.getsignal(signal.SIGTERM)
    process: subprocess.Popen | None = None

    def cancel_child(_signum, _frame):
        if process is not None:
            _stop_process(process)
        raise MinerUError(cancellation_message)

    # Prevent SIGTERM from landing between Popen returning and the child being
    # registered in the handler closure. A pending signal is delivered as soon
    # as the previous mask is restored, with `process` already assigned.
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
    try:
        signal.signal(signal.SIGTERM, cancel_child)
        try:
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as error:
            _stop_process(process)
            raise MinerUError(timeout_message) from error
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
    assert process is not None
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def _run_upload(command: list[str]) -> subprocess.CompletedProcess:
    """Run curl with a deadline and cancellable child-process handling."""
    return _run_cancellable_process(
        command,
        timeout=UPLOAD_MAX_SECONDS + 60,
        cancellation_message="Téléversement MinerU annulé.",
        timeout_message=f"Le téléversement MinerU a dépassé {UPLOAD_MAX_SECONDS // 60} minutes.",
    )


def token() -> str:
    path = Path.home() / ".mineru_token"
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise RuntimeError(f"MinerU token is empty: {path}")
    return value


def needs_ocr(pdf: Path) -> bool:
    try:
        result = subprocess.run(
            ["pdftotext", "-l", "3", str(pdf), "-"],
            capture_output=True,
            check=False,
            timeout=30,
        )
        return len(result.stdout.strip()) <= 600
    except (OSError, subprocess.SubprocessError):
        return True


def extraction_language(pdf: Path) -> str:
    """Choose a documented MinerU OCR language pack for English/French papers."""
    override = os.getenv("MINERU_LANGUAGE")
    if override:
        return override
    try:
        result = subprocess.run(
            ["pdftotext", "-l", "3", str(pdf), "-"],
            capture_output=True,
            check=False,
            timeout=30,
        )
        sample = result.stdout.decode("utf-8", errors="ignore").casefold()
    except (OSError, subprocess.SubprocessError):
        return "en"
    padded = f" {sample} "
    french_markers = (" le ", " la ", " les ", " des ", " une ", " avec ", " étude ", " résumé ")
    return "latin" if sum(marker in padded for marker in french_markers) >= 3 else "en"


def pdf_page_count(pdf: Path) -> int:
    """Read a PDF page count without loading page contents into memory."""
    qpdf = shutil.which("qpdf")
    if qpdf:
        try:
            result = subprocess.run(
                [qpdf, "--show-npages", str(pdf)],
                capture_output=True,
                text=True,
                check=True,
                timeout=60,
            )
            pages = int(result.stdout.strip())
            if pages > 0:
                return pages
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    try:
        from pypdf import PdfReader
        pages = len(PdfReader(str(pdf)).pages)
    except (ImportError, OSError, ValueError) as error:
        raise MinerUError(
            "Impossible de compter les pages du PDF; installez qpdf ou pypdf."
        ) from error
    if pages < 1:
        raise MinerUError("Le PDF ne contient aucune page lisible.")
    return pages


def _split_with_pypdf(pdf: Path, parts: list[tuple[Path, int, int]]) -> None:
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError as error:
        raise MinerUError(
            "Ce PDF dépasse 200 pages; installez qpdf ou pypdf pour le découpage automatique."
        ) from error
    reader = PdfReader(str(pdf))
    for destination, page_offset, page_count in parts:
        writer = PdfWriter()
        for page in reader.pages[page_offset:page_offset + page_count]:
            writer.add_page(page)
        with destination.open("wb") as stream:
            writer.write(stream)


def split_pdf(pdf: Path, workspace: Path) -> list[tuple[Path, int, int]]:
    """Return PDF parts as (path, zero-based page offset, page count)."""
    page_count = pdf_page_count(pdf)
    if page_count <= MINERU_MAX_PAGES:
        return [(pdf, 0, page_count)]

    workspace.mkdir(parents=True, exist_ok=True)
    parts = []
    for index, page_offset in enumerate(range(0, page_count, MINERU_MAX_PAGES), start=1):
        count = min(MINERU_MAX_PAGES, page_count - page_offset)
        parts.append((workspace / f"part-{index:03d}.pdf", page_offset, count))

    qpdf = shutil.which("qpdf")
    if qpdf:
        for destination, page_offset, count in parts:
            first_page = page_offset + 1
            last_page = page_offset + count
            try:
                result = _run_cancellable_process(
                    [
                        qpdf, "--empty", "--pages", str(pdf),
                        f"{first_page}-{last_page}", "--", str(destination),
                    ],
                    timeout=300,
                    cancellation_message="Découpage du PDF annulé.",
                    timeout_message=(
                        f"Le découpage des pages {first_page}–{last_page} a dépassé 5 minutes."
                    ),
                )
                if result.returncode != 0:
                    detail = (result.stderr or result.stdout).strip().splitlines()
                    reason = detail[-1] if detail else f"qpdf code {result.returncode}"
                    raise MinerUError(
                        f"Le découpage du PDF a échoué pour les pages "
                        f"{first_page}–{last_page}: {reason}"
                    )
            except (OSError, subprocess.SubprocessError) as error:
                raise MinerUError(
                    f"Le découpage du PDF a échoué pour les pages {first_page}–{last_page}."
                ) from error
    else:
        _split_with_pypdf(pdf, parts)

    if any(not path.is_file() or path.stat().st_size == 0 for path, _, _ in parts):
        raise MinerUError("Le découpage automatique du PDF a produit une partie vide.")
    return parts


def upload_pdf(pdf: Path, upload_url: str) -> None:
    """Upload through curl so the transfer has a real wall-clock deadline.

    requests' timeout does not cap the time spent progressively writing a large
    request body. curl's max-time plus the subprocess timeout does.
    """
    curl = shutil.which("curl")
    if not curl:
        raise MinerUError("curl est introuvable; le téléversement MinerU ne peut pas démarrer.")
    command = [
        curl,
        "--fail-with-body",
        "--silent",
        "--show-error",
        "--connect-timeout", "20",
        "--max-time", str(UPLOAD_MAX_SECONDS),
        "--speed-time", str(UPLOAD_STALL_SECONDS),
        "--speed-limit", str(UPLOAD_MIN_BYTES_PER_SECOND),
        "--retry", "2",
        "--retry-delay", "2",
        "--retry-all-errors",
        "--header", f"Content-Length: {pdf.stat().st_size}",
        "--upload-file", str(pdf),
        upload_url,
    ]
    result = _run_upload(command)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        reason = detail[-1] if detail else f"curl code {result.returncode}"
        raise MinerUError(f"Le téléversement MinerU a échoué après plusieurs tentatives: {reason}")


def submit(pdf: Path, bearer: str) -> str:
    data_id = str(uuid.uuid4())
    headers = {"Authorization": f"Bearer {bearer}", "Content-Type": "application/json"}
    try:
        response = requests.post(
            f"{MINERU_API}/file-urls/batch",
            headers=headers,
            json={
                "enable_formula": True,
                "language": extraction_language(pdf),
                "model_version": "vlm",
                "enable_table": True,
                "files": [{"name": pdf.name, "is_ocr": needs_ocr(pdf), "data_id": data_id}],
            },
            timeout=(20, 60),
        )
        response.raise_for_status()
    except requests.RequestException as error:
        raise MinerUError(f"MinerU ne répond pas à la demande de téléversement: {error}") from error
    try:
        body = response.json()
    except (ValueError, TypeError) as error:
        raise MinerUError("MinerU a renvoyé une réponse de téléversement illisible.") from error
    if body.get("code") != 0:
        raise MinerUError(body.get("msg", "MinerU refuse la demande de téléversement"))
    batch_id = body["data"]["batch_id"]
    upload_url = body["data"]["file_urls"][0]
    upload_pdf(pdf, upload_url)
    return batch_id


def wait_for_archive(batch_id: str, bearer: str, timeout: int = POLL_MAX_SECONDS) -> str:
    headers = {"Authorization": f"Bearer {bearer}"}
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        try:
            response = requests.get(
                f"{MINERU_API}/extract-results/batch/{batch_id}",
                headers=headers,
                timeout=(20, 60),
            )
            response.raise_for_status()
        except (requests.ConnectionError, requests.Timeout):
            # A transient polling failure must not discard a conversion that is
            # still running remotely. The outer deadline remains authoritative.
            time.sleep(5)
            continue
        except requests.HTTPError as error:
            status = error.response.status_code if error.response is not None else None
            if status in {408, 425, 429} or (status is not None and status >= 500):
                time.sleep(5)
                continue
            if status in {401, 403}:
                raise MinerUError("MinerU refuse le jeton d’accès; vérifiez ~/.mineru_token.") from error
            raise MinerUError(f"MinerU refuse le suivi de la conversion (HTTP {status or 'inconnu'}).") from error
        except requests.RequestException as error:
            raise MinerUError(f"Le suivi MinerU a échoué: {error}") from error
        body = response.json()
        if body.get("code") != 0:
            raise MinerUError(body.get("msg", "MinerU ne peut pas lire l’état de la conversion"))
        results = body.get("data", {}).get("extract_result", [])
        if results:
            result = results[0]
            if result.get("state") == "done":
                return result["full_zip_url"]
            if result.get("state") == "failed":
                reason = result.get("err_msg") or "raison non fournie"
                raise MinerUError(f"MinerU a refusé la conversion: {reason}")
        time.sleep(5)
    raise MinerUError(f"La conversion MinerU a dépassé {timeout // 60} minutes.")


def convert_part(pdf: Path, bearer: str) -> bytes:
    """Convert one PDF part, retrying only MinerU's explicit transient failure."""
    attempts = len(TRANSIENT_PARSE_RETRY_DELAYS) + 1
    for attempt in range(attempts):
        try:
            batch_id = submit(pdf, bearer)
            archive_url = wait_for_archive(batch_id, bearer)
            return download_archive(archive_url)
        except MinerUError as error:
            message = str(error).casefold()
            transient = "parsing failed" in message and "try again later" in message
            if not transient or attempt == attempts - 1:
                raise
            delay = TRANSIENT_PARSE_RETRY_DELAYS[attempt]
            print(
                f"MinerU: analyse refusée temporairement; nouvelle tentative "
                f"{attempt + 2}/{attempts} dans {delay} s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def caption_text(item: dict) -> str:
    for key in ("table_caption", "image_caption", "chart_caption"):
        value = item.get(key)
        if isinstance(value, list):
            return " ".join(str(part).strip() for part in value if str(part).strip())
        if isinstance(value, str):
            return value.strip()
    return ""


def artifact_label(kind: str, caption: str, ordinal: int) -> str:
    match = re.match(r"\s*((?:table|figure|fig\.)\s+[A-Za-z]?\d+)", caption, re.IGNORECASE)
    if match:
        return match.group(1).replace("Fig.", "Figure").replace("fig.", "Figure")
    names = {"table": "Tableau", "image": "Figure", "chart": "Graphique"}
    return f"{names.get(kind, kind.title())} {ordinal}"


def build_manifest(content: list[dict], source: str, asset_names: set[str]) -> dict:
    artifacts = []
    counters: dict[str, int] = {}
    for item in content:
        kind = str(item.get("type", "")).lower()
        if kind not in VISUAL_TYPES:
            continue
        counters[kind] = counters.get(kind, 0) + 1
        caption = caption_text(item)
        original_image = Path(str(item.get("img_path", ""))).name
        stored_image = original_image if original_image in asset_names else None
        page_index = item.get("page_idx")
        identity = json.dumps(
            [source, kind, page_index, item.get("bbox"), caption, item.get("table_body", "")],
            ensure_ascii=False,
            sort_keys=True,
        )
        artifacts.append(
            {
                "artifact_id": hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24],
                "type": kind,
                "label": artifact_label(kind, caption, counters[kind]),
                "page": page_index + 1 if isinstance(page_index, int) else None,
                "bbox": item.get("bbox"),
                "caption": caption,
                "body": item.get("table_body", "") if kind == "table" else "",
                "image": f"assets/{stored_image}" if stored_image else None,
            }
        )
    return {
        "schema_version": 1,
        "source": source,
        "page_spans": [],
        "artifacts": artifacts,
    }


def page_spans_from_content(content: list[dict], markdown: str) -> list[dict]:
    """Map unique MinerU text to exact Python character offsets by PDF page."""
    bounds: dict[int, list[int]] = {}
    cursor = 0
    for item in content:
        text = item.get("text")
        page = item.get("page_idx")
        if not text or not isinstance(page, int) or markdown.count(text) != 1:
            continue
        start = markdown.find(text, cursor)
        if start < 0:
            continue
        end = start + len(text)
        current = bounds.setdefault(page + 1, [start, end])
        current[0] = min(current[0], start)
        current[1] = max(current[1], end)
        cursor = end
    return [
        {"page": page, "start": start, "end": end}
        for page, (start, end) in sorted(bounds.items())
        if start < end
    ]


def clean_markdown(markdown: str) -> str:
    markdown = re.sub(r"!\[([^]]*)\]\([^)]+\)", r"[Figure: \1]", markdown)
    return re.sub(r"\n{4,}", "\n\n\n", markdown).strip() + "\n"


def archive_components(payload: bytes) -> tuple[str, list[dict], dict[str, bytes]]:
    """Read the Markdown, content list and referenced visual assets from one result."""
    if len(payload) > MAX_ARCHIVE_BYTES:
        raise RuntimeError("MinerU archive is too large")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        markdown_names = [name for name in archive.namelist() if name.lower().endswith(".md")]
        if not markdown_names:
            raise RuntimeError("MinerU archive contains no Markdown")
        content_names = [
            name for name in archive.namelist()
            if name.endswith("_content_list.json") and "_v2" not in name
        ]
        content = json.loads(archive.read(content_names[0])) if content_names else []
        markdown = archive.read(max(markdown_names, key=lambda n: archive.getinfo(n).file_size)).decode("utf-8")
        referenced = {
            Path(str(item.get("img_path", ""))).name
            for item in content
            if str(item.get("type", "")).lower() in VISUAL_TYPES and item.get("img_path")
        }
        members = {Path(name).name: name for name in archive.namelist() if not name.endswith("/")}
        assets = {}
        for filename in referenced:
            member = members.get(filename)
            if not member:
                continue
            data = archive.read(member)
            if data:
                assets[filename] = data
    return clean_markdown(markdown), content, assets


def _remove_output_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


def _publish_outputs(
    staged_output: Path,
    staged_bundle: Path,
    output: Path,
    bundle: Path,
) -> None:
    """Publish Markdown and artifacts together, restoring old outputs on failure."""
    output_backup = output.with_name(output.name + ".previous")
    bundle_backup = bundle.with_name(bundle.name + ".previous")
    # A SIGTERM between os.replace() and its bookkeeping flag would make a
    # correct rollback unknowable. Defer it for this very short transaction;
    # any pending signal is delivered as soon as the outputs are stable.
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
    try:
        _remove_output_path(output_backup)
        _remove_output_path(bundle_backup)
        output_backed_up = False
        bundle_backed_up = False
        bundle_installed = False
        output_installed = False

        try:
            if output.exists():
                os.replace(output, output_backup)
                output_backed_up = True
            if bundle.exists():
                os.replace(bundle, bundle_backup)
                bundle_backed_up = True
            # Install the bundle first. The caller only learns the Markdown
            # path after both moves succeed, so partial output is not consumed.
            os.replace(staged_bundle, bundle)
            bundle_installed = True
            os.replace(staged_output, output)
            output_installed = True
        except BaseException:
            if output_installed:
                _remove_output_path(output)
            if bundle_installed:
                _remove_output_path(bundle)
            if output_backed_up and output_backup.exists():
                os.replace(output_backup, output)
            if bundle_backed_up and bundle_backup.exists():
                os.replace(bundle_backup, bundle)
            raise
        else:
            _remove_output_path(output_backup)
            _remove_output_path(bundle_backup)
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


def extract_archives(payloads: list[tuple[bytes, int]], output_name: str) -> tuple[Path, Path, int]:
    """Merge ordered MinerU archives while preserving original PDF page numbers."""
    if not payloads:
        raise RuntimeError("MinerU returned no archives")
    output = Path("/tmp") / f"{output_name}.md"
    bundle = Path("/tmp") / f"{output_name}.ragdoc-artifacts"
    temporary_bundle = bundle.with_name(bundle.name + ".part")
    temporary_output = output.with_name(output.name + ".part")
    _remove_output_path(temporary_bundle)
    _remove_output_path(temporary_output)

    try:
        temporary_bundle.mkdir(parents=True)
        assets_directory = temporary_bundle / "assets"
        assets_directory.mkdir()

        combined_markdown = ""
        combined_content = []
        combined_spans = []
        saved_assets = set()
        multiple_parts = len(payloads) > 1
        for part_number, (payload, page_offset) in enumerate(payloads, start=1):
            markdown, content, archive_assets = archive_components(payload)
            if combined_markdown:
                combined_markdown += "\n\n"
            character_offset = len(combined_markdown)

            adjusted_content = []
            renamed_assets = {}
            for filename, data in archive_assets.items():
                stored_name = f"part-{part_number:03d}-{filename}" if multiple_parts else filename
                renamed_assets[filename] = stored_name
                (assets_directory / stored_name).write_bytes(data)
                saved_assets.add(stored_name)
            for item in content:
                adjusted = dict(item)
                page_index = adjusted.get("page_idx")
                if isinstance(page_index, int):
                    adjusted["page_idx"] = page_index + page_offset
                image_name = Path(str(adjusted.get("img_path", ""))).name
                if image_name in renamed_assets:
                    adjusted["img_path"] = renamed_assets[image_name]
                adjusted_content.append(adjusted)

            for span in page_spans_from_content(adjusted_content, markdown):
                combined_spans.append({
                    "page": span["page"],
                    "start": span["start"] + character_offset,
                    "end": span["end"] + character_offset,
                })
            combined_content.extend(adjusted_content)
            combined_markdown += markdown

        manifest = build_manifest(combined_content, f"{output_name}.md", saved_assets)
        manifest["page_spans"] = combined_spans
        (temporary_bundle / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (temporary_bundle / "content_list.json").write_text(
            json.dumps(combined_content, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        temporary_output.write_text(combined_markdown, encoding="utf-8")
        _publish_outputs(temporary_output, temporary_bundle, output, bundle)
        return output, bundle, len(manifest["artifacts"])
    finally:
        _remove_output_path(temporary_output)
        _remove_output_path(temporary_bundle)


def extract_archive(payload: bytes, output_name: str) -> tuple[Path, Path, int]:
    return extract_archives([(payload, 0)], output_name)


def download_archive(archive_url: str) -> bytes:
    try:
        response = requests.get(archive_url, timeout=(20, 300))
        response.raise_for_status()
        return response.content
    except requests.RequestException as error:
        raise MinerUError(f"Le résultat MinerU ne peut pas être téléchargé: {error}") from error


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: ragdrop_mineru_convert.py PDF OUTPUT_NAME", file=sys.stderr)
        return 2
    pdf = Path(sys.argv[1]).expanduser().resolve()
    output_name = sys.argv[2]
    if not pdf.is_file() or pdf.suffix.lower() != ".pdf":
        raise FileNotFoundError(pdf)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", output_name):
        raise ValueError("Invalid output name")
    previous_handler = signal.getsignal(signal.SIGTERM)

    def cancel_conversion(_signum, _frame):
        raise MinerUError("Conversion MinerU annulée.")

    signal.signal(signal.SIGTERM, cancel_conversion)
    try:
        bearer = token()
        payloads = []
        with tempfile.TemporaryDirectory(prefix="ragdrop-mineru-") as temporary:
            parts = split_pdf(pdf, Path(temporary))
            for part_number, (part, page_offset, page_count) in enumerate(parts, start=1):
                if len(parts) > 1:
                    first_page = page_offset + 1
                    last_page = page_offset + page_count
                    print(
                        f"MinerU: partie {part_number}/{len(parts)}, pages {first_page}–{last_page}",
                        file=sys.stderr,
                        flush=True,
                    )
                payloads.append((convert_part(part, bearer), page_offset))
        output, bundle, count = extract_archives(payloads, output_name)
        print(json.dumps({
            "markdown": str(output),
            "artifact_bundle": str(bundle),
            "artifacts": count,
        }))
        return 0
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (MinerUError, FileNotFoundError, ValueError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
    except Exception as error:
        print(f"Erreur MinerU inattendue ({type(error).__name__}): {error}", file=sys.stderr)
        if os.getenv("RAGDROP_DEBUG"):
            traceback.print_exc()
        raise SystemExit(1)
