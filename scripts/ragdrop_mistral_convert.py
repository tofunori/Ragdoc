#!/usr/bin/env python3
"""Convert one PDF with Mistral OCR for Ragdrop and preserve page provenance."""

from __future__ import annotations

import base64
import copy
import json
import mimetypes
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid

import requests


API_ROOT = "https://api.mistral.ai/v1"
MODEL = "mistral-ocr-latest"
KEYCHAIN_SERVICE = "com.tofunori.ragdrop.mistral"
KEYCHAIN_ACCOUNT = "api-key"
MAX_PDF_BYTES = 512 * 1024 * 1024
TRANSIENT_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}
RETRY_DELAYS = (2, 5, 12)


class MistralOCRError(RuntimeError):
    """An actionable conversion failure safe to display in Ragdrop."""


def _cancel(_signum, _frame) -> None:
    raise MistralOCRError("Mistral conversion canceled. You can retry this PDF.")


def load_api_key() -> str:
    key = os.environ.get("MISTRAL_API_KEY", "").strip()
    if key:
        return key
    key_file = Path.home() / ".mistral_api_key"
    if key_file.is_file():
        key = key_file.read_text(encoding="utf-8").strip()
        if key:
            return key
    if sys.platform == "darwin" and Path("/usr/bin/security").exists():
        result = subprocess.run(
            [
                "/usr/bin/security",
                "find-generic-password",
                "-s",
                KEYCHAIN_SERVICE,
                "-a",
                KEYCHAIN_ACCOUNT,
                "-w",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        key = result.stdout.strip()
        if result.returncode == 0 and key:
            return key
    raise MistralOCRError(
        "Mistral key missing. Add it in Ragdrop Settings > Mistral OCR."
    )


def _useful_api_error(response: requests.Response) -> str:
    try:
        payload = response.json()
        detail = payload.get("message") or payload.get("detail") or payload.get("error")
        if isinstance(detail, dict):
            detail = detail.get("message") or detail.get("detail")
        if detail:
            return str(detail)
    except (ValueError, AttributeError):
        pass
    return response.text.strip()[:500] or f"HTTP {response.status_code}"


def _request(method: str, url: str, api_key: str, **kwargs) -> requests.Response:
    headers = dict(kwargs.pop("headers", {}))
    headers["Authorization"] = f"Bearer {api_key}"
    last_error: Exception | None = None
    for attempt, delay in enumerate((*RETRY_DELAYS, None)):
        try:
            for file_value in (kwargs.get("files") or {}).values():
                if isinstance(file_value, tuple) and len(file_value) > 1:
                    file_value[1].seek(0)
            response = requests.request(method, url, headers=headers, **kwargs)
            intermittent_auth_failure = (
                response.status_code == 401
                and "invalid api key" in _useful_api_error(response).lower()
            )
            if response.status_code not in TRANSIENT_STATUS_CODES and not intermittent_auth_failure:
                if not response.ok:
                    raise MistralOCRError(
                        f"Mistral rejected the conversion ({response.status_code}) : "
                        f"{_useful_api_error(response)}"
                    )
                return response
            last_error = MistralOCRError(
                f"Mistral est temporairement indisponible ({response.status_code}) : "
                f"{_useful_api_error(response)}"
            )
        except requests.RequestException as error:
            last_error = error
        if delay is not None:
            time.sleep(delay)
    raise MistralOCRError(f"The Mistral request failed after several attempts : {last_error}")


def upload_pdf(pdf: Path, api_key: str) -> str:
    with pdf.open("rb") as handle:
        response = _request(
            "POST",
            f"{API_ROOT}/files",
            api_key,
            files={"file": (pdf.name, handle, "application/pdf")},
            data={"purpose": "ocr", "visibility": "user"},
            timeout=(30, 1800),
        )
    file_id = str(response.json().get("id") or "").strip()
    if not file_id:
        raise MistralOCRError("Mistral returned no file identifier.")
    return file_id


def signed_url(file_id: str, api_key: str) -> str:
    response = _request(
        "GET", f"{API_ROOT}/files/{file_id}/url", api_key, timeout=(20, 60)
    )
    url = str(response.json().get("url") or "").strip()
    if not url:
        raise MistralOCRError("Mistral returned no PDF reading URL.")
    return url


def run_ocr(document_url: str, api_key: str) -> dict:
    payload = {
        "model": MODEL,
        "document": {"type": "document_url", "document_url": document_url},
        "table_format": "html",
        "extract_header": True,
        "extract_footer": True,
        "include_image_base64": True,
        "include_blocks": True,
        "confidence_scores_granularity": "block",
    }
    response = _request(
        "POST",
        f"{API_ROOT}/ocr",
        api_key,
        json=payload,
        timeout=(30, 1800),
    )
    result = response.json()
    if not isinstance(result.get("pages"), list) or not result["pages"]:
        raise MistralOCRError("Mistral returned no OCR pages.")
    return result


def delete_remote_file(file_id: str, api_key: str) -> None:
    try:
        _request("DELETE", f"{API_ROOT}/files/{file_id}", api_key, timeout=(20, 60))
    except MistralOCRError:
        # Cleanup must never hide a successful OCR result or its real failure.
        pass


def convert_pdf(pdf: Path, api_key: str) -> dict:
    file_id = upload_pdf(pdf, api_key)
    try:
        return run_ocr(signed_url(file_id, api_key), api_key)
    finally:
        delete_remote_file(file_id, api_key)


def _safe_filename(value: str, fallback: str) -> str:
    name = Path(value).name.strip() or fallback
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip(".-") or fallback
    return name[:180]


def _image_bytes(value: str) -> bytes:
    encoded = value.split(",", 1)[-1]
    try:
        return base64.b64decode(encoded, validate=True)
    except ValueError as error:
        raise MistralOCRError("A Mistral figure contains unreadable data.") from error


def _bbox(item: dict) -> list[float] | None:
    coordinates = item.get("coordinates") or item.get("bbox")
    if isinstance(coordinates, list) and len(coordinates) == 4:
        return coordinates
    if isinstance(coordinates, dict):
        keys = ("x1", "y1", "x2", "y2")
        if all(key in coordinates for key in keys):
            return [coordinates[key] for key in keys]
    return None


def _nearest_caption(blocks: list[dict], target: dict) -> str:
    target_box = _bbox(target)
    captions = [block for block in blocks if str(block.get("type", "")).lower() == "caption"]
    if not captions:
        return ""
    if target_box is None:
        return str(captions[0].get("content") or "").strip()
    target_y = target_box[3]
    caption = min(
        captions,
        key=lambda block: abs(((_bbox(block) or [0, 0, 0, target_y])[1]) - target_y),
    )
    return str(caption.get("content") or "").strip()


def _inline_tables(markdown: str, tables: list[dict]) -> tuple[str, dict[str, str]]:
    table_contents: dict[str, str] = {}
    for index, table in enumerate(tables, 1):
        table_id = str(table.get("id") or table.get("table_id") or f"table-{index}.html")
        content = str(table.get("content") or table.get("html") or table.get("markdown") or "").strip()
        if not content:
            continue
        table_contents[table_id] = content
        markdown = markdown.replace(f"[{table_id}]({table_id})", content)
        markdown = markdown.replace(f"![]({table_id})", content)
    return markdown, table_contents


def _replace_image_reference(markdown: str, image_id: str, caption: str) -> str:
    label = caption or Path(image_id).stem.replace("_", " ").replace("-", " ")
    replacement = f"[Figure: {label.strip()}]"
    pattern = re.compile(r"!\[([^\]]*)\]\(" + re.escape(image_id) + r"\)")
    return pattern.sub(lambda _match: replacement, markdown)


def _remove_path(path: Path) -> None:
    if path.is_dir():
        import shutil

        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


def _publish_outputs(staged_output: Path, staged_bundle: Path, output: Path, bundle: Path) -> None:
    output_backup = output.with_name(output.name + ".previous")
    bundle_backup = bundle.with_name(bundle.name + ".previous")
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
    try:
        _remove_path(output_backup)
        _remove_path(bundle_backup)
        output_backed_up = bundle_backed_up = False
        output_installed = bundle_installed = False
        try:
            if output.exists():
                os.replace(output, output_backup)
                output_backed_up = True
            if bundle.exists():
                os.replace(bundle, bundle_backup)
                bundle_backed_up = True
            os.replace(staged_bundle, bundle)
            bundle_installed = True
            os.replace(staged_output, output)
            output_installed = True
        except BaseException:
            if output_installed:
                _remove_path(output)
            if bundle_installed:
                _remove_path(bundle)
            if output_backed_up and output_backup.exists():
                os.replace(output_backup, output)
            if bundle_backed_up and bundle_backup.exists():
                os.replace(bundle_backup, bundle)
            raise
        else:
            _remove_path(output_backup)
            _remove_path(bundle_backup)
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


def materialize(
    response: dict, output_name: str, temporary_root: Path = Path("/tmp")
) -> tuple[Path, Path, int]:
    output = temporary_root / f"{output_name}.md"
    bundle = temporary_root / f"{output_name}.ragdoc-artifacts"
    temporary_output = output.with_name(output.name + ".part")
    temporary_bundle = bundle.with_name(bundle.name + ".part")
    _remove_path(temporary_output)
    _remove_path(temporary_bundle)
    temporary_bundle.mkdir(parents=True)
    assets = temporary_bundle / "assets"
    assets.mkdir()

    pages = sorted(response.get("pages", []), key=lambda page: int(page.get("index", 0)))
    combined = ""
    page_spans: list[dict] = []
    artifacts: list[dict] = []
    normalized_blocks: list[dict] = []
    try:
        for position, page in enumerate(pages, 1):
            page_number = int(page.get("index", position - 1)) + 1
            markdown = str(page.get("markdown") or "").strip()
            blocks = [block for block in (page.get("blocks") or []) if isinstance(block, dict)]
            markdown, table_contents = _inline_tables(markdown, page.get("tables") or [])

            image_paths: dict[str, str] = {}
            for image_index, image in enumerate(page.get("images") or [], 1):
                image_id = str(image.get("id") or f"image-{image_index}.png")
                encoded = image.get("image_base64") or image.get("base64")
                image_block = next(
                    (block for block in blocks if str(block.get("id") or block.get("image_id") or "") == image_id),
                    {},
                )
                caption = _nearest_caption(blocks, image_block) if image_block else ""
                markdown = _replace_image_reference(markdown, image_id, caption)
                stored_path = None
                if encoded:
                    suffix = Path(image_id).suffix or mimetypes.guess_extension("image/png") or ".png"
                    safe_name = _safe_filename(
                        f"page-{page_number:03d}-{Path(image_id).stem}{suffix}",
                        f"page-{page_number:03d}-image-{image_index}.png",
                    )
                    (assets / safe_name).write_bytes(_image_bytes(str(encoded)))
                    stored_path = f"assets/{safe_name}"
                    image_paths[image_id] = stored_path
                artifacts.append(
                    {
                        "artifact_id": f"{output_name}:p{page_number}:image:{image_index}",
                        "type": "image",
                        "label": f"Figure {image_index}",
                        "page": page_number,
                        "bbox": _bbox(image_block) or _bbox(image),
                        "caption": caption,
                        "body": str(image_block.get("content") or caption),
                        "image": stored_path,
                    }
                )

            for table_index, table in enumerate(page.get("tables") or [], 1):
                table_id = str(table.get("id") or table.get("table_id") or f"table-{table_index}.html")
                table_block = next(
                    (block for block in blocks if str(block.get("id") or block.get("table_id") or "") == table_id),
                    {},
                )
                content = table_contents.get(table_id, str(table.get("content") or ""))
                artifacts.append(
                    {
                        "artifact_id": f"{output_name}:p{page_number}:table:{table_index}",
                        "type": "table",
                        "label": f"Table {table_index}",
                        "page": page_number,
                        "bbox": _bbox(table_block) or _bbox(table),
                        "caption": _nearest_caption(blocks, table_block) if table_block else "",
                        "body": content,
                        "image": None,
                    }
                )

            page_text = f"<!-- page {page_number} -->\n\n{markdown}\n"
            if combined:
                combined += "\n"
            start = len(combined)
            combined += page_text
            page_spans.append({"page": page_number, "start": start, "end": len(combined)})
            for block in blocks:
                normalized = copy.deepcopy(block)
                normalized["page_idx"] = page_number - 1
                normalized_blocks.append(normalized)

        manifest = {
            "schema_version": 1,
            "source": f"{output_name}.md",
            "parser": "mistral-ocr",
            "model": str(response.get("model") or MODEL),
            "page_spans": page_spans,
            "artifacts": artifacts,
        }
        scrubbed = copy.deepcopy(response)
        for page in scrubbed.get("pages", []):
            for image in page.get("images") or []:
                if "image_base64" in image:
                    image["image_base64"] = "<saved separately>"
                if "base64" in image:
                    image["base64"] = "<saved separately>"
        (temporary_bundle / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (temporary_bundle / "content_list.json").write_text(
            json.dumps(normalized_blocks, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (temporary_bundle / "mistral-response.json").write_text(
            json.dumps(scrubbed, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        temporary_output.write_text(combined, encoding="utf-8")
        _publish_outputs(temporary_output, temporary_bundle, output, bundle)
        return output, bundle, len(artifacts)
    finally:
        _remove_path(temporary_output)
        _remove_path(temporary_bundle)


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: ragdrop_mistral_convert.py PDF OUTPUT_NAME", file=sys.stderr)
        return 2
    pdf = Path(sys.argv[1]).expanduser().resolve()
    output_name = sys.argv[2]
    if not pdf.is_file() or pdf.suffix.lower() != ".pdf":
        print("The selected file is not a readable PDF.", file=sys.stderr)
        return 2
    if pdf.stat().st_size > MAX_PDF_BYTES:
        print("Mistral limits each uploaded file to 512 MB.", file=sys.stderr)
        return 2
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", output_name):
        print("The output name contains invalid characters.", file=sys.stderr)
        return 2

    previous_handler = signal.signal(signal.SIGTERM, _cancel)
    try:
        response = convert_pdf(pdf, load_api_key())
        output, bundle, artifact_count = materialize(response, output_name)
    except (MistralOCRError, requests.RequestException, ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
    print(
        json.dumps(
            {"markdown": str(output), "artifacts": str(bundle), "artifact_count": artifact_count},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
