#!/usr/bin/env python3
"""Convert one PDF with MinerU and preserve its tables and figures for Ragdrop."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid
import zipfile

import requests


MINERU_API = "https://mineru.net/api/v4"
MAX_ARCHIVE_BYTES = 500 * 1024 * 1024
VISUAL_TYPES = {"table", "image", "chart"}


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


def submit(pdf: Path, bearer: str) -> str:
    data_id = str(uuid.uuid4())
    headers = {"Authorization": f"Bearer {bearer}", "Content-Type": "application/json"}
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
        timeout=60,
    )
    response.raise_for_status()
    body = response.json()
    if body.get("code") != 0:
        raise RuntimeError(body.get("msg", "MinerU upload request failed"))
    batch_id = body["data"]["batch_id"]
    upload_url = body["data"]["file_urls"][0]
    with pdf.open("rb") as stream:
        upload = requests.put(
            upload_url,
            data=stream,
            headers={"Content-Length": str(pdf.stat().st_size)},
            timeout=300,
        )
    upload.raise_for_status()
    return batch_id


def wait_for_archive(batch_id: str, bearer: str, timeout: int = 900) -> str:
    headers = {"Authorization": f"Bearer {bearer}"}
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        response = requests.get(
            f"{MINERU_API}/extract-results/batch/{batch_id}",
            headers=headers,
            timeout=60,
        )
        response.raise_for_status()
        body = response.json()
        if body.get("code") != 0:
            raise RuntimeError(body.get("msg", "MinerU polling failed"))
        results = body.get("data", {}).get("extract_result", [])
        if results:
            result = results[0]
            if result.get("state") == "done":
                return result["full_zip_url"]
            if result.get("state") == "failed":
                raise RuntimeError(result.get("err_msg", "MinerU conversion failed"))
        time.sleep(5)
    raise TimeoutError("MinerU conversion timed out")


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


def extract_archive(payload: bytes, output_name: str) -> tuple[Path, Path, int]:
    output = Path("/tmp") / f"{output_name}.md"
    bundle = Path("/tmp") / f"{output_name}.ragdoc-artifacts"
    temporary = bundle.with_name(bundle.name + ".part")
    if len(payload) > MAX_ARCHIVE_BYTES:
        raise RuntimeError("MinerU archive is too large")
    if temporary.exists():
        import shutil
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    assets = temporary / "assets"
    assets.mkdir()
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
        saved = set()
        for filename in referenced:
            member = members.get(filename)
            if not member:
                continue
            data = archive.read(member)
            if data:
                (assets / filename).write_bytes(data)
                saved.add(filename)
    cleaned_markdown = clean_markdown(markdown)
    manifest = build_manifest(content, f"{output_name}.md", saved)
    manifest["page_spans"] = page_spans_from_content(content, cleaned_markdown)
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (temporary / "content_list.json").write_text(
        json.dumps(content, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    output.write_text(cleaned_markdown, encoding="utf-8")
    if bundle.exists():
        import shutil
        shutil.rmtree(bundle)
    os.replace(temporary, bundle)
    return output, bundle, len(manifest["artifacts"])


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
    bearer = token()
    batch_id = submit(pdf, bearer)
    archive_url = wait_for_archive(batch_id, bearer)
    response = requests.get(archive_url, timeout=300)
    response.raise_for_status()
    output, bundle, count = extract_archive(response.content, output_name)
    print(json.dumps({"markdown": str(output), "artifact_bundle": str(bundle), "artifacts": count}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
