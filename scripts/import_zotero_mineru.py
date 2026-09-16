#!/usr/bin/env python3
"""Resumable Zotero PDF inventory and MinerU batch importer for Ragdoc."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "articles_markdown"
STATE_DIR = ROOT / "build" / "zotero_mineru"
ZOTERO_API = "http://127.0.0.1:23119/api/users/0"
MINERU_API = "https://mineru.net/api/v4"
BATCH_SIZE = 50
MAX_ARCHIVE_BYTES = 500 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024


def atomic_write_text(path: Path, content: str) -> None:
    """Replace a text artifact only after its complete contents reach disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def safe_key(value: str) -> str:
    if not re.fullmatch(r"[A-Z0-9_]+", value or ""):
        raise ValueError(f"Unsafe Zotero attachment key: {value!r}")
    return value


def api_json(url: str) -> object:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def zotero_items(item_type: str) -> list[dict]:
    items = []
    for start in range(0, 100_000, 100):
        page = api_json(f"{ZOTERO_API}/items?itemType={item_type}&limit=100&start={start}")
        if not page:
            break
        items.extend(page)
    return items


def local_pdf_path(item: dict) -> Path | None:
    enclosure = item.get("links", {}).get("enclosure", {})
    href = enclosure.get("href", "")
    if enclosure.get("type") != "application/pdf" or not href.startswith("file:"):
        return None
    return Path(unquote(urlparse(href).path))


def parent_map(keys: set[str]) -> dict[str, dict]:
    parents = {}
    ordered = sorted(keys)
    for start in range(0, len(ordered), 50):
        joined = ",".join(ordered[start:start + 50])
        for item in api_json(f"{ZOTERO_API}/items?itemKey={joined}&limit=100"):
            parents[item["key"]] = item
    return parents


def pdf_pages(path: Path) -> int | None:
    try:
        result = subprocess.run(["pdfinfo", str(path)], check=True, capture_output=True,
                                text=True, timeout=30)
        match = re.search(r"(?m)^Pages:\s+(\d+)", result.stdout)
        return int(match.group(1)) if match else None
    except (OSError, subprocess.SubprocessError):
        return None


def has_text_layer(path: Path) -> bool:
    try:
        result = subprocess.run(["pdftotext", "-l", "3", str(path), "-"],
                                check=True, capture_output=True, timeout=30)
        return len(result.stdout.strip()) > 600
    except (OSError, subprocess.SubprocessError):
        return False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def creators(data: dict) -> list[str]:
    names = []
    for creator in data.get("creators", []):
        if creator.get("creatorType") != "author":
            continue
        name = creator.get("name") or " ".join(
            part for part in (creator.get("firstName"), creator.get("lastName")) if part)
        if name:
            names.append(name)
    return names


def year(data: dict) -> int | None:
    match = re.search(r"\b(1[5-9]\d{2}|20\d{2}|2100)\b", data.get("date", ""))
    return int(match.group(1)) if match else None


def inventory() -> list[dict]:
    attachments = []
    for item in zotero_items("attachment"):
        path = local_pdf_path(item)
        if path is None:
            continue
        data = item["data"]
        attachments.append({"attachment_key": item["key"], "parent_key": data.get("parentItem"),
                            "path": path, "md5": data.get("md5"),
                            "date_added": data.get("dateAdded", ""),
                            "size": item["links"]["enclosure"].get("length")})
    parents = parent_map({row["parent_key"] for row in attachments if row["parent_key"]})
    records = []
    for row in attachments:
        parent = parents.get(row["parent_key"], {})
        data = parent.get("data", {})
        path = row.pop("path")
        records.append({**row, "path": str(path), "exists": path.is_file(),
                        "title": data.get("title") or path.stem,
                        "authors": creators(data), "year": year(data),
                        "doi": data.get("DOI") or None, "item_type": data.get("itemType"),
                        "pages": pdf_pages(path) if path.is_file() else None,
                        "text_layer": has_text_layer(path) if path.is_file() else False})
    return records


def deduplicate(records: list[dict]) -> tuple[list[dict], list[dict]]:
    chosen, duplicates = {}, []
    for record in sorted(records, key=lambda row: (bool(row["parent_key"]), row["date_added"]),
                         reverse=True):
        key = record["md5"] or f"path:{record['path']}"
        if key in chosen:
            duplicates.append({"attachment_key": record["attachment_key"],
                               "duplicate_of": chosen[key]["attachment_key"]})
        else:
            chosen[key] = record
    return list(chosen.values()), duplicates


def write_inventory(records: list[dict], duplicates: list[dict]) -> Path:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    destination = STATE_DIR / "inventory.json"
    atomic_write_text(destination, json.dumps({"created_at": datetime.now(timezone.utc).isoformat(),
                                               "records": records, "duplicates": duplicates},
                                              ensure_ascii=False, indent=2))
    return destination


def token() -> str:
    path = Path.home() / ".mineru_token"
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise RuntimeError("MinerU token is empty")
    return value


def journal(event: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with (STATE_DIR / "journal.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(), **event},
                                ensure_ascii=False) + "\n")


def submit_batch(records: list[dict], bearer: str, model: str) -> tuple[str, list[str]]:
    headers = {"Authorization": f"Bearer {bearer}", "Content-Type": "application/json"}
    payload = {
        "model_version": model,
        "language": os.getenv("MINERU_LANGUAGE", "en"),
        "enable_formula": True,
        "enable_table": True,
        "files": [
        {"name": Path(row["path"]).name, "data_id": row["attachment_key"],
         "is_ocr": not row["text_layer"]} for row in records],
    }
    for attempt in range(6):
        response = requests.post(f"{MINERU_API}/file-urls/batch", headers=headers,
                                 json=payload, timeout=60)
        if response.status_code != 429:
            break
        delay = min(30, int(response.headers.get("Retry-After", 5 * (attempt + 1))))
        print(f"MinerU rate limit; retrying batch submission in {delay}s", flush=True)
        time.sleep(delay)
    try:
        response.raise_for_status()
    except requests.HTTPError as error:
        message = response.text[:500].replace("\n", " ")
        raise RuntimeError(f"MinerU HTTP {response.status_code}: {message}") from error
    body = response.json()
    if body.get("code") != 0:
        raise RuntimeError(f"MinerU submission failed: {body.get('msg', 'unknown error')}")
    return body["data"]["batch_id"], body["data"]["file_urls"]


def upload(records: list[dict], urls: list[str]) -> None:
    if len(records) != len(urls):
        raise RuntimeError("MinerU returned an unexpected number of upload URLs")
    for row, url in zip(records, urls):
        path = Path(row["path"])
        with path.open("rb") as stream:
            response = requests.put(url, data=stream,
                                    headers={"Content-Length": str(path.stat().st_size)}, timeout=600)
        response.raise_for_status()
        journal({"attachment_key": row["attachment_key"], "state": "uploaded"})


def poll(batch_id: str, bearer: str, wanted: set[str], timeout: int = 3600) -> dict[str, dict]:
    headers = {"Authorization": f"Bearer {bearer}"}
    done = {}
    started = time.monotonic()
    while wanted - done.keys():
        if time.monotonic() - started > timeout:
            raise TimeoutError(f"MinerU batch {batch_id} timed out")
        response = requests.get(f"{MINERU_API}/extract-results/batch/{batch_id}",
                                headers=headers, timeout=60)
        response.raise_for_status()
        body = response.json()
        if body.get("code") != 0:
            raise RuntimeError(f"MinerU polling failed: {body.get('msg', 'unknown error')}")
        for result in body.get("data", {}).get("extract_result", []):
            key = result.get("data_id")
            if not key:
                continue
            if result.get("state") == "failed" and key not in done:
                journal({"attachment_key": key, "state": "failed", "error": result.get("err_msg")})
                done[key] = result
            elif result.get("state") == "done" and key not in done:
                done[key] = result
        if wanted - done.keys():
            time.sleep(5)
    return done


def page_spans_from_content(items: list[dict], markdown: str) -> list[dict]:
    bounds = {}
    cursor = 0
    for item in items:
        text = item.get("text")
        page = item.get("page_idx")
        if not text or type(page) is not int or markdown.count(text) != 1:
            continue
        start = markdown.find(text, cursor)
        if start < 0:
            continue
        end = start + len(text)
        current = bounds.setdefault(page + 1, [start, end])
        current[0] = min(current[0], start)
        current[1] = max(current[1], end)
        cursor = end
    return [{"page": page, "start": start, "end": end}
            for page, (start, end) in sorted(bounds.items()) if start < end]


def markdown_from_zip(record: dict, url: str) -> tuple[str, Path, list[dict]]:
    raw_dir = STATE_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / f"{safe_key(record['attachment_key'])}.zip"
    response = requests.get(url, timeout=300, stream=True)
    response.raise_for_status()
    advertised = int(response.headers.get("Content-Length", 0))
    if advertised > MAX_ARCHIVE_BYTES:
        raise RuntimeError(f"MinerU archive exceeds {MAX_ARCHIVE_BYTES} bytes")
    downloaded = 0
    temporary = archive.with_suffix(".zip.part")
    try:
        with temporary.open("wb") as stream:
            for block in response.iter_content(1024 * 1024):
                downloaded += len(block)
                if downloaded > MAX_ARCHIVE_BYTES:
                    raise RuntimeError(f"MinerU archive exceeds {MAX_ARCHIVE_BYTES} bytes")
                stream.write(block)
        os.replace(temporary, archive)
    finally:
        temporary.unlink(missing_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        if sum(info.file_size for info in bundle.infolist()) > MAX_UNCOMPRESSED_BYTES:
            raise RuntimeError("MinerU archive expands beyond the safety limit")
        names = [name for name in bundle.namelist() if name.lower().endswith(".md")]
        if not names:
            raise RuntimeError("MinerU archive contains no Markdown")
        markdown = bundle.read(max(names, key=lambda name: bundle.getinfo(name).file_size)).decode("utf-8")
        content_names = [name for name in bundle.namelist()
                         if name.endswith("_content_list.json") and "_v2" not in name]
        content = json.loads(bundle.read(content_names[0])) if content_names else []
    # Preserve captions and references; only remove paths to images not copied beside the article.
    markdown = re.sub(r"!\[([^]]*)\]\([^)]+\)", r"[Figure: \1]", markdown)
    markdown = re.sub(r"\n{4,}", "\n\n\n", markdown).strip() + "\n"
    content_path = raw_dir / f"{safe_key(record['attachment_key'])}.content_list.json"
    atomic_write_text(content_path, json.dumps(content, ensure_ascii=False))
    archive.unlink()
    return markdown, content_path, page_spans_from_content(content, markdown)


def output_paths(record: dict) -> tuple[Path, Path]:
    attachment_key = safe_key(record["attachment_key"])
    parent_key = safe_key(record["parent_key"]) if record["parent_key"] else "standalone"
    stem = f"zotero_{parent_key}_{attachment_key}"
    article = OUTPUT_DIR / f"{stem}.md"
    return article, article.with_suffix(".metadata.json")


def save_article(record: dict, markdown: str, content_path: Path, page_spans: list[dict], model: str) -> Path:
    article, sidecar = output_paths(record)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = Path(record.get("original_path", record["path"]))
    parsed_pdf = Path(record["path"])
    pdf_sha256 = sha256_file(pdf)
    atomic_write_text(article, markdown)
    metadata = {"title": record["title"], "authors": record["authors"],
                "version": "zotero-pdf", "source_pdf": str(pdf),
                "parser": "mineru", "parser_version": f"api-v4-{model}",
                "completeness": "not_assessed", "zotero_item_key": record["parent_key"],
                "zotero_attachment_key": record["attachment_key"], "pdf_sha256": pdf_sha256,
                "parsed_pdf_sha256": sha256_file(parsed_pdf),
                "mineru_content_list": str(content_path),
                "content_sha256": hashlib.sha256(markdown.encode("utf-8")).hexdigest(),
                "page_spans": page_spans}
    if record["year"] is not None:
        metadata["year"] = record["year"]
    if record["doi"]:
        metadata["doi"] = record["doi"]
    atomic_write_text(sidecar, json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    journal({"attachment_key": record["attachment_key"], "state": "converted",
             "article": str(article), "characters": len(markdown)})
    return article


def refresh_existing(record: dict) -> bool:
    article, sidecar = output_paths(record)
    if not article.exists() or not sidecar.exists():
        return False
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    source_pdf = Path(record["path"])
    current_pdf_hash = sha256_file(source_pdf)
    if metadata.get("pdf_sha256") != current_pdf_hash:
        return False
    markdown = article.read_text(encoding="utf-8")
    current_content_hash = hashlib.sha256(markdown.encode("utf-8")).hexdigest()
    if metadata.get("content_sha256") not in (None, current_content_hash):
        return False
    metadata.update({"title": record["title"], "authors": record["authors"],
                     "zotero_item_key": record["parent_key"],
                     "zotero_attachment_key": record["attachment_key"]})
    if record["year"] is None:
        metadata.pop("year", None)
    else:
        metadata["year"] = record["year"]
    if record["doi"]:
        metadata["doi"] = record["doi"]
    else:
        metadata.pop("doi", None)
    metadata["parsed_pdf_sha256"] = metadata.get("parsed_pdf_sha256", current_pdf_hash)
    content_value = metadata.get("mineru_content_list")
    if content_value and Path(content_value).exists():
        content = json.loads(Path(content_value).read_text(encoding="utf-8"))
        metadata["content_sha256"] = current_content_hash
        metadata["page_spans"] = page_spans_from_content(content, markdown)
        atomic_write_text(sidecar, json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
        return True
    archive_value = metadata.get("mineru_archive")
    if not archive_value or not Path(archive_value).exists():
        return True
    archive = Path(archive_value).resolve()
    raw_dir = (STATE_DIR / "raw").resolve()
    if not archive.is_relative_to(raw_dir):
        raise ValueError("Legacy MinerU archive must remain inside the importer state directory")
    with zipfile.ZipFile(archive) as bundle:
        names = [name for name in bundle.namelist()
                 if name.endswith("_content_list.json") and "_v2" not in name]
        content = json.loads(bundle.read(names[0])) if names else []
    content_path = archive.with_suffix(".content_list.json")
    atomic_write_text(content_path, json.dumps(content, ensure_ascii=False))
    metadata.pop("mineru_archive", None)
    metadata["mineru_content_list"] = str(content_path)
    metadata["content_sha256"] = current_content_hash
    metadata["page_spans"] = page_spans_from_content(content, markdown)
    atomic_write_text(sidecar, json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    archive.unlink()
    return True


def import_records(records: list[dict], limit: int | None) -> None:
    pending, oversized = [], []
    for record in sorted(records, key=lambda row: row["date_added"], reverse=True):
        article, sidecar = output_paths(record)
        if not record["exists"] or record["pages"] is None:
            continue
        if refresh_existing(record):
            continue
        if limit is not None and len(pending) + len(oversized) >= limit:
            break
        if record["pages"] > 200:
            oversized.append(record)
            continue
        pending.append(record)
    for record in oversized:
        import_split_record(record, Path(record["path"]), record["pages"])
    if not pending and not oversized:
        print("No eligible PDF remains to convert")
        return
    if not pending:
        return
    bearer = token()
    groups = (("vlm", pending),)
    for model, group in groups:
        for start in range(0, len(group), BATCH_SIZE):
            batch = group[start:start + BATCH_SIZE]
            if not batch:
                continue
            batch_id, urls = submit_batch(batch, bearer, model)
            journal({"batch_id": batch_id, "state": "submitted", "model": model,
                     "attachments": [row["attachment_key"] for row in batch]})
            upload(batch, urls)
            results = poll(batch_id, bearer, {row["attachment_key"] for row in batch})
            for record in batch:
                result = results[record["attachment_key"]]
                if result.get("state") != "done":
                    continue
                try:
                    markdown, content_path, page_spans = markdown_from_zip(record, result["full_zip_url"])
                    save_article(record, markdown, content_path, page_spans, model)
                except Exception as error:
                    journal({"attachment_key": record["attachment_key"], "state": "failed",
                             "error": str(error)})
                    print(f"FAILED {record['attachment_key']}: {error}", file=sys.stderr)


def import_split_record(record: dict, repaired: Path, pages: int, part_size: int = 190) -> None:
    parts_dir = STATE_DIR / "repaired" / safe_key(record["attachment_key"])
    parts_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for first in range(1, pages + 1, part_size):
        last = min(pages, first + part_size - 1)
        part = parts_dir / f"{safe_key(record['attachment_key'])}-p{first:03d}-{last:03d}.pdf"
        subprocess.run(["qpdf", str(repaired), "--pages", str(repaired), f"{first}-{last}",
                        "--", str(part)], check=True)
        parts.append({**record, "attachment_key": f"{record['attachment_key']}_p{first:03d}",
                      "path": str(part), "pages": last - first + 1,
                      "page_offset": first - 1, "text_layer": has_text_layer(part)})
    bearer = token()
    model = "vlm"
    batch_id, urls = submit_batch(parts, bearer, model)
    journal({"batch_id": batch_id, "state": "submitted", "model": model,
             "attachments": [row["attachment_key"] for row in parts],
             "combined_attachment": record["attachment_key"]})
    upload(parts, urls)
    results = poll(batch_id, bearer, {row["attachment_key"] for row in parts})
    sections, spans, combined_content, cursor = [], [], [], 0
    for part in parts:
        result = results[part["attachment_key"]]
        if result.get("state") != "done":
            raise RuntimeError(f"MinerU failed split {part['attachment_key']}: {result.get('err_msg')}")
        markdown, content_path, local_spans = markdown_from_zip(part, result["full_zip_url"])
        separator = "" if not sections else "\n\n"
        cursor += len(separator)
        sections.append(separator + markdown)
        for span in local_spans:
            spans.append({"page": span["page"] + part["page_offset"],
                          "start": span["start"] + cursor, "end": span["end"] + cursor})
        content = json.loads(content_path.read_text(encoding="utf-8"))
        for item in content:
            if type(item.get("page_idx")) is int:
                item["page_idx"] += part["page_offset"]
        combined_content.extend(content)
        cursor += len(markdown)
        content_path.unlink()
    combined_path = STATE_DIR / "raw" / f"{record['attachment_key']}.content_list.json"
    atomic_write_text(combined_path, json.dumps(combined_content, ensure_ascii=False))
    original_key = record["attachment_key"]
    final_record = {**record, "attachment_key": original_key, "path": str(repaired)}
    save_article(final_record, "".join(sections), combined_path, spans, f"{model}-split")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inventory", "import"))
    parser.add_argument("--limit", type=int, help="Convert at most this many eligible PDFs")
    parser.add_argument("--repair-pdf", metavar="ATTACHMENT_KEY",
                        help="Repair one malformed PDF with qpdf before importing it")
    args = parser.parse_args()
    records, duplicates = deduplicate(inventory())
    path = write_inventory(records, duplicates)
    print(json.dumps({"local_pdf_records": len(records) + len(duplicates),
                      "unique_candidates": len(records),
                      "eligible_unique_pdfs": sum(row["exists"] and row["pages"] is not None
                                                   for row in records),
                      "duplicates": len(duplicates),
                      "missing_files": sum(not row["exists"] for row in records),
                      "requires_split": sum((row["pages"] or 0) > 200 for row in records),
                      "unknown_pages": sum(row["pages"] is None for row in records),
                      "inventory": str(path)}, indent=2))
    if args.command == "import":
        if args.repair_pdf:
            matches = [row for row in records if row["attachment_key"] == args.repair_pdf]
            if len(matches) != 1:
                raise SystemExit(f"Unknown unique attachment key: {args.repair_pdf}")
            record = matches[0]
            repaired_dir = STATE_DIR / "repaired"
            repaired_dir.mkdir(parents=True, exist_ok=True)
            repaired = repaired_dir / f"{args.repair_pdf}.pdf"
            repair = subprocess.run(["qpdf", record["path"], str(repaired)])
            if repair.returncode not in (0, 3) or not repaired.exists():
                raise RuntimeError(f"qpdf could not repair the PDF (exit {repair.returncode})")
            check = subprocess.run(["qpdf", "--check", str(repaired)],
                                   stdout=subprocess.DEVNULL)
            if check.returncode not in (0, 3):
                raise RuntimeError(f"qpdf rejected the repaired PDF (exit {check.returncode})")
            record = {**record, "original_path": record["path"], "path": str(repaired),
                      "exists": True, "pages": pdf_pages(repaired),
                      "text_layer": has_text_layer(repaired)}
            if record["pages"] and record["pages"] > 200:
                import_split_record(record, repaired, record["pages"])
            else:
                import_records([record], 1)
        else:
            import_records(records, args.limit)


if __name__ == "__main__":
    main()
