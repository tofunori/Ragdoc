"""Immutable source snapshots and explicit, non-inferred scientific provenance."""

import hashlib
import json
import os
import re
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .document_structure import parse_sections, section_metadata


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_sidecar(path: Path) -> dict:
    sidecar = path.with_suffix(".metadata.json")
    if not sidecar.exists():
        return {}
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Metadata sidecar must be a JSON object")
    for key in ("title", "doi", "version", "source_pdf", "parser", "parser_version", "collection"):
        if key in data and not isinstance(data[key], str):
            raise ValueError(f"Metadata {key} must be a string")
    if "authors" in data and (not isinstance(data["authors"], list) or
                              any(not isinstance(a, str) for a in data["authors"])):
        raise ValueError("Metadata authors must be a list of strings")
    if "year" in data and (type(data["year"]) is not int or not 1000 <= data["year"] <= 3000):
        raise ValueError("Metadata year must be an integer between 1000 and 3000")
    spans = data.get("page_spans", [])
    if not isinstance(spans, list):
        raise ValueError("page_spans must be a list")
    previous_end = -1
    previous_page = 0
    for span in spans:
        if not isinstance(span, dict) or any(type(span.get(k)) is not int for k in ("start", "end", "page")):
            raise ValueError("Page spans require integer start, end and page")
        if not 0 <= span["start"] < span["end"] or span["page"] < 1:
            raise ValueError("Invalid page span")
        if span["start"] < previous_end or span["page"] < previous_page:
            raise ValueError("Page spans must be ordered and non-overlapping")
        previous_end = span["end"]
        previous_page = span["page"]
    return data


class Library:
    """Local canonical store. Readers never create files or connect to a live index."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()

    def snapshot(self, content: str) -> str:
        digest = sha256(content)
        folder = self.root / "snapshots"
        folder.mkdir(parents=True, exist_ok=True)
        destination = folder / f"{digest}.md"
        if destination.exists():
            self.read(digest)  # A corrupt snapshot must not be silently reused.
            return digest
        fd, temporary = tempfile.mkstemp(dir=folder, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return digest

    def read(self, digest: str) -> str:
        if not re.fullmatch(r"[a-f0-9]{64}", digest or ""):
            raise ValueError("Invalid canonical content hash")
        with (self.root / "snapshots" / f"{digest}.md").open(encoding="utf-8", newline="") as stream:
            content = stream.read()
        if sha256(content) != digest:
            raise ValueError("Canonical snapshot checksum mismatch")
        return content

    def record(self, source: str, status: str, digest: str | None = None, error: str | None = None):
        self.root.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.root / "ingestion.sqlite") as db:
            db.execute("CREATE TABLE IF NOT EXISTS events (source TEXT, status TEXT, digest TEXT, error TEXT, at TEXT)")
            db.execute("INSERT INTO events VALUES (?, ?, ?, ?, ?)",
                       (source, status, digest, error, datetime.now(timezone.utc).isoformat()))

    def latest_events(self) -> list[dict]:
        path = self.root / "ingestion.sqlite"
        if not path.exists():
            return []
        with sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True) as db:
            db.row_factory = sqlite3.Row
            return [dict(r) for r in db.execute(
                "SELECT source, status, digest, error, at FROM events WHERE rowid IN "
                "(SELECT MAX(rowid) FROM events GROUP BY source) ORDER BY source")]


def document_metadata(path: Path, content: str, sidecar: dict) -> dict:
    """Only supplied bibliography is treated as known; a filename is not a title."""
    bibliographic = {k: sidecar[k] for k in
                     ("title", "authors", "year", "doi", "version", "source_pdf", "collection",
                      "zotero_item_key", "zotero_attachment_key") if k in sidecar}
    metadata = {
        "document_id": sha256("source:" + path.name),
        "canonical_sha256": sha256(content),
        "bibliography_json": json.dumps(bibliographic, ensure_ascii=False),
        "title": sidecar.get("title", path.stem),
        "title_verified": bool(sidecar.get("title")),
        "metadata_sha256": sha256(json.dumps(sidecar, sort_keys=True, ensure_ascii=False)),
        "parser": sidecar.get("parser", "unknown"),
        "parser_version": sidecar.get("parser_version", "unknown"),
        "completeness": sidecar.get("completeness", "not_assessed"),
    }
    for key in ("year", "doi", "collection"):
        if key in sidecar:
            metadata[key] = sidecar[key]
    return metadata


def locate_chunks(content: str, texts: list[str], sidecar: dict) -> list[dict]:
    """Locate exact text only. Do not invent page numbers for legacy Markdown."""
    pages = sidecar.get("page_spans", []) if sidecar.get("content_sha256") == sha256(content) else []
    sections = parse_sections(content)
    cursor = 0
    locations = []
    for text in texts:
        # Repeated wording cannot prove which section/page the chunk came from.
        start = content.find(text, cursor) if text and content.count(text) == 1 else -1
        location = {"locator_status": "unresolved"}
        if start >= 0:
            end = start + len(text)
            location.update(locator_status="exact", char_start=start, char_end=end)
            location.update(section_metadata(content, start, end, sections))
            starts = [s["page"] for s in pages if s["start"] <= start < s["end"] <= len(content)]
            ends = [s["page"] for s in pages if 0 <= s["start"] <= end - 1 < s["end"] <= len(content)]
            if len(set(starts)) == len(set(ends)) == 1 and starts[0] <= ends[0]:
                location["page_start"] = starts[0]
                location["page_end"] = ends[0]
            cursor = start + 1
        locations.append(location)
    return locations


def docling_page_spans(document, canonical: str) -> list[dict]:
    """Preserve only unambiguous exact matches to Docling's PDF page provenance."""
    spans = []
    for item, _level in document.iterate_items():
        text = getattr(item, 'text', None)
        if not text or canonical.count(text) != 1:
            continue
        pages = {p.page_no for p in getattr(item, 'prov', [])}
        if len(pages) != 1:
            continue
        start = canonical.index(text)
        spans.append({"start": start, "end": start + len(text), "page": next(iter(pages))})
    return sorted(spans, key=lambda s: s['start'])


def provenance(metadata: dict) -> dict:
    bibliographic = json.loads(metadata.get("bibliography_json", "{}"))
    return {
        "document_id": metadata.get("document_id"),
        "source": metadata.get("source", metadata.get("filename")),
        "content_sha256": metadata.get("canonical_sha256"),
        "bibliography": {k: bibliographic.get(k) for k in ("title", "authors", "year", "doi", "version")},
        "location": {
            **{k: metadata.get(k) for k in (
                "section", "section_id", "section_level", "section_path", "section_start", "section_end",
                "page_start", "page_end", "char_start", "char_end", "structure_version"
            )},
            "section_types": json.loads(metadata.get("section_types_json", "[]")),
            "section_overlap": bool(metadata.get("section_overlap", False)),
        },
        "locator_status": metadata.get("locator_status", "unavailable"),
        "source_pdf": bibliographic.get("source_pdf"),
        "completeness": metadata.get("completeness", "not_assessed"),
    }
