"""Persistent exact and lexical index for MinerU tables, figures and charts."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3


SCHEMA = """
CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    kind TEXT NOT NULL,
    label TEXT NOT NULL,
    page INTEGER,
    bbox_json TEXT,
    caption TEXT NOT NULL,
    body TEXT NOT NULL,
    image_path TEXT,
    manifest_path TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS artifacts_source ON artifacts(source);
CREATE INDEX IF NOT EXISTS artifacts_label ON artifacts(source, label COLLATE NOCASE);
CREATE VIRTUAL TABLE IF NOT EXISTS artifacts_fts USING fts5(
    artifact_id UNINDEXED, source UNINDEXED, kind, label, caption, body,
    tokenize='unicode61 remove_diacritics 2'
);
"""


class ArtifactIndex:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.database = self.root / "artifacts.sqlite3"

    def _connect(self, readonly: bool = False) -> sqlite3.Connection:
        if readonly:
            connection = sqlite3.connect(f"{self.database.as_uri()}?mode=ro", uri=True)
            connection.execute("PRAGMA query_only=ON")
        else:
            self.root.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.database)
            connection.executescript(SCHEMA)
        connection.row_factory = sqlite3.Row
        return connection

    def index(self, sources: set[str] | None = None) -> dict:
        manifests = sorted(self.root.glob("*/manifest.json"))
        selected = []
        for path in manifests:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            source = manifest.get("source")
            if not isinstance(source, str) or (sources is not None and source not in sources):
                continue
            selected.append((path, manifest))
        indexed_sources = {manifest["source"] for _, manifest in selected}
        requested = sources or indexed_sources
        with self._connect() as db:
            for source in requested:
                ids = [row[0] for row in db.execute(
                    "SELECT artifact_id FROM artifacts WHERE source=?", (source,)
                )]
                db.execute("DELETE FROM artifacts WHERE source=?", (source,))
                db.executemany("DELETE FROM artifacts_fts WHERE artifact_id=?", ((item,) for item in ids))
            count = 0
            for manifest_path, manifest in selected:
                source = manifest["source"]
                for item in manifest.get("artifacts", []):
                    if not isinstance(item, dict) or not item.get("artifact_id"):
                        continue
                    image = item.get("image")
                    image_path = None
                    if image:
                        candidate = (manifest_path.parent / image).resolve()
                        if candidate.is_relative_to(manifest_path.parent.resolve()) and candidate.is_file():
                            image_path = str(candidate)
                    row = (
                        str(item["artifact_id"]), source, str(item.get("type", "unknown")),
                        str(item.get("label", "")), item.get("page"),
                        json.dumps(item.get("bbox")), str(item.get("caption", "")),
                        str(item.get("body", "")), image_path, str(manifest_path),
                    )
                    db.execute("INSERT OR REPLACE INTO artifacts VALUES (?,?,?,?,?,?,?,?,?,?)", row)
                    db.execute("INSERT INTO artifacts_fts VALUES (?,?,?,?,?,?)", row[:4] + row[6:8])
                    count += 1
        return {"sources": len(indexed_sources), "artifacts": count}

    def search(self, query: str = "", source: str | None = None,
               kind: str | None = None, label: str | None = None,
               limit: int = 20) -> list[dict]:
        if not self.database.exists():
            return []
        clauses, params = [], []
        if source:
            clauses.append("a.source=?")
            params.append(source)
        if kind:
            clauses.append("a.kind=?")
            params.append(kind.lower())
        if label:
            clauses.append("a.label=? COLLATE NOCASE")
            params.append(label)
        join = ""
        order = "a.source, a.page, a.label"
        if query.strip():
            join = "JOIN artifacts_fts f ON f.artifact_id=a.artifact_id"
            clauses.append("artifacts_fts MATCH ?")
            params.append(query.strip())
            order = "bm25(artifacts_fts), a.source, a.page"
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        sql = f"SELECT a.* FROM artifacts a {join}{where} ORDER BY {order} LIMIT ?"
        params.append(limit)
        with self._connect(readonly=True) as db:
            return [self._public(dict(row)) for row in db.execute(sql, params)]

    def get(self, artifact_id: str) -> dict | None:
        if not self.database.exists():
            return None
        with self._connect(readonly=True) as db:
            row = db.execute("SELECT * FROM artifacts WHERE artifact_id=?", (artifact_id,)).fetchone()
            return self._public(dict(row)) if row else None

    @staticmethod
    def _public(row: dict) -> dict:
        row["bbox"] = json.loads(row.pop("bbox_json")) if row.get("bbox_json") else None
        row["image_available"] = bool(row.get("image_path"))
        row.pop("manifest_path", None)
        return row
