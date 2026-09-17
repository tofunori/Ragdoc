"""Revision-pinned SQLite FTS index for bounded-memory lexical retrieval."""

from __future__ import annotations

import json
import os
import re
import sqlite3
from pathlib import Path


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "de", "des", "du",
    "en", "et", "for", "from", "how", "in", "is", "la", "le", "les",
    "of", "on", "or", "our", "par", "pour", "que", "qui", "the", "to",
    "un", "une", "what", "which", "with", "quelles", "quels", "comment",
}

_SCHEMA_VERSION = "3"
_MAX_CHUNKS_PER_SOURCE = 10
_RANK_FUNCTION = "bm25(0.0, 0.0, 1.0, 8.0, 4.0, 20.0, 6.0, 0.0)"


def _query_terms(query: str) -> list[str]:
    terms: list[str] = []
    for term in re.findall(r"[^\W_]+", query.casefold(), flags=re.UNICODE):
        if len(term) < 2 or term in _STOPWORDS or term in terms:
            continue
        terms.append(term)
    return terms[:32]


def _fts_query(query: str) -> str:
    # Quoted terms prevent user text from becoming FTS query syntax. OR keeps
    # acronym/identifier searches useful when surrounding prose differs.
    compact_identifier = bool(re.fullmatch(r"\s*[\w./:-]+\s*", query, flags=re.UNICODE))
    doi = bool(re.search(r"\b10\.\d{4,9}/\S+", query, flags=re.IGNORECASE))
    if doi or compact_identifier:
        terms = []
        for term in re.findall(r"[^\W_]+", query.casefold(), flags=re.UNICODE):
            if term not in terms:
                terms.append(term)
        terms = terms[:32]
    else:
        terms = _query_terms(query)
    if not terms:
        return ""
    if doi or compact_identifier:
        phrase = f'"{" ".join(terms)}"'
        # Prefer catalogue identifiers and filenames over incidental citations
        # in article bodies, while retaining a body fallback for legacy rows.
        return f"identifier : {phrase} OR source_terms : {phrase} OR {phrase}"

    expression = " OR ".join(f'"{term}"' for term in terms)
    raw_terms = list(dict.fromkeys(
        re.findall(r"[^\W_]+", query.casefold(), flags=re.UNICODE)
    ))[:32]
    if len(raw_terms) >= 2:
        # An exact title phrase is valuable evidence for title-like queries.
        # The broad OR branch preserves recall for ordinary questions.
        phrase = f'"{" ".join(raw_terms)}"'
        return f"title : {phrase} OR ({expression})"
    return expression


def _metadata_fields(source: str, metadata: dict | None) -> tuple[str, str, str]:
    metadata = metadata or {}
    bibliography = metadata.get("bibliography_json")
    if isinstance(bibliography, str):
        try:
            bibliography = json.loads(bibliography)
        except (TypeError, ValueError):
            bibliography = {}
    if not isinstance(bibliography, dict):
        bibliography = {}

    title = metadata.get("title") or bibliography.get("title") or ""
    authors = metadata.get("authors") or bibliography.get("authors")
    author_values: list[str] = []
    if isinstance(authors, list):
        author_values.extend(str(author) for author in authors if author)
    elif isinstance(authors, str):
        author_values.append(authors)

    identifier_values = []
    for key in ("doi", "pmid", "arxiv_id", "isbn", "zotero_item_key", "zotero_attachment_key"):
        value = metadata.get(key) or bibliography.get(key)
        if value:
            identifier_values.append(str(value))
    return str(title), "\n".join(author_values), "\n".join(identifier_values)


def _lexical_row(chunk_id: str, source: str, document: str, metadata: dict | None) -> tuple:
    title, authors, identifier = _metadata_fields(source, metadata)
    return (
        chunk_id,
        source,
        document,
        title,
        authors,
        identifier,
        source,
        json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True),
    )


def _sql_filter(where: dict | None) -> tuple[str, list[object]]:
    """Compile the supported Chroma filter subset to parameterized SQLite SQL."""
    if where is None:
        return "1", []
    if not isinstance(where, dict) or not where:
        raise ValueError("where must be a nonempty filter object")
    if "$and" in where or "$or" in where:
        if len(where) != 1:
            raise ValueError("Logical filters cannot contain sibling fields")
        operator = "$and" if "$and" in where else "$or"
        clauses = where[operator]
        if not isinstance(clauses, list) or not clauses:
            raise ValueError(f"{operator} requires a nonempty list")
        compiled = [_sql_filter(clause) for clause in clauses]
        joiner = " AND " if operator == "$and" else " OR "
        return "(" + joiner.join(fragment for fragment, _ in compiled) + ")", [
            parameter for _, parameters in compiled for parameter in parameters
        ]
    if len(where) != 1:
        return _sql_filter({"$and": [{key: value} for key, value in where.items()]})

    field, condition = next(iter(where.items()))
    if not isinstance(field, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", field):
        raise ValueError(f"Unsupported lexical filter field: {field}")
    expression = "source" if field == "source" else f"json_extract(metadata_json, '$.{field}')"
    if not isinstance(condition, dict):
        return f"{expression} = ?", [condition]
    if len(condition) != 1:
        raise ValueError("Filter operator object must contain exactly one operator")
    operator, target = next(iter(condition.items()))
    sql_operators = {"$eq": "=", "$ne": "!=", "$gt": ">", "$gte": ">=", "$lt": "<", "$lte": "<="}
    if operator in {"$ne"}:
        return f"({expression} IS NULL OR {expression} != ?)", [target]
    if operator in sql_operators:
        return f"{expression} {sql_operators[operator]} ?", [target]
    if operator in {"$in", "$nin"}:
        if not isinstance(target, list) or not target:
            raise ValueError(f"{operator} requires a nonempty list")
        placeholders = ", ".join("?" for _ in target)
        if operator == "$in":
            return f"{expression} IN ({placeholders})", list(target)
        return f"({expression} IS NULL OR {expression} NOT IN ({placeholders}))", list(target)
    raise ValueError(f"Unsupported lexical filter operator: {operator}")


def _single_source(where: dict | None) -> str | None:
    if not isinstance(where, dict):
        return None
    if "source" in where and len(where) == 1:
        condition = where["source"]
        if isinstance(condition, str):
            return condition
        if isinstance(condition, dict) and len(condition) == 1:
            if isinstance(condition.get("$eq"), str):
                return condition["$eq"]
            values = condition.get("$in")
            if isinstance(values, list) and len(values) == 1 and isinstance(values[0], str):
                return values[0]
    clauses = where.get("$and")
    if isinstance(clauses, list):
        sources = {source for clause in clauses if (source := _single_source(clause)) is not None}
        if len(sources) == 1:
            return next(iter(sources))
    return None


class PersistentLexicalIndex:
    """Small connection-per-operation wrapper around an atomic FTS5 sidecar."""

    def __init__(self, path: Path | str):
        self.path = Path(path)

    @staticmethod
    def _schema(connection: sqlite3.Connection) -> None:
        connection.execute("CREATE TABLE state (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute(
            "CREATE VIRTUAL TABLE passages USING fts5("
            "chunk_id UNINDEXED, source UNINDEXED, document, title, authors, identifier, "
            "source_terms, metadata_json UNINDEXED, "
            "tokenize='porter unicode61 remove_diacritics 2')"
        )
        connection.execute(
            "INSERT INTO passages(passages, rank) VALUES('rank', ?)",
            (_RANK_FUNCTION,),
        )

    @staticmethod
    def _state(connection: sqlite3.Connection) -> dict[str, str]:
        return dict(connection.execute("SELECT key, value FROM state"))

    def status(self, revision: object, chunk_count: int | None = None) -> dict:
        if not self.path.is_file():
            return {"ready": False, "reason": "missing", "path": str(self.path)}
        try:
            with sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, timeout=5) as connection:
                state = self._state(connection)
                indexed = connection.execute("SELECT COUNT(*) FROM passages").fetchone()[0]
            count_matches = chunk_count is None or indexed == chunk_count
            ready = (state.get("revision") == str(revision)
                     and state.get("schema_version") == _SCHEMA_VERSION
                     and count_matches)
            return {
                "ready": ready,
                "reason": None if ready else "revision_or_count_mismatch",
                "path": str(self.path),
                "revision": state.get("revision"),
                "chunks": indexed,
                "schema_version": state.get("schema_version"),
                "ranking": "fielded_bm25",
                "max_chunks_per_source": _MAX_CHUNKS_PER_SOURCE,
            }
        except (OSError, sqlite3.Error) as error:
            return {"ready": False, "reason": f"unreadable: {error}", "path": str(self.path)}

    def rebuild(self, collection, *, allow_repairing: bool = False) -> dict:
        metadata = dict(collection.metadata or {})
        revision = metadata.get("ragdoc_revision")
        if (metadata.get("ragdoc_write_state", "ready") != "ready"
                or (metadata.get("ragdoc_repairing", False) and not allow_repairing)):
            raise RuntimeError("Cannot build lexical index from a non-ready collection")
        if revision is None:
            raise RuntimeError("Cannot build lexical index without a collection revision")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.tmp-{os.getpid()}")
        temporary.unlink(missing_ok=True)
        seen: set[str] = set()
        inserted = 0
        try:
            with sqlite3.connect(temporary) as connection:
                connection.execute("PRAGMA journal_mode=DELETE")
                connection.execute("PRAGMA synchronous=NORMAL")
                self._schema(connection)
                offset = 0
                while True:
                    page = collection.get(
                        include=["documents", "metadatas"], limit=500, offset=offset
                    )
                    ids = page.get("ids", [])
                    if not ids:
                        break
                    if len(ids) != len(set(ids)) or seen.intersection(ids):
                        raise RuntimeError("Collection changed during lexical index build")
                    documents = page.get("documents") or []
                    metadatas = page.get("metadatas") or []
                    if len(documents) != len(ids) or len(metadatas) != len(ids):
                        raise RuntimeError("Incomplete collection page during lexical index build")
                    rows = []
                    for chunk_id, document, chunk_metadata in zip(ids, documents, metadatas):
                        source = (chunk_metadata or {}).get("source")
                        if not source:
                            raise RuntimeError(f"Lexical index row lacks source metadata: {chunk_id}")
                        rows.append(_lexical_row(chunk_id, source, document, chunk_metadata))
                    connection.executemany(
                        "INSERT INTO passages("
                        "chunk_id, source, document, title, authors, identifier, source_terms, metadata_json"
                        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        rows,
                    )
                    seen.update(ids)
                    inserted += len(ids)
                    offset += len(ids)

                fresh = dict(collection.metadata or {})
                if (fresh.get("ragdoc_revision") != revision
                        or fresh.get("ragdoc_write_state", "ready") != "ready"
                        or (fresh.get("ragdoc_repairing", False) and not allow_repairing)):
                    raise RuntimeError("Collection changed during lexical index build")
                if inserted != collection.count():
                    raise RuntimeError(f"Lexical index count mismatch: {inserted} != {collection.count()}")
                connection.execute("INSERT INTO passages(passages) VALUES('optimize')")
                connection.executemany(
                    "INSERT INTO state(key, value) VALUES (?, ?)",
                    [("revision", str(revision)), ("chunks", str(inserted)),
                     ("schema_version", _SCHEMA_VERSION)],
                )
                connection.commit()
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)
        return self.status(revision, inserted)

    def sync_sources(self, collection, sources: set[str], previous_revision: object,
                     allow_repairing: bool = False) -> dict:
        """Update changed sources, rebuilding if the prior sidecar was not exact."""
        previous = self.status(previous_revision)
        if not previous.get("ready"):
            return self.rebuild(collection, allow_repairing=allow_repairing)

        metadata = dict(collection.metadata or {})
        revision = metadata.get("ragdoc_revision")
        if (metadata.get("ragdoc_write_state", "ready") != "ready" or revision is None
                or (metadata.get("ragdoc_repairing", False) and not allow_repairing)):
            raise RuntimeError("Cannot synchronize lexical index from a non-ready collection")
        with sqlite3.connect(self.path, timeout=30) as connection:
            connection.execute("BEGIN IMMEDIATE")
            for source in sorted(sources):
                connection.execute("DELETE FROM passages WHERE source = ?", (source,))
                offset = 0
                source_ids: set[str] = set()
                while True:
                    rows = collection.get(
                        where={"source": source}, include=["documents", "metadatas"],
                        limit=500, offset=offset,
                    )
                    ids = rows.get("ids", [])
                    if not ids:
                        break
                    documents = rows.get("documents") or []
                    metadatas = rows.get("metadatas") or []
                    if (len(ids) != len(documents) or len(ids) != len(metadatas)
                            or source_ids.intersection(ids)):
                        raise RuntimeError(f"Incomplete source during lexical sync: {source}")
                    connection.executemany(
                        "INSERT INTO passages("
                        "chunk_id, source, document, title, authors, identifier, source_terms, metadata_json"
                        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        [_lexical_row(chunk_id, source, document, chunk_metadata)
                         for chunk_id, document, chunk_metadata in zip(ids, documents, metadatas)],
                    )
                    source_ids.update(ids)
                    offset += len(ids)
            indexed = connection.execute("SELECT COUNT(*) FROM passages").fetchone()[0]
            if indexed != collection.count():
                connection.rollback()
                return self.rebuild(collection, allow_repairing=allow_repairing)
            connection.execute("UPDATE state SET value = ? WHERE key = 'revision'", (str(revision),))
            connection.execute("UPDATE state SET value = ? WHERE key = 'chunks'", (str(indexed),))
            connection.commit()
        return self.status(revision, collection.count())

    def search(self, query: str, *, top_n: int, revision: object, chunk_count: int,
               where: dict | None = None) -> tuple[list[tuple[str, float, int]], dict]:
        state = self.status(revision, chunk_count)
        if not state.get("ready"):
            raise RuntimeError(f"Lexical index unavailable: {state.get('reason')}")
        expression = _fts_query(query)
        if not expression:
            return [], {}
        candidate_limit = min(max(top_n * 20, 500), 5000)
        filter_sql, filter_parameters = _sql_filter(where)
        sql = (
            "SELECT chunk_id, source, document, metadata_json, rank "
            "FROM passages WHERE passages MATCH ? AND source NOT GLOB '._*' "
            f"AND {filter_sql}"
        )
        parameters: list[object] = [expression, *filter_parameters]
        sql += " ORDER BY rank LIMIT ?"
        parameters.append(candidate_limit)
        with sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, timeout=5) as connection:
            rows = connection.execute(sql, parameters).fetchall()
        results = []
        payload = {}
        source_counts: dict[str, int] = {}
        exact_source = _single_source(where)
        for chunk_id, row_source, document, metadata_json, score in rows:
            if exact_source is None and source_counts.get(row_source, 0) >= _MAX_CHUNKS_PER_SOURCE:
                continue
            source_counts[row_source] = source_counts.get(row_source, 0) + 1
            results.append((chunk_id, -float(score), len(results)))
            payload[chunk_id] = (document, json.loads(metadata_json))
            if len(results) >= top_n:
                break
        return results, payload
