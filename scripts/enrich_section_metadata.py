#!/usr/bin/env python3
"""Enrich existing chunks with canonical section metadata without re-embedding."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import chromadb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.index_incremental import acquire_lock, release_lock
from src.chroma_connection import open_chroma_client
from src.chroma_reads import read_collection
from src.config import CHROMA_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL, LEXICAL_INDEX_PATH, LIBRARY_PATH
from src.document_structure import parse_sections, section_metadata
from src.index_safety import replace_document, update_collection_state
from src.lexical_index import PersistentLexicalIndex
from src.library import Library


_SECTION_KEYS = {
    "section", "section_id", "section_level", "section_path", "section_start", "section_end",
    "section_types_json", "section_overlap", "structure_version",
    "section_is_abstract", "section_is_introduction", "section_is_methods",
    "section_is_results", "section_is_discussion", "section_is_conclusion",
    "section_is_references", "section_is_supplementary", "section_is_acknowledgements",
}


def _vectors(values) -> list[list[float]]:
    if values is None:
        return []
    return [list(vector) for vector in values]


def _rows(data: dict) -> dict:
    vectors = _vectors(data["embeddings"])
    return {chunk_id: (document, metadata, vector)
            for chunk_id, document, metadata, vector in
            zip(data["ids"], data["documents"], data["metadatas"], vectors)}


def _same_rows(actual: dict, expected: dict) -> bool:
    """Verify identity plus numerically stable Chroma vector round-tripping."""
    if set(actual) != set(expected):
        return False
    for chunk_id, (actual_document, actual_metadata, actual_vector) in actual.items():
        expected_document, expected_metadata, expected_vector = expected[chunk_id]
        if actual_document != expected_document or actual_metadata != expected_metadata:
            return False
        if len(actual_vector) != len(expected_vector):
            return False
        if any(not math.isclose(float(left), float(right), rel_tol=1e-6, abs_tol=1e-7)
               for left, right in zip(actual_vector, expected_vector)):
            return False
    return True


def enrich(*, apply: bool, sources: list[str] | None = None) -> dict:
    lock_path = CHROMA_DB_PATH.parent / ".indexing.lock"
    lock = acquire_lock(lock_path)
    if not lock:
        raise RuntimeError("Another index writer holds the shared lock")
    try:
        client, _mode = open_chroma_client(chromadb, CHROMA_DB_PATH)
        collection = client.get_collection(name=COLLECTION_NAME)
        state = dict(collection.metadata or {})
        if state.get("ragdoc_write_state", "ready") != "ready":
            raise RuntimeError("Section enrichment requires a ready collection")
        if state.get("ragdoc_repairing", False) and not apply:
            raise RuntimeError("A blocked section migration must be resumed with --apply")
        if state.get("embedding_model") != EMBEDDING_MODEL:
            raise RuntimeError("Configured embedding model does not match the collection")
        initial_revision = state.get("ragdoc_revision")
        resuming = bool(state.get("ragdoc_repairing", False))
        if initial_revision is None:
            raise RuntimeError("Section enrichment requires a versioned collection")

        catalogue = read_collection(collection, include=["metadatas"])
        available = sorted({metadata.get("source") for metadata in catalogue["metadatas"]
                            if metadata.get("source")})
        selected = available if sources is None else list(dict.fromkeys(sources))
        missing = sorted(set(selected) - set(available))
        if missing:
            raise RuntimeError(f"Unknown indexed sources: {', '.join(missing)}")

        library = Library(LIBRARY_PATH)
        changed_sources: set[str] = set()
        stats = {"sources": len(selected), "changed_sources": 0, "exact_chunks": 0,
                 "unresolved_chunks": 0, "cross_section_chunks": 0, "dry_run": not apply}
        if apply:
            update_collection_state(collection, ragdoc_repairing=True)
        for source in selected:
            current = read_collection(
                collection, include=["documents", "metadatas", "embeddings"],
                where={"source": source},
            )
            digests = {metadata.get("canonical_sha256") for metadata in current["metadatas"]}
            if len(digests) != 1 or None in digests:
                stats["unresolved_chunks"] += len(current["ids"])
                continue
            canonical = library.read(next(iter(digests)))
            sections = parse_sections(canonical)
            enriched = []
            changed = False
            for text, old in zip(current["documents"], current["metadatas"]):
                metadata = dict(old)
                for key in _SECTION_KEYS:
                    metadata.pop(key, None)
                start, end = metadata.get("char_start"), metadata.get("char_end")
                exact = (metadata.get("locator_status") == "exact" and type(start) is int
                         and type(end) is int and 0 <= start < end <= len(canonical)
                         and canonical[start:end] == text)
                if exact:
                    metadata.update(section_metadata(canonical, start, end, sections))
                    stats["exact_chunks"] += 1
                    stats["cross_section_chunks"] += int(metadata.get("section_overlap", False))
                else:
                    stats["unresolved_chunks"] += 1
                changed = changed or metadata != old
                enriched.append(metadata)
            if not changed:
                continue
            stats["changed_sources"] += 1
            if not apply:
                continue
            before_vectors = _vectors(current["embeddings"])
            replace_document(collection, source, {
                "ids": current["ids"], "documents": current["documents"],
                "embeddings": before_vectors, "metadatas": enriched,
            })
            verified = read_collection(
                collection, include=["documents", "metadatas", "embeddings"],
                where={"source": source},
            )
            expected = {chunk_id: (document, metadata, vector) for chunk_id, document, metadata, vector in
                        zip(current["ids"], current["documents"], enriched, before_vectors)}
            if not _same_rows(_rows(verified), expected):
                raise RuntimeError(f"Post-write verification failed for {source}")
            changed_sources.add(source)

        if apply and resuming:
            # The previous run may already have committed one or more Chroma
            # sources without recording them in this process's changed set.
            PersistentLexicalIndex(LEXICAL_INDEX_PATH).rebuild(
                collection, allow_repairing=True
            )
        elif apply and changed_sources:
            PersistentLexicalIndex(LEXICAL_INDEX_PATH).sync_sources(
                collection, changed_sources, initial_revision, allow_repairing=True
            )
        if apply:
            update_collection_state(collection, ragdoc_repairing=False)
        return stats
    finally:
        release_lock(lock_path, lock)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Write changes; otherwise report only")
    parser.add_argument("--source", action="append", help="Limit to one indexed source; repeatable")
    args = parser.parse_args()
    print(enrich(apply=args.apply, sources=args.source))


if __name__ == "__main__":
    main()
