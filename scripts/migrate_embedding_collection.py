#!/usr/bin/env python3
"""Build a resumable embedding collection from an existing Ragdoc collection.

The source collection remains untouched.  Existing chunk text, IDs, and
provenance metadata are preserved while only the embedding model is replaced.
"""

from __future__ import annotations

import argparse
import fcntl
import math
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chromadb
import voyageai
from dotenv import load_dotenv

from src.chroma_reads import read_collection
from src.index_safety import bump_revision, update_collection_state


DEFAULT_SOURCE = "ragdoc_contextualized_v1"
DEFAULT_TARGET = "ragdoc_contextualized_v1"
DEFAULT_MODEL = "voyage-context-4"
GROUP_SIZE = 25
WRITE_BATCH_SIZE = 100
MIGRATION_METADATA_KEYS = {
    "model",
    "embedding_migrated_at",
    "embedding_source_collection",
}


def _scientific_metadata(metadata: dict | None) -> dict:
    """Return provenance metadata without fields introduced by the migration."""
    return {
        key: value
        for key, value in (metadata or {}).items()
        if key not in MIGRATION_METADATA_KEYS
    }


def _rows_by_id(rows: dict) -> dict[str, tuple[str, dict]]:
    return {
        chunk_id: (document, _scientific_metadata(metadata))
        for chunk_id, document, metadata in zip(
            rows["ids"], rows["documents"], rows["metadatas"]
        )
    }


def source_is_identical(source, target, source_name: str, model: str) -> bool:
    """Compare one migrated document exactly, including text and provenance."""
    source_rows = read_collection(
        source,
        where={"source": source_name},
        include=["documents", "metadatas"],
    )
    target_rows = read_collection(
        target,
        where={"source": source_name},
        include=["documents", "metadatas"],
    )
    if any((metadata or {}).get("model") != model for metadata in target_rows["metadatas"]):
        return False
    return _rows_by_id(source_rows) == _rows_by_id(target_rows)


def require_ready_source(source) -> tuple[object, int]:
    metadata = dict(source.metadata or {})
    state = metadata.get("ragdoc_write_state", "legacy")
    if state != "ready" or metadata.get("ragdoc_repairing", False):
        raise RuntimeError(
            f"Source collection is not ready: state={state!r}, "
            f"repairing={metadata.get('ragdoc_repairing', False)!r}"
        )
    revision = metadata.get("ragdoc_revision")
    if revision is None:
        raise RuntimeError("Source collection has no ragdoc_revision")
    return revision, source.count()


def _embed_group(client: voyageai.Client, group: list[str], model: str) -> tuple[list[list[float]], int]:
    """Embed one contextual group, splitting only when Voyage rejects its token size."""
    for attempt in range(1, 6):
        try:
            response = client.contextualized_embed(
                inputs=[group],
                model=model,
                input_type="document",
            )
            vectors = response.results[0].embeddings
            if len(vectors) != len(group):
                raise RuntimeError(f"Embedding count mismatch: {len(vectors)} != {len(group)}")
            return vectors, int(getattr(response, "total_tokens", 0) or 0)
        except Exception as error:
            message = str(error).casefold()
            if "too many tokens" in message or "context window" in message:
                if len(group) == 1:
                    raise RuntimeError(
                        "A single stored chunk exceeds the Voyage context window; "
                        "the source index must be repaired before migration"
                    ) from error
                midpoint = len(group) // 2
                left_vectors, left_tokens = _embed_group(client, group[:midpoint], model)
                right_vectors, right_tokens = _embed_group(client, group[midpoint:], model)
                return left_vectors + right_vectors, left_tokens + right_tokens
            if attempt == 5:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def embed_groups(client: voyageai.Client, texts: list[str], model: str) -> tuple[list[list[float]], int]:
    embeddings: list[list[float]] = []
    token_total = 0
    for start in range(0, len(texts), GROUP_SIZE):
        vectors, tokens = _embed_group(client, texts[start:start + GROUP_SIZE], model)
        embeddings.extend(vectors)
        token_total += tokens
    return embeddings, token_total


def write_batches(collection, *, ids, documents, embeddings, metadatas) -> None:
    for start in range(0, len(ids), WRITE_BATCH_SIZE):
        end = start + WRITE_BATCH_SIZE
        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )


def validate_vectors(vectors: list[list[float]], expected: int) -> int:
    if len(vectors) != expected or not vectors:
        raise RuntimeError("Incomplete embedding response")
    dimension = len(vectors[0])
    if dimension <= 0 or any(len(v) != dimension or not all(math.isfinite(x) for x in v) for v in vectors):
        raise RuntimeError("Invalid embedding vectors")
    return dimension


def migrate(
    source_db_path: Path,
    target_db_path: Path,
    source_name: str,
    target_name: str,
    model: str,
) -> None:
    load_dotenv(source_db_path.parent / ".env")
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        raise RuntimeError("VOYAGE_API_KEY is required")
    if source_db_path == target_db_path:
        raise RuntimeError("Source and target databases must differ")
    target_db_path.mkdir(parents=True, exist_ok=True)

    source_chroma = chromadb.PersistentClient(path=str(source_db_path))
    target_chroma = chromadb.PersistentClient(path=str(target_db_path))
    source = source_chroma.get_collection(source_name)
    source_metadata = dict(source.metadata or {})
    source_revision, source_chunk_count = require_ready_source(source)
    target = target_chroma.get_or_create_collection(
        target_name,
        metadata={
            "hnsw:space": "cosine",
            "hnsw:construction_ef": 400,
            "hnsw:M": 64,
            "pipeline": source_metadata.get("pipeline", "contextualized_adaptive"),
            "embedding_model": model,
            "migration_source_collection": source_name,
            "ragdoc_write_state": "building",
            "ragdoc_repairing": False,
        },
    )
    target_metadata = dict(target.metadata or {})
    if target_metadata.get("embedding_model") != model:
        raise RuntimeError(
            f"Target model mismatch: {target_metadata.get('embedding_model')} != {model}"
        )

    source_rows = read_collection(source, include=["metadatas"])
    source_counts = Counter((metadata or {}).get("source") for metadata in source_rows["metadatas"])
    if None in source_counts:
        raise RuntimeError("Source collection contains chunks without source metadata")
    sources = sorted(source_counts)
    source_ids: dict[str, set[str]] = defaultdict(set)
    for chunk_id, metadata in zip(source_rows["ids"], source_rows["metadatas"]):
        source_ids[metadata["source"]].add(chunk_id)

    target_rows = read_collection(target, include=["metadatas"])
    target_counts = Counter((metadata or {}).get("source") for metadata in target_rows["metadatas"])
    target_ids: dict[str, set[str]] = defaultdict(set)
    target_models: dict[str, set[str | None]] = defaultdict(set)
    for chunk_id, metadata in zip(target_rows["ids"], target_rows["metadatas"]):
        source_value = (metadata or {}).get("source")
        if source_value:
            target_ids[source_value].add(chunk_id)
            target_models[source_value].add((metadata or {}).get("model"))
    client = voyageai.Client(api_key=api_key)
    completed = 0
    migrated_tokens = int(target_metadata.get("migration_tokens", 0) or 0)

    update_collection_state(
        target,
        ragdoc_write_state="building",
        migration_total_documents=len(sources),
        migration_source_chunks=len(source_rows["ids"]),
        migration_source_revision=source_revision,
    )

    for number, source_name_value in enumerate(sources, 1):
        if (
            target_counts[source_name_value] == source_counts[source_name_value]
            and target_ids[source_name_value] == source_ids[source_name_value]
            and target_models[source_name_value] == {model}
            and source_is_identical(source, target, source_name_value, model)
        ):
            completed += 1
            print(f"[{number:3d}/{len(sources)}] SKIP {source_name_value}", flush=True)
            continue

        existing = read_collection(target, where={"source": source_name_value}, include=["metadatas"])
        if existing["ids"]:
            target.delete(ids=existing["ids"])

        rows = read_collection(
            source,
            where={"source": source_name_value},
            include=["documents", "metadatas"],
        )
        ordered = sorted(
            zip(rows["ids"], rows["documents"], rows["metadatas"]),
            key=lambda row: ((row[2] or {}).get("chunk_index", 0), row[0]),
        )
        ids = [row[0] for row in ordered]
        documents = [row[1] for row in ordered]
        metadatas = []
        for _, _, original in ordered:
            metadata = dict(original or {})
            metadata["model"] = model
            metadata["embedding_migrated_at"] = datetime.now(timezone.utc).isoformat()
            metadata["embedding_source_collection"] = source_name
            metadatas.append(metadata)

        vectors, token_count = embed_groups(client, documents, model)
        dimension = validate_vectors(vectors, len(documents))
        write_batches(
            target,
            ids=ids,
            documents=documents,
            embeddings=vectors,
            metadatas=metadatas,
        )
        if not source_is_identical(source, target, source_name_value, model):
            target.delete(ids=ids)
            raise RuntimeError(f"Read-back failed for {source_name_value}")

        completed += 1
        migrated_tokens += token_count
        update_collection_state(
            target,
            migration_completed_documents=completed,
            migration_last_source=source_name_value,
            migration_tokens=migrated_tokens,
            embedding_dimension=dimension,
        )
        print(
            f"[{number:3d}/{len(sources)}] OK   {source_name_value} "
            f"({len(ids)} chunks, {token_count} tokens)",
            flush=True,
        )

    # Fetch a fresh collection handle before the final gate.  Chroma collection
    # objects retain metadata in memory, so reusing ``source`` could miss an
    # external revision change that happened while the migration was running.
    final_source = chromadb.PersistentClient(path=str(source_db_path)).get_collection(source_name)
    final_source_revision, final_source_chunk_count = require_ready_source(final_source)
    if (final_source_revision, final_source_chunk_count) != (source_revision, source_chunk_count):
        raise RuntimeError(
            "Source collection changed during migration; candidate remains in building state"
        )
    if target.count() != source_chunk_count:
        raise RuntimeError(f"Final chunk count mismatch: {target.count()} != {source_chunk_count}")
    final_rows = read_collection(target, include=["metadatas"])
    final_sources = {(metadata or {}).get("source") for metadata in final_rows["metadatas"]}
    if final_sources != set(sources):
        raise RuntimeError("Final document set mismatch")
    if any((metadata or {}).get("model") != model for metadata in final_rows["metadatas"]):
        raise RuntimeError("Target contains mixed embedding models")
    for source_name_value in sources:
        if not source_is_identical(final_source, target, source_name_value, model):
            raise RuntimeError(
                f"Final text/provenance mismatch for {source_name_value}; "
                "candidate remains in building state"
            )

    update_collection_state(
        target,
        embedding_model=model,
        migration_completed_documents=len(sources),
        migration_completed_at=datetime.now(timezone.utc).isoformat(),
        ragdoc_repairing=False,
    )
    bump_revision(target, "ready")
    print(
        f"COMPLETE documents={len(sources)} chunks={target.count()} model={model}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate a Ragdoc collection to a new embedding model")
    parser.add_argument("--source-db", type=Path, default=Path("chroma_db_new"))
    parser.add_argument("--target-db", type=Path, default=Path("chroma_db_context4_candidate"))
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    lock_path = args.source_db.resolve().parent / ".indexing.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another Ragdoc indexer is already running") from error
        migrate(
            args.source_db.resolve(),
            args.target_db.resolve(),
            args.source,
            args.target,
            args.model,
        )


if __name__ == "__main__":
    main()
