"""Prepare-first replacement with rollback for recoverable index write errors.

Chroma has no cross-request transaction here. A rollback failure is explicit;
process termination during writes requires repair from the canonical snapshots.
"""

import uuid


class IndexRepairRequired(RuntimeError):
    """A failed rollback must stop the writer, not continue with the next paper."""


def bump_revision(collection, state="ready"):
    # Index writers must hold the shared indexer lock throughout a run.
    metadata = dict(collection.metadata or {})
    metadata["ragdoc_revision"] = uuid.uuid4().hex
    metadata["ragdoc_write_state"] = state
    collection.modify(metadata=metadata)


def _batches(data, size):
    for offset in range(0, len(data["ids"]), size):
        yield {key: value[offset:offset + size] for key, value in data.items()}


def replace_document(collection, source: str, data: dict, batch_size: int = 100):
    """Called only once all text, vectors and metadata have passed validation."""
    metadata = collection.metadata or {}
    if metadata.get("ragdoc_write_state", "ready") != "ready" and not metadata.get("ragdoc_repairing", False):
        raise IndexRepairRequired("Previous index write requires explicit repair before any further replacement")
    size = len(data["ids"])
    if not size or any(len(data[k]) != size for k in ("documents", "embeddings", "metadatas")):
        raise ValueError("Replacement must contain matching nonempty vectors, texts and metadata")
    if len(set(data["ids"])) != size:
        raise ValueError("Duplicate replacement chunk IDs")
    old = collection.get(where={"source": source}, include=["documents", "metadatas", "embeddings"])
    backup = {key: list(old[key]) if old.get(key) is not None else []
              for key in ("ids", "documents", "embeddings", "metadatas")}
    if any(len(values) != len(backup["ids"]) for values in backup.values()):
        raise RuntimeError("Cannot back up the indexed document; refusing replacement")
    new_only = sorted(set(data["ids"]) - set(backup["ids"]))
    obsolete = sorted(set(backup["ids"]) - set(data["ids"]))
    bump_revision(collection, "writing")
    try:
        for batch in _batches(data, batch_size):
            collection.upsert(**batch)
        # An acknowledged write is checked before removing obsolete chunks.
        for batch in _batches(data, batch_size):
            written = collection.get(ids=batch["ids"], include=["documents", "metadatas"])
            actual = {i: (d, m) for i, d, m in zip(written["ids"], written["documents"], written["metadatas"])}
            if any(actual.get(i) != (d, m) for i, d, m in zip(batch["ids"], batch["documents"], batch["metadatas"])):
                raise RuntimeError("Index read-back did not match the prepared document")
        for offset in range(0, len(obsolete), batch_size):
            collection.delete(ids=obsolete[offset:offset + batch_size])
        bump_revision(collection)
    except Exception as error:
        try:
            for batch in _batches(backup, batch_size):
                collection.upsert(**batch)
            for offset in range(0, len(new_only), batch_size):
                collection.delete(ids=new_only[offset:offset + batch_size])
            bump_revision(collection)
        except Exception as rollback_error:
            raise IndexRepairRequired(f"Index replacement failed and rollback failed; repair required: {rollback_error}") from error
        raise
