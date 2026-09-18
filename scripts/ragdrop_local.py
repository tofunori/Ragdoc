#!/usr/bin/env python3
"""Local desktop entry point. Data paths are explicit; stdout is JSON or MCP only."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import re
import shutil
import uuid
import subprocess
import sys

ENGINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ENGINE))
COLLECTION = "ragdoc_contextualized_v1"
MODEL = "voyage-context-4"


def configure(root: Path) -> None:
    if not root.is_absolute():
        raise ValueError("Choose an absolute library folder.")
    # Never inherit a server, another library or a developer .env from the caller.
    for key in list(os.environ):
        if key.startswith(("RAGDOC_", "CHROMA_")) or key in {"COLLECTION_NAME", "VOYAGE_API_KEY", "COHERE_API_KEY"}:
            os.environ.pop(key, None)
    os.environ.update({
        "PYTHON_DOTENV_DISABLED": "1", "ANONYMIZED_TELEMETRY": "False",
        "RAGDOC_CHROMA_MODE": "persistent", "COLLECTION_NAME": COLLECTION,
        "RAGDOC_EMBEDDING_MODEL": MODEL,
        "RAGDOC_MARKDOWN_DIR": str(root / "articles_markdown"),
        "CHROMA_DB_PATH": str(root / "chroma_db_new"),
        "RAGDOC_LIBRARY_DIR": str(root / "ragdoc_library" / COLLECTION),
        "RAGDOC_ARTIFACTS_DIR": str(root / "ragdoc_artifacts"),
        "NLTK_DATA": str(ENGINE / "nltk_data"),
        "HF_HOME": str(ENGINE / "model_cache"),
        "TOKENIZERS_PARALLELISM": "false",
    })


def load_voyage_key() -> None:
    if sys.platform != "darwin":
        raise RuntimeError("The desktop key store requires macOS.")
    result = subprocess.run([
        "/usr/bin/security", "find-generic-password", "-s",
        "com.tofunori.ragdrop.voyage", "-a", "api-key", "-w"
    ], capture_output=True, text=True, timeout=30)
    if result.returncode == 0 and result.stdout.strip():
        os.environ["VOYAGE_API_KEY"] = result.stdout.strip()


def collection(root: Path, create: bool = False):
    import chromadb
    from src.config import COLLECTION_CONTEXTUALIZED_METADATA
    if not create and not (root / "chroma_db_new" / "chroma.sqlite3").is_file():
        raise RuntimeError("Prepare this library in Ragdrop first.")
    client = chromadb.PersistentClient(path=str(root / "chroma_db_new"))
    if create:
        result = client.get_or_create_collection(COLLECTION, metadata=COLLECTION_CONTEXTUALIZED_METADATA)
    else:
        result = client.get_collection(COLLECTION)
    if (result.metadata or {}).get("embedding_model") != MODEL:
        raise RuntimeError("This library uses another embedding model. Choose a new folder or migrate it separately.")
    return result


def documents(root: Path) -> list[dict]:
    from src.chroma_reads import read_collection
    col = collection(root)
    state = col.metadata or {}
    if state.get("ragdoc_write_state", "ready") != "ready" or state.get("ragdoc_repairing"):
        raise RuntimeError("Indexing is incomplete. Finish or repair the index before reading it.")
    result = read_collection(col, include=["metadatas"])
    rows = {}
    for meta in result.get("metadatas") or []:
        meta = meta or {}
        source = meta.get("source")
        if not source:
            continue
        row = rows.setdefault(source, {"source": source, "title": meta.get("title") or source,
            "chunks": 0, "indexedDate": meta.get("indexed_date"), "doi": meta.get("doi")})
        row["chunks"] += 1
    for source, row in rows.items():
        if Path(source).name != source:
            continue
        sidecar = root / "articles_markdown" / Path(source).with_suffix(".metadata.json")
        try:
            identity = json.loads(sidecar.read_text())
            row["pdfFingerprint"] = identity.get("pdf_sha256")
            row["zoteroAttachmentKey"] = identity.get("zotero_attachment_key")
        except (OSError, ValueError):
            pass
    final_state = collection(root).metadata or {}
    if (final_state.get("ragdoc_revision") != state.get("ragdoc_revision")
            or final_state.get("ragdoc_write_state", "ready") != "ready"
            or final_state.get("ragdoc_repairing")):
        raise RuntimeError("The library changed while reading. Please retry.")
    return sorted(rows.values(), key=lambda row: (row["indexedDate"] or "", row["source"]), reverse=True)


def validate_sources(sources: list[str]) -> None:
    if not sources or any(not re.fullmatch(r"[A-Za-z0-9_.-]+\.md", s) or Path(s).name != s for s in sources):
        raise ValueError("Invalid document filename.")


def backup_library(root: Path) -> Path:
    # Called under the indexer's exclusive writer lock, before any index mutation.
    target = root / "Backups" / ("index-before-repair-" + uuid.uuid4().hex)
    staging = target.with_name("." + target.name)
    staging.mkdir(parents=True)
    try:
        for name in ["articles_markdown", "chroma_db_new", "ragdoc_library", "ragdoc_artifacts"]:
            source = root / name
            if source.exists():
                shutil.copytree(source, staging / name)
        (staging / "README.txt").write_text("Snapshot before an explicitly requested index rebuild. Original PDF files are not moved.\n")
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def dispatch(args):
    root = args.root
    if args.action == "prepare":
        for name in ["articles_markdown", "ragdoc_artifacts"]:
            (root / name).mkdir(parents=True, exist_ok=True)
        collection(root, create=True)
        from src.library import Library
        Library(root / "ragdoc_library" / COLLECTION)
        return {"ready": True}
    if args.action == "documents":
        return documents(root)
    if args.action == "count":
        validate_sources(args.source)
        from src.chroma_reads import read_collection
        col = collection(root)
        state = col.metadata or {}
        if state.get("ragdoc_write_state", "ready") != "ready" or state.get("ragdoc_repairing"):
            raise RuntimeError("The index is not ready for verification.")
        count = len(read_collection(col, where={"source": args.source[0]}, include=[])["ids"])
        after = collection(root).metadata or {}
        if after.get("ragdoc_revision") != state.get("ragdoc_revision") or after.get("ragdoc_write_state", "ready") != "ready" or after.get("ragdoc_repairing"):
            raise RuntimeError("The index changed during verification. Please retry.")
        return {"count": count}
    if args.action == "status":
        col = collection(root)
        state = col.metadata or {}
        repair = state.get("ragdoc_write_state", "ready") != "ready" or bool(state.get("ragdoc_repairing"))
        rows = [] if repair else documents(root)
        from src.runtime_status import runtime_status
        runtime = runtime_status()
        return {"documents": len(rows), "chunks": sum(r["chunks"] for r in rows),
            "ready": not runtime["issues"] and not repair, "needsRepair": repair, "issues": runtime["issues"]}
    if args.action in {"index", "repair"}:
        if args.action == "index":
            validate_sources(args.source)
        elif not args.confirm_rebuild:
            raise ValueError("Confirm the rebuild in Ragdrop. Rebuilding uses Voyage and may incur charges.")
        load_voyage_key()
        if not os.environ.get("VOYAGE_API_KEY"):
            raise RuntimeError("Add your Voyage AI key in Ragdrop Settings before indexing.")
        from scripts.index_incremental import index_incremental
        from src.artifacts import ArtifactIndex
        backup = []
        if args.action == "repair":
            stats = index_incremental(force_reindex=True, before_write=lambda: backup.append(str(backup_library(root))))
        else:
            stats = index_incremental(sources=args.source)
        if stats["errors"]:
            raise RuntimeError("Some documents could not be indexed. Review the error and retry.")
        ArtifactIndex(root / "ragdoc_artifacts").index(set(args.source) if args.source else None)
        if backup: stats["backup"] = backup[0]
        return stats
    if args.action == "mcp":
        collection(root)
        load_voyage_key()
        # Imports log on stderr; only the MCP protocol owns stdout.
        from src.server import mcp
        mcp.run(show_banner=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("action", choices=["prepare", "documents", "count", "status", "index", "repair", "mcp"])
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--confirm-rebuild", action="store_true")
    args = parser.parse_args()
    configure(args.root)
    try:
        if args.action == "mcp":
            dispatch(args)
        else:
            with contextlib.redirect_stdout(sys.stderr):
                result = dispatch(args)
            print(json.dumps(result, ensure_ascii=False))
        return 0
    except Exception as error:
        # Key values are never part of our own messages.
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
