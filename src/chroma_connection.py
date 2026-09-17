"""One connection policy shared by indexing and MCP reads."""
import os
from pathlib import Path


def open_chroma_client(chromadb_module, db_path: Path, mode: str | None = None):
    resolved = (mode or os.getenv("RAGDOC_CHROMA_MODE", "auto")).strip().lower()
    if resolved not in {"auto", "persistent", "http"}:
        raise RuntimeError("RAGDOC_CHROMA_MODE must be 'auto', 'persistent' or 'http'")
    if resolved == "persistent":
        return chromadb_module.PersistentClient(path=str(db_path)), "persistent-forced"
    try:
        client = chromadb_module.HttpClient(
            host=os.getenv("RAGDOC_CHROMA_HOST", "localhost"),
            port=int(os.getenv("RAGDOC_CHROMA_PORT", "8000")),
        )
        client.heartbeat()
        return client, "http"
    except Exception:
        if resolved == "http":
            raise  # An explicit store must never silently change after an outage.
        return chromadb_module.PersistentClient(path=str(db_path)), "persistent-fallback"
