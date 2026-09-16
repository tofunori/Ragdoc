"""Canonical Ragdoc MCP entry point shared by every client."""

import os
from pathlib import Path

import chromadb
from chromadb.config import Settings


ROOT = Path(__file__).resolve().parent
CANONICAL_DB = (ROOT / "chroma_db_new").resolve()
CANONICAL_LIBRARY = (ROOT / "ragdoc_library" / "ragdoc_contextualized_v1").resolve()
COLLECTION = "ragdoc_contextualized_v1"

configured_db = Path(os.environ.get("CHROMA_DB_PATH", CANONICAL_DB)).resolve()
if configured_db != CANONICAL_DB:
    raise RuntimeError(
        f"Refusing non-canonical Ragdoc database: {configured_db}; expected {CANONICAL_DB}"
    )

os.environ["CHROMA_DB_PATH"] = str(CANONICAL_DB)
os.environ["RAGDOC_LIBRARY_DIR"] = str(CANONICAL_LIBRARY)
os.environ["COLLECTION_NAME"] = COLLECTION

# Import after pinning the environment because src.config reads it at import time.
from src import server  # noqa: E402
from src.config import EMBEDDING_MODEL  # noqa: E402


server.chroma_client = chromadb.PersistentClient(
    path=str(CANONICAL_DB),
    settings=Settings(anonymized_telemetry=False),
)
collection = server.chroma_client.get_collection(COLLECTION)
collection_model = (collection.metadata or {}).get("embedding_model")
if collection_model != EMBEDDING_MODEL:
    raise RuntimeError(
        f"Embedding model mismatch: collection={collection_model!r}, "
        f"configured={EMBEDDING_MODEL!r}"
    )

transport = os.environ.get("RAGDOC_TRANSPORT", "stdio").lower()
if transport == "sse":
    server.mcp.run(
        transport="sse",
        host="0.0.0.0",
        port=int(os.environ.get("RAGDOC_PORT", "8484")),
        show_banner=False,
    )
elif transport in {"http", "streamable-http"}:
    server.mcp.run(
        transport="http",
        host="0.0.0.0",
        port=int(os.environ.get("RAGDOC_PORT", "8484")),
        path="/mcp",
        show_banner=False,
    )
else:
    server.mcp.run(show_banner=False)
