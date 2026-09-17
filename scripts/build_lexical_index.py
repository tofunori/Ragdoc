#!/usr/bin/env python3
"""Build the revision-pinned SQLite FTS sidecar for the active Ragdoc corpus."""

from pathlib import Path
import fcntl
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chromadb

from src.chroma_connection import open_chroma_client
from src.config import CHROMA_DB_PATH, COLLECTION_NAME, LEXICAL_INDEX_PATH
from src.lexical_index import PersistentLexicalIndex


def main() -> None:
    lock_path = CHROMA_DB_PATH.parent / ".indexing.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        client, _ = open_chroma_client(chromadb, CHROMA_DB_PATH)
        collection = client.get_collection(COLLECTION_NAME)
        status = PersistentLexicalIndex(LEXICAL_INDEX_PATH).rebuild(collection)
    print(
        f"LEXICAL_READY chunks={status['chunks']} revision={status['revision']} "
        f"path={status['path']}"
    )


if __name__ == "__main__":
    main()
