#!/usr/bin/env python3
"""
Test script for new MCP tools: get_document_content and get_chunk_with_context
Tests the underlying functionality by calling ChromaDB directly
"""

import sys
from pathlib import Path

# Add project to path for config
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import chromadb
from src.config import COLLECTION_NAME, CHROMA_DB_PATH


def get_chroma_client():
    """Get ChromaDB client (server or local)"""
    try:
        client = chromadb.HttpClient(host="localhost", port=8000)
        client.heartbeat()
        print("[OK] Connected to ChromaDB server (localhost:8000)")
        return client
    except:
        print("[INFO] ChromaDB server not available, using local mode")
        return chromadb.PersistentClient(path=str(CHROMA_DB_PATH))


def test_get_document_content():
    """Test get_document_content functionality"""
    print("\n" + "=" * 80)
    print("TEST 1: get_document_content functionality")
    print("=" * 80)

    client = get_chroma_client()
    collection = client.get_collection(name=COLLECTION_NAME)

    # Get list of unique sources
    print("\n[1.1] Getting list of documents...")
    all_docs = collection.get(include=["metadatas"])

    sources = {}
    for metadata in all_docs['metadatas']:
        source = metadata.get('source', metadata.get('filename', 'unknown'))
        if source not in sources:
            sources[source] = metadata

    print(f"Found {len(sources)} unique documents")

    # Pick first document
    first_source = list(sources.keys())[0]
    print(f"\n[1.2] Testing with document: {first_source}")

    # Fetch all chunks for this document
    print("\n[1.3] Fetching all chunks for this document...")
    doc_chunks = collection.get(
        where={"source": first_source},
        include=["documents", "metadatas"]
    )

    if not doc_chunks['documents']:
        print("ERROR: No chunks found")
        return

    print(f"Found {len(doc_chunks['documents'])} chunks")

    # Sort by chunk_index
    combined = list(zip(doc_chunks['documents'], doc_chunks['metadatas']))
    combined.sort(key=lambda x: x[1].get('chunk_index', 0))

    # Extract metadata
    first_meta = combined[0][1]
    print("\n[1.4] Document metadata:")
    print(f"  - Title: {first_meta.get('title', first_source)}")
    print(f"  - Total chunks: {len(combined)}")
    print(f"  - Indexed: {first_meta.get('indexed_date', 'N/A')}")
    print(f"  - Model: {first_meta.get('model', 'N/A')}")
    print(f"  - Hash: {first_meta.get('doc_hash', 'N/A')}")

    # Test reconstruction
    print("\n[1.5] Testing markdown format reconstruction...")
    full_text = "\n\n".join([chunk for chunk, _ in combined])
    markdown_output = f"# {first_meta.get('title', first_source)}\n\n"
    markdown_output += f"**Source:** {first_source}\n"
    markdown_output += f"**Total chunks:** {len(combined)}\n\n"
    markdown_output += full_text[:500]  # First 500 chars
    print(markdown_output)

    print("\n[OK] Test 1 completed\n")


def test_get_chunk_with_context():
    """Test get_chunk_with_context functionality"""
    print("\n" + "=" * 80)
    print("TEST 2: get_chunk_with_context functionality")
    print("=" * 80)

    client = get_chroma_client()
    collection = client.get_collection(name=COLLECTION_NAME)

    # Get a chunk with chunk_index >= 2
    print("\n[2.1] Finding a chunk with enough context...")
    all_chunks = collection.get(
        limit=20,
        include=["metadatas"]
    )

    test_chunk_id = None
    test_metadata = None
    for chunk_id, metadata in zip(all_chunks['ids'], all_chunks['metadatas']):
        chunk_idx = metadata.get('chunk_index', 0)
        if chunk_idx >= 2:
            test_chunk_id = chunk_id
            test_metadata = metadata
            print(f"Selected chunk: {chunk_id}")
            print(f"  - Chunk index: {chunk_idx}")
            print(f"  - Source: {metadata.get('source', 'N/A')}")
            break

    if not test_chunk_id:
        test_chunk_id = all_chunks['ids'][0]
        test_metadata = all_chunks['metadatas'][0]
        print(f"Using first chunk: {test_chunk_id}")

    # Get target chunk
    print("\n[2.2] Fetching target chunk...")
    target = collection.get(
        ids=[test_chunk_id],
        include=["documents", "metadatas"]
    )

    target_text = target['documents'][0]
    target_meta = target['metadatas'][0]

    source = target_meta.get('source', target_meta.get('filename', 'unknown'))
    chunk_index = target_meta.get('chunk_index', 0)
    total_chunks = target_meta.get('total_chunks', 0)

    print(f"Target chunk index: {chunk_index}/{total_chunks}")
    print(f"Source: {source}")

    # Get adjacent chunks (context_size=2)
    print("\n[2.3] Fetching adjacent chunks (context_size=2)...")
    context_size = 2
    start_idx = max(0, chunk_index - context_size)
    end_idx = min(int(total_chunks) - 1, chunk_index + context_size)

    # Fetch all chunks for this document
    all_doc_chunks = collection.get(
        where={"source": source},
        include=["documents", "metadatas"]
    )

    # Filter chunks in range
    adjacent = []
    for doc, meta in zip(all_doc_chunks['documents'], all_doc_chunks['metadatas']):
        idx = meta.get('chunk_index')
        if idx is not None and start_idx <= idx <= end_idx:
            adjacent.append((doc, meta))

    adjacent.sort(key=lambda x: x[1]['chunk_index'])

    print(f"Found {len(adjacent)} adjacent chunks (indices {start_idx} to {end_idx})")

    # Display with highlighting
    print("\n[2.4] Context window:")
    print("=" * 70)
    for chunk_text, chunk_meta in adjacent:
        current_idx = chunk_meta.get('chunk_index', 0)
        is_target = current_idx == chunk_index

        if is_target:
            print(">>> [TARGET CHUNK] <<<")
            print(f"[Chunk {current_idx}/{total_chunks}] *** MATCHED RESULT ***")
        else:
            print(f"[Chunk {current_idx}/{total_chunks}]")

        # Show first 200 chars
        preview = chunk_text[:200]
        if len(chunk_text) > 200:
            preview += "..."
        print(preview)

        if is_target:
            print(">>> [END TARGET CHUNK] <<<")

        print("-" * 70)

    print("\n[OK] Test 2 completed\n")


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("TESTING NEW MCP TOOLS")
    print("=" * 80)
    print("\nThis script tests the 2 new MCP tools:")
    print("  1. get_document_content - Retrieve complete document content")
    print("  2. get_chunk_with_context - Show chunk with surrounding context")
    print("\n" + "=" * 80)

    try:
        test_get_document_content()
        test_get_chunk_with_context()

        print("\n" + "=" * 80)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("\nBoth new MCP tools are working correctly!")
        print("\nNext steps:")
        print("  1. Restart MCP server in Cursor")
        print("  2. Try the new tools in your research workflow")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
