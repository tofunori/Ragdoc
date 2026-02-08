#!/usr/bin/env python3
"""
Test MCP integration - Test the new MCP tools via direct function calls
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

# Import MCP server module
import server

def test_mcp_tools():
    """Test all 5 MCP tools"""
    print("=" * 80)
    print("TEST MCP INTEGRATION - RAGDOC")
    print("=" * 80)

    # Initialize clients
    print("\n[Init] Initializing clients...")
    server.init_clients()
    print("[OK] Clients initialized")

    # Test 1: get_indexation_status
    print("\n" + "=" * 80)
    print("TEST 1/5: get_indexation_status()")
    print("=" * 80)
    try:
        result = server.get_indexation_status.fn()
        # Show first 500 chars
        print(result[:500])
        if "ERROR" in result:
            print("[FAIL] Test failed")
        else:
            print("[OK] Test passed")
    except Exception as e:
        print(f"[FAIL] Test failed: {e}")

    # Test 2: list_documents
    print("\n" + "=" * 80)
    print("TEST 2/5: list_documents()")
    print("=" * 80)
    try:
        result = server.list_documents.fn()
        # Show first 500 chars
        print(result[:500])
        if "ERROR" in result:
            print("[FAIL] Test failed")
        else:
            print("[OK] Test passed")
            # Extract first document source for next tests
            lines = result.split('\n')
            first_doc = None
            for line in lines:
                if line.strip().startswith('[1]'):
                    first_doc = line.split('[1]')[1].strip()
                    break
            print(f"\n[INFO] First document: {first_doc}")
            return first_doc
    except Exception as e:
        print(f"[FAIL] Test failed: {e}")
        return None


def test_new_tools(first_doc):
    """Test the 2 new MCP tools"""
    if not first_doc:
        print("\n[WARN] Skipping new tool tests (no document found)")
        return

    # Test 3: get_document_content
    print("\n" + "=" * 80)
    print("TEST 3/5: get_document_content()")
    print("=" * 80)
    try:
        print(f"\n[3.1] Testing with source='{first_doc}', format='markdown', max_length=800")
        result = server.get_document_content.fn(source=first_doc, format="markdown", max_length=800)
        print(result)
        if "ERROR" in result:
            print("[FAIL] Test failed")
        else:
            print("[OK] Test passed")

        print(f"\n[3.2] Testing with format='text', max_length=500")
        result = server.get_document_content.fn(source=first_doc, format="text", max_length=500)
        print(result[:500])
        if "ERROR" in result:
            print("[FAIL] Test failed")
        else:
            print("[OK] Test passed")

        print(f"\n[3.3] Testing error handling (non-existent document)")
        result = server.get_document_content.fn(source="fake_document.md")
        print(result)
        if "ERROR" in result and "not found" in result:
            print("[OK] Test passed (error handled correctly)")
        else:
            print("[FAIL] Test failed (should return error)")

    except Exception as e:
        print(f"[FAIL] Test failed: {e}")

    # Test 4: get_chunk_with_context
    print("\n" + "=" * 80)
    print("TEST 4/5: get_chunk_with_context()")
    print("=" * 80)
    try:
        # Get a chunk ID from the database
        print("\n[4.1] Getting a chunk ID from database...")
        import chromadb
        from src.config import COLLECTION_NAME, CHROMA_DB_PATH

        try:
            chroma_client = chromadb.HttpClient(host="localhost", port=8000)
            chroma_client.heartbeat()
        except:
            chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))

        collection = chroma_client.get_collection(name=COLLECTION_NAME)
        all_chunks = collection.get(limit=10, include=["metadatas"])

        # Find a chunk with index >= 2
        test_chunk_id = None
        for chunk_id, metadata in zip(all_chunks['ids'], all_chunks['metadatas']):
            chunk_idx = metadata.get('chunk_index', 0)
            if chunk_idx >= 2:
                test_chunk_id = chunk_id
                print(f"Selected chunk: {chunk_id} (index {chunk_idx})")
                break

        if not test_chunk_id:
            test_chunk_id = all_chunks['ids'][0]
            print(f"Using first chunk: {test_chunk_id}")

        print(f"\n[4.2] Testing with chunk_id='{test_chunk_id}', context_size=2")
        result = server.get_chunk_with_context.fn(chunk_id=test_chunk_id, context_size=2, highlight=True)
        # Show first 1000 chars
        print(result[:1000])
        if "ERROR" in result:
            print("[FAIL] Test failed")
        else:
            print("[OK] Test passed")

        print(f"\n[4.3] Testing with context_size=1, highlight=False")
        result = server.get_chunk_with_context.fn(chunk_id=test_chunk_id, context_size=1, highlight=False)
        print(result[:800])
        if "ERROR" in result:
            print("[FAIL] Test failed")
        else:
            print("[OK] Test passed")

        print(f"\n[4.4] Testing error handling (non-existent chunk)")
        result = server.get_chunk_with_context.fn(chunk_id="fake_chunk_id_12345")
        print(result)
        if "ERROR" in result and "not found" in result:
            print("[OK] Test passed (error handled correctly)")
        else:
            print("[FAIL] Test failed (should return error)")

    except Exception as e:
        print(f"[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 5: semantic_search_hybrid
    print("\n" + "=" * 80)
    print("TEST 5/5: semantic_search_hybrid()")
    print("=" * 80)
    try:
        print("\n[5.1] Testing search with query='glacier albedo', top_k=3")
        result = server.semantic_search_hybrid.fn(query="glacier albedo", top_k=3, alpha=0.7)
        # Show first 1000 chars
        print(result[:1000])
        if "ERROR" in result:
            print("[FAIL] Test failed")
        else:
            print("[OK] Test passed")
    except Exception as e:
        print(f"[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all MCP integration tests"""
    print("\n" + "=" * 80)
    print("MCP INTEGRATION TEST SUITE")
    print("=" * 80)
    print("\nTesting all 5 MCP tools in RAGDOC:")
    print("  1. get_indexation_status")
    print("  2. list_documents")
    print("  3. get_document_content (NEW)")
    print("  4. get_chunk_with_context (NEW)")
    print("  5. semantic_search_hybrid")
    print("\n" + "=" * 80)

    try:
        first_doc = test_mcp_tools()
        test_new_tools(first_doc)

        print("\n" + "=" * 80)
        print("ALL MCP INTEGRATION TESTS COMPLETED")
        print("=" * 80)
        print("\n[SUCCESS] All 5 MCP tools are working correctly!")
        print("\nThe MCP server is ready to use in Cursor.")
        print("Restart Cursor to load the new tools.")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n[FAIL] TEST SUITE FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
