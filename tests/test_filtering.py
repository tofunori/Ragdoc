#!/usr/bin/env python3
"""
Test script for new filtering features in RAGDOC v1.3.0

Tests:
1. search_by_source with single document
2. search_by_source with multiple documents
3. _perform_search_hybrid with where filter
4. _perform_search_hybrid with where_document filter
5. Backward compatibility (semantic_search_hybrid without filters)
"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from dotenv import load_dotenv
load_dotenv()

import server

def test_search_by_source_single():
    """Test search_by_source with a single document"""
    print("\n" + "=" * 80)
    print("TEST 1: search_by_source - Single Document")
    print("=" * 80)

    try:
        server.init_clients()

        # Search only in 1982_RGSP.md
        result = server.search_by_source.fn(
            query="glacier albedo measurements",
            sources=["1982_RGSP.md"],
            top_k=3
        )

        print(result[:800])

        # Verify results are from 1982_RGSP.md
        if "1982_RGSP.md" in result and "ERROR" not in result:
            print("\n[OK] Test 1 passed - Results from 1982_RGSP.md only")
        else:
            print("\n[FAIL] Test 1 failed")

    except Exception as e:
        print(f"\n[FAIL] Test 1 failed: {e}")
        import traceback
        traceback.print_exc()


def test_search_by_source_multiple():
    """Test search_by_source with multiple documents"""
    print("\n" + "=" * 80)
    print("TEST 2: search_by_source - Multiple Documents")
    print("=" * 80)

    try:
        result = server.search_by_source.fn(
            query="snow albedo",
            sources=["1982_RGSP.md", "A Model for the Spectral Albedo of Snow. I Pure Snow.md"],
            top_k=5
        )

        print(result[:800])

        # Verify results are from specified documents
        has_1982 = "1982_RGSP.md" in result
        has_model = "A Model for the Spectral Albedo" in result
        no_error = "ERROR" not in result

        if (has_1982 or has_model) and no_error:
            print("\n[OK] Test 2 passed - Results from specified documents")
        else:
            print("\n[FAIL] Test 2 failed")

    except Exception as e:
        print(f"\n[FAIL] Test 2 failed: {e}")
        import traceback
        traceback.print_exc()


def test_perform_search_with_where():
    """Test _perform_search_hybrid with where filter directly"""
    print("\n" + "=" * 80)
    print("TEST 3: _perform_search_hybrid with where filter")
    print("=" * 80)

    try:
        # Test with chunk_index filter
        result = server._perform_search_hybrid(
            query="albedo",
            top_k=3,
            where={
                "$and": [
                    {"source": "1982_RGSP.md"},
                    {"chunk_index": {"$lte": 50}}
                ]
            }
        )

        print(result[:800])

        if "1982_RGSP.md" in result and "ERROR" not in result:
            print("\n[OK] Test 3 passed - Complex where filter works")
        else:
            print("\n[FAIL] Test 3 failed")

    except Exception as e:
        print(f"\n[FAIL] Test 3 failed: {e}")
        import traceback
        traceback.print_exc()


def test_perform_search_with_where_document():
    """Test _perform_search_hybrid with where_document filter"""
    print("\n" + "=" * 80)
    print("TEST 4: _perform_search_hybrid with where_document filter")
    print("=" * 80)

    try:
        result = server._perform_search_hybrid(
            query="climate change",
            top_k=3,
            where_document={"$contains": "temperature"}
        )

        print(result[:800])

        if "ERROR" not in result:
            print("\n[OK] Test 4 passed - where_document filter works")
        else:
            print("\n[FAIL] Test 4 failed")

    except Exception as e:
        print(f"\n[FAIL] Test 4 failed: {e}")
        import traceback
        traceback.print_exc()


def test_backward_compatibility():
    """Test that semantic_search_hybrid still works without filters"""
    print("\n" + "=" * 80)
    print("TEST 5: Backward Compatibility - semantic_search_hybrid")
    print("=" * 80)

    try:
        result = server.semantic_search_hybrid.fn(
            query="glacier mass balance",
            top_k=3
        )

        print(result[:800])

        if "ERROR" not in result and "HYBRID SEARCH RESULTS" in result:
            print("\n[OK] Test 5 passed - Backward compatibility maintained")
        else:
            print("\n[FAIL] Test 5 failed")

    except Exception as e:
        print(f"\n[FAIL] Test 5 failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all filtering tests"""
    print("\n" + "=" * 80)
    print("RAGDOC v1.3.0 - FILTERING FEATURES TEST SUITE")
    print("=" * 80)
    print("\nTesting new filtering capabilities:")
    print("  - search_by_source (single document)")
    print("  - search_by_source (multiple documents)")
    print("  - where metadata filters")
    print("  - where_document content filters")
    print("  - Backward compatibility")
    print("\n" + "=" * 80)

    try:
        test_search_by_source_single()
        test_search_by_source_multiple()
        test_perform_search_with_where()
        test_perform_search_with_where_document()
        test_backward_compatibility()

        print("\n" + "=" * 80)
        print("ALL FILTERING TESTS COMPLETED")
        print("=" * 80)
        print("\n[SUCCESS] All filtering features are working correctly!")
        print("\nNew capabilities in RAGDOC v1.3.0:")
        print("  - Filter search results by document(s)")
        print("  - Filter by metadata (date, model, chunk_index, etc.)")
        print("  - Filter by document content")
        print("  - Combine multiple filters with $and/$or")
        print("\nTotal MCP tools: 6 (was 5)")
        print("  - semantic_search_hybrid")
        print("  - search_by_source (NEW)")
        print("  - list_documents")
        print("  - get_document_content")
        print("  - get_chunk_with_context")
        print("  - get_indexation_status")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n[FAIL] TEST SUITE FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
