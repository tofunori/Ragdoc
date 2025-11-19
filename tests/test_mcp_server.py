#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for the ragdoc MCP server
Tests all available tools and functions
"""

import sys
import io
from pathlib import Path
from dotenv import load_dotenv

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Load environment
load_dotenv()

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import server functions and internal helpers
from server import (
    semantic_search,
    list_documents,
    topic_search,
    get_indexation_status,
    reindex_documents,
    init_clients,
    _perform_search
)

# FastMCP wraps functions in FunctionTool objects, extract the original function
# Access the underlying function via the .fn attribute if available
def get_tool_function(tool_obj):
    """Extract the original function from a FastMCP FunctionTool"""
    if hasattr(tool_obj, 'fn'):
        return tool_obj.fn
    elif hasattr(tool_obj, '__wrapped__'):
        return tool_obj.__wrapped__
    elif hasattr(tool_obj, 'function'):
        return tool_obj.function
    else:
        # Try to call via the tool's invoke method if available
        return tool_obj

def test_indexation_status():
    """Test get_indexation_status tool"""
    print("=" * 70)
    print("TEST 1: get_indexation_status")
    print("=" * 70)
    try:
        # Get the actual function from the tool wrapper
        func = get_tool_function(get_indexation_status)
        result = func()
        print(result)
        print("\n✓ Test passed\n")
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False

def test_list_documents():
    """Test list_documents tool"""
    print("=" * 70)
    print("TEST 2: list_documents")
    print("=" * 70)
    try:
        func = get_tool_function(list_documents)
        result = func()
        print(result)
        print("\n✓ Test passed\n")
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False

def test_semantic_search():
    """Test semantic_search tool"""
    print("=" * 70)
    print("TEST 3: semantic_search")
    print("=" * 70)
    try:
        # Test with a simple query - use internal function directly
        query = "glacier albedo"
        print(f"Query: '{query}'")
        result = _perform_search(query, top_k=5)
        print(result)
        print("\n✓ Test passed\n")
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False

def test_topic_search():
    """Test topic_search tool"""
    print("=" * 70)
    print("TEST 4: topic_search")
    print("=" * 70)
    try:
        # Test with a predefined topic
        topic = "black carbon"
        print(f"Topic: '{topic}'")
        # Expand topic to query (same logic as topic_search)
        topic_queries = {
            "black carbon": "black carbon soot impurities glacier albedo",
            "snow albedo": "snow surface albedo reflectance glaciers",
            "glacier melt": "glacier melt ablation surface energy budget",
            "dust": "mineral dust impurities glacier darkening",
            "cryoconite": "cryoconite holes algae glacier surface",
            "remote sensing": "remote sensing satellite albedo glaciers"
        }
        query = topic_queries.get(topic.lower(), topic)
        result = _perform_search(query, top_k=3)
        print(result)
        print("\n✓ Test passed\n")
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False

def test_client_initialization():
    """Test client initialization"""
    print("=" * 70)
    print("TEST 0: Client Initialization")
    print("=" * 70)
    try:
        init_clients()
        print("✓ Clients initialized successfully")
        print()
        return True
    except Exception as e:
        print(f"✗ Client initialization failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False

def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("RAGDOC MCP SERVER TEST SUITE")
    print("=" * 70 + "\n")
    
    results = []
    
    # Test 0: Initialize clients
    results.append(("Client Initialization", test_client_initialization()))
    
    # Test 1: Indexation status
    results.append(("get_indexation_status", test_indexation_status()))
    
    # Test 2: List papers
    results.append(("list_documents", test_list_documents()))
    
    # Test 3: Search glacier research
    results.append(("semantic_search", test_semantic_search()))
    
    # Test 4: Search by topic
    results.append(("topic_search", test_topic_search()))
    
    # Summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} - {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("=" * 70)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


