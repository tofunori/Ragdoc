#!/usr/bin/env python3
"""
MCP Server for Glacier Research Search
HYBRID VERSION: BM25 + Voyage-3-Large embeddings + Cohere v3.5 reranking
"""

import os
import sys
import hashlib
import logging
from pathlib import Path
from datetime import datetime

from fastmcp import FastMCP
from dotenv import load_dotenv
import chromadb
import voyageai
import cohere

# Import hybrid retriever
sys.path.insert(0, str(Path(__file__).parent))
from hybrid_retriever import HybridRetriever

# Load environment
load_dotenv()

# Import indexing configuration
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from indexing_config import (
    MARKDOWN_DIR, COLLECTION_HYBRID_NAME, CHROMA_DB_HYBRID_PATH,
    CHUNK_SIZE, CHUNK_OVERLAP,
    DEFAULT_MODEL, LARGE_DOC_MODEL, LARGE_DOC_THRESHOLD,
    CONTEXT_WINDOW_SIZE
)

# Configuration
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Initialize MCP server
mcp = FastMCP("ragdoc-hybrid")

# Global clients (initialized on first use)
voyage_client = None
chroma_client = None
cohere_client = None
hybrid_retriever = None


def init_clients():
    """Initialize API clients with auto-detection of ChromaDB server"""
    global voyage_client, chroma_client, cohere_client, hybrid_retriever

    if not voyage_client:
        voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)

    if not chroma_client:
        # Try HttpClient (server mode) first, fallback to PersistentClient
        try:
            test_client = chromadb.HttpClient(host="localhost", port=8000)
            test_client.heartbeat()
            chroma_client = test_client
            logging.info("[OK] MCP: Connected to ChromaDB server (localhost:8000)")
        except Exception as e:
            logging.info("[INFO] MCP: ChromaDB server not available, using local mode")
            chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_HYBRID_PATH))

    if not cohere_client:
        cohere_client = cohere.ClientV2(api_key=COHERE_API_KEY)

    # Initialize hybrid retriever
    if not hybrid_retriever:
        collection = chroma_client.get_collection(name=COLLECTION_HYBRID_NAME)

        # Embedding function for hybrid retriever
        def voyage_embed(texts):
            result = voyage_client.embed(texts=texts, model="voyage-3-large")
            return result.embeddings

        hybrid_retriever = HybridRetriever(
            collection=collection,
            embedding_function=voyage_embed
        )
        logging.info("[OK] Hybrid retriever initialized (BM25 + Semantic)")


def _compute_doc_hash(content: str) -> str:
    """Compute MD5 hash of document content"""
    return hashlib.md5(content.encode()).hexdigest()


def _fetch_document_chunks(collection, source: str) -> dict:
    """
    Fetch all chunks for a given document source.

    Returns a normalized dict with 'documents' and 'metadatas' lists
    (may be empty if nothing is found).
    """
    for field in ("source", "filename"):
        try:
            results = collection.get(
                where={field: source},
                include=["documents", "metadatas"]
            )
        except Exception as fetch_error:
            logging.warning(
                "Failed to fetch chunks for %s=%s : %s",
                field,
                source,
                fetch_error
            )
            continue

        if results and results.get("documents"):
            return {
                "documents": results.get("documents", []),
                "metadatas": results.get("metadatas", [])
            }

    return {"documents": [], "metadatas": []}


def _get_document_cache_entry(collection, source: str, doc_cache: dict | None) -> dict:
    """Return cached document chunks, fetching and caching if needed."""
    if doc_cache is None:
        return _fetch_document_chunks(collection, source)

    if source not in doc_cache:
        doc_cache[source] = _fetch_document_chunks(collection, source)

    return doc_cache[source]


def _get_adjacent_chunks(collection, source: str, chunk_index: int, total_chunks: int,
                        window_size: int = CONTEXT_WINDOW_SIZE, doc_cache: dict | None = None) -> list:
    """
    Retrieve adjacent chunks around a target chunk.

    Args:
        collection: Chroma collection object
        source: Document source filename
        chunk_index: Index of the target chunk
        total_chunks: Total chunks in the document
        window_size: Number of chunks before/after target chunk

    Returns:
        List of adjacent chunks (including target) sorted by chunk_index
    """
    try:
        cached_entry = None
        if doc_cache is not None:
            cached_entry = doc_cache.get(source)

        if not cached_entry:
            cached_entry = _fetch_document_chunks(collection, source)
            if doc_cache is not None:
                doc_cache[source] = cached_entry

        documents = cached_entry.get('documents') or []
        metadatas = cached_entry.get('metadatas') or []

        effective_total = total_chunks
        if effective_total is None:
            effective_total = len(metadatas) if metadatas else 1

        # Calculate indices of adjacent chunks with validated totals
        start_idx = max(0, chunk_index - window_size)
        end_idx = min(effective_total - 1, chunk_index + window_size)

        if not documents or not metadatas or end_idx < start_idx:
            return []

        combined = []
        for document, metadata in zip(documents, metadatas):
            chunk_idx = metadata.get('chunk_index')
            if chunk_idx is None:
                continue
            if start_idx <= chunk_idx <= min(end_idx, effective_total - 1):
                combined.append((document, metadata))

        if not combined:
            return []

        combined.sort(key=lambda x: x[1]['chunk_index'])
        return combined

    except Exception as e:
        logging.warning(
            "Error retrieving adjacent chunks for %s: %s",
            source,
            e
        )
        return []


def _perform_search_hybrid(
    query: str,
    top_k: int = 10,
    alpha: float = 0.7,
    where: dict = None,
    where_document: dict = None
) -> str:
    """
    HYBRID search implementation
    Pipeline: BM25 + Semantic (RRF) → Cohere v3.5 reranking → Context window expansion

    Args:
        query: Search query
        top_k: Number of final results
        alpha: Weight for semantic (0.7 = 70% semantic, 30% BM25)
        where: Optional metadata filter (e.g., {"source": "doc.md"})
        where_document: Optional document content filter (e.g., {"$contains": "text"})
    """
    try:
        init_clients()
        collection = chroma_client.get_collection(name=COLLECTION_HYBRID_NAME)

        # 1. Hybrid retrieval (BM25 + Semantic with RRF)
        hybrid_results = hybrid_retriever.search(
            query=query,
            top_k=50,  # Get 50 candidates for reranking
            alpha=alpha,
            bm25_top_n=100,
            semantic_top_n=100,
            where=where,
            where_document=where_document
        )

        if not hybrid_results:
            return "No results found for your search."

        # 2. Prepare for reranking
        documents_for_rerank = [r['text'] for r in hybrid_results]
        metadatas = [r['metadata'] for r in hybrid_results]

        # 3. Rerank with Cohere v3.5
        rerank_results = cohere_client.rerank(
            model="rerank-v3.5",
            query=query,
            documents=documents_for_rerank,
            top_n=top_k
        )

        # 4. Format results with context window expansion
        output = f"HYBRID SEARCH RESULTS: {query}\n"
        output += "=" * 70 + "\n\n"

        doc_cache = {}

        for i, result in enumerate(rerank_results.results, 1):
            idx = result.index
            score = result.relevance_score
            metadata = metadatas[idx]
            hybrid_score = hybrid_results[idx]['score']
            bm25_rank = hybrid_results[idx].get('bm25_rank')
            semantic_rank = hybrid_results[idx].get('semantic_rank')

            source = metadata.get('source', metadata.get('filename', 'unknown'))
            chunk_index = metadata.get('chunk_index', 0)

            total_chunks = metadata.get('total_chunks')
            try:
                total_chunks = int(total_chunks)
            except (TypeError, ValueError):
                total_chunks = None

            if total_chunks is None:
                doc_entry = _get_document_cache_entry(collection, source, doc_cache)
                meta_list = doc_entry.get('metadatas') if doc_entry else []
                total_chunks = len(meta_list) if meta_list else 1

            output += f"[{i}] Rerank Score: {score:.4f} | Hybrid: {hybrid_score:.4f}\n"
            output += f"    Source: {source}\n"
            output += f"    Position: chunk {chunk_index}/{total_chunks}\n"
            output += f"    Rankings: BM25 #{bm25_rank}, Semantic #{semantic_rank}\n\n"

            # Retrieve adjacent chunks for context
            adjacent_chunks = _get_adjacent_chunks(
                collection, source, chunk_index, total_chunks, doc_cache=doc_cache
            )

            if adjacent_chunks:
                # Display context window with visual indicator for main chunk
                for chunk_content, chunk_meta in adjacent_chunks:
                    is_main = chunk_meta['chunk_index'] == chunk_index
                    marker = "[*]" if is_main else "[ ]"

                    output += f"{marker} [Chunk {chunk_meta['chunk_index']}]"
                    if is_main:
                        output += " <-- MAIN RESULT"
                    output += "\n"

                    # Show content preview
                    preview_length = 300 if is_main else 150
                    preview = chunk_content[:preview_length]
                    if len(chunk_content) > preview_length:
                        preview += "..."
                    output += f"    {preview}\n\n"
            else:
                # Fallback if adjacent chunks not found
                output += f"    Content: {documents_for_rerank[idx][:200]}...\n\n"

            output += "-" * 70 + "\n\n"

        return output

    except Exception as e:
        return f"ERROR: {str(e)}"


@mcp.tool()
def semantic_search_hybrid(query: str, top_k: int = 10, alpha: float = 0.7) -> str:
    """
    Hybrid search with BM25 + Voyage-Context-3 + Cohere v3.5 reranking.

    Args:
        query: Search query about the indexed knowledge base.
        top_k: Number of results to return (default: 10)
        alpha: Semantic weight (0.7 = 70% semantic, 30% BM25). Use 0.5 for equal weight.

    Returns:
        Formatted search results with hybrid ranking scores and source information.
    """
    return _perform_search_hybrid(query, top_k, alpha)


@mcp.tool()
def search_by_source(query: str, sources: list, top_k: int = 10, alpha: float = 0.7) -> str:
    """
    Hybrid search limited to specific documents.

    Args:
        query: Search query about the indexed knowledge base.
        sources: List of document filenames to search in (e.g., ["1982_RGSP.md", "2009_RSE_Painter.md"])
        top_k: Number of results to return (default: 10)
        alpha: Semantic weight (0.7 = 70% semantic, 30% BM25). Use 0.5 for equal weight.

    Returns:
        Formatted search results from specified documents only.

    Examples:
        search_by_source("glacier albedo", sources=["1982_RGSP.md"])
        search_by_source("ice mass balance", sources=["Warren_1982.md", "Painter_2009.md"], top_k=5)
    """
    # Build where filter for source filtering
    if len(sources) == 1:
        where = {"source": sources[0]}
    else:
        where = {"source": {"$in": sources}}

    return _perform_search_hybrid(query, top_k, alpha, where=where)


@mcp.tool()
def list_documents() -> str:
    """
    List all indexed documents.

    Returns:
        List of available papers with metadata.
    """

    try:
        init_clients()
        collection = chroma_client.get_collection(name=COLLECTION_HYBRID_NAME)

        # Get metadata for all documents
        all_docs = collection.get(include=["metadatas"])

        # Extract unique sources
        sources = {}
        for metadata in all_docs['metadatas']:
            source = metadata.get('source', metadata.get('filename', 'unknown'))
            if source not in sources:
                sources[source] = metadata

        output = f"INDEXED DOCUMENTS: {len(sources)} papers\n"
        output += "=" * 70 + "\n\n"

        for i, (source, metadata) in enumerate(sorted(sources.items()), 1):
            output += f"[{i}] {source}\n"
            output += f"    Title: {metadata.get('title', 'No title')}\n"
            output += f"    Chunks: {metadata.get('total_chunks', 'N/A')}\n\n"

        return output

    except Exception as e:
        return f"ERROR: {str(e)}"


@mcp.tool()
def get_document_content(source: str, format: str = "markdown", max_length: int = None) -> str:
    """
    Get complete document content by reconstructing from chunks.

    Args:
        source: Document source filename (from list_documents)
        format: Output format - "markdown" (default), "text", or "chunks"
        max_length: Maximum characters to return (None = unlimited)

    Returns:
        Complete reconstructed document with metadata
    """
    try:
        init_clients()
        collection = chroma_client.get_collection(name=COLLECTION_HYBRID_NAME)

        # Fetch all chunks for this document
        doc_data = _fetch_document_chunks(collection, source)

        if not doc_data['documents']:
            return f"ERROR: Document '{source}' not found in the database."

        documents = doc_data['documents']
        metadatas = doc_data['metadatas']

        # Sort chunks by chunk_index
        combined = list(zip(documents, metadatas))
        combined.sort(key=lambda x: x[1].get('chunk_index', 0))

        # Extract metadata from first chunk
        first_meta = combined[0][1]
        total_chunks = len(combined)
        doc_hash = first_meta.get('doc_hash', 'N/A')
        indexed_date = first_meta.get('indexed_date', 'N/A')
        model = first_meta.get('model', 'N/A')
        title = first_meta.get('title', source)

        # Build output based on format
        output = ""

        if format == "chunks":
            # Show individual chunks with metadata
            output += f"DOCUMENT: {source}\n"
            output += "=" * 70 + "\n"
            output += f"Title: {title}\n"
            output += f"Total chunks: {total_chunks}\n"
            output += f"Indexed: {indexed_date}\n"
            output += f"Model: {model}\n"
            output += f"Hash: {doc_hash}\n"
            output += "=" * 70 + "\n\n"

            for chunk_text, chunk_meta in combined:
                chunk_idx = chunk_meta.get('chunk_index', 0)
                output += f"[Chunk {chunk_idx}]\n"
                output += f"{chunk_text}\n"
                output += "-" * 70 + "\n\n"

        elif format == "text":
            # Plain text reconstruction
            full_text = "\n\n".join([chunk for chunk, _ in combined])
            output = full_text

        else:  # markdown (default)
            # Markdown with metadata header
            output += f"# {title}\n\n"
            output += f"**Source:** {source}  \n"
            output += f"**Total chunks:** {total_chunks}  \n"
            output += f"**Indexed:** {indexed_date}  \n"
            output += f"**Model:** {model}  \n"
            output += f"**Hash:** {doc_hash}  \n"
            output += "\n" + "=" * 70 + "\n\n"

            # Reconstruct full text
            full_text = "\n\n".join([chunk for chunk, _ in combined])
            output += full_text

        # Apply max_length if specified
        if max_length and len(output) > max_length:
            output = output[:max_length] + "\n\n... (truncated)"

        return output

    except Exception as e:
        return f"ERROR: {str(e)}"


@mcp.tool()
def get_chunk_with_context(chunk_id: str, context_size: int = 2, highlight: bool = True) -> str:
    """
    Show a chunk with surrounding chunks for context.

    Args:
        chunk_id: ID of the target chunk (from search results)
        context_size: Number of chunks before and after (default: 2)
        highlight: Highlight the matched chunk (default: True)

    Returns:
        Target chunk with surrounding context chunks
    """
    try:
        init_clients()
        collection = chroma_client.get_collection(name=COLLECTION_HYBRID_NAME)

        # Get the target chunk
        target = collection.get(
            ids=[chunk_id],
            include=["documents", "metadatas"]
        )

        if not target['documents']:
            return f"ERROR: Chunk '{chunk_id}' not found in the database."

        target_text = target['documents'][0]
        target_meta = target['metadatas'][0]

        source = target_meta.get('source', target_meta.get('filename', 'unknown'))
        chunk_index = target_meta.get('chunk_index', 0)
        total_chunks = target_meta.get('total_chunks')

        # Convert total_chunks to int
        try:
            total_chunks = int(total_chunks)
        except (TypeError, ValueError):
            # Fallback: fetch all chunks to determine total
            doc_data = _fetch_document_chunks(collection, source)
            total_chunks = len(doc_data['metadatas']) if doc_data['metadatas'] else 1

        # Get adjacent chunks using existing helper
        adjacent_chunks = _get_adjacent_chunks(
            collection,
            source,
            chunk_index,
            total_chunks,
            window_size=context_size
        )

        if not adjacent_chunks:
            # Fallback to showing just the target chunk
            output = f"CHUNK CONTEXT: {source}\n"
            output += "=" * 70 + "\n"
            output += f"⚠️  No context chunks found. Showing target chunk only.\n\n"
            output += f"[Chunk {chunk_index}/{total_chunks}]\n"
            output += f"{target_text}\n"
            return output

        # Build output with context
        output = f"CHUNK CONTEXT: {source}\n"
        output += "=" * 70 + "\n"
        output += f"Showing chunk {chunk_index}/{total_chunks} "
        output += f"with ±{context_size} chunks of context\n"
        output += "=" * 70 + "\n\n"

        for chunk_text, chunk_meta in adjacent_chunks:
            current_idx = chunk_meta.get('chunk_index', 0)
            is_target = current_idx == chunk_index

            # Add visual marker
            if highlight and is_target:
                output += ">>> [TARGET CHUNK] <<<\n"
                output += f"[Chunk {current_idx}/{total_chunks}] *** MATCHED RESULT ***\n"
            else:
                output += f"[Chunk {current_idx}/{total_chunks}]\n"

            output += f"{chunk_text}\n"

            if highlight and is_target:
                output += ">>> [END TARGET CHUNK] <<<\n"

            output += "-" * 70 + "\n\n"

        return output

    except Exception as e:
        return f"ERROR: {str(e)}"


@mcp.tool()
def get_indexation_status() -> str:
    """
    Get current indexation database statistics.

    Returns:
        Formatted report with document count, chunk count, and metadata information.
    """
    try:
        init_clients()
        collection = chroma_client.get_collection(name=COLLECTION_HYBRID_NAME)

        all_docs = collection.get(include=["metadatas"])

        docs_by_source = {}
        for metadata in all_docs['metadatas']:
            source = metadata.get('source', metadata.get('filename', 'unknown'))
            if source not in docs_by_source:
                docs_by_source[source] = {
                    'chunks': 0,
                    'hash': metadata.get('doc_hash'),
                    'indexed_date': metadata.get('indexed_date'),
                    'model': metadata.get('model')
                }
            docs_by_source[source]['chunks'] += 1

        total_chunks = len(all_docs['ids'])
        total_docs = len(docs_by_source)

        output = "INDEXATION STATUS - HYBRID SEARCH ENABLED\n"
        output += "=" * 70 + "\n\n"

        output += f"GLOBAL STATISTICS:\n"
        output += f"   Number of documents: {total_docs}\n"
        output += f"   Total chunks: {total_chunks}\n"
        if total_docs > 0:
            output += f"   Average chunks/doc: {total_chunks / total_docs:.1f}\n\n"

        docs_with_hash = sum(1 for d in docs_by_source.values() if d['hash'])
        docs_with_date = sum(1 for d in docs_by_source.values() if d['indexed_date'])

        output += f"METADATA VERIFICATION:\n"
        output += f"   Documents with MD5 hash: {docs_with_hash}/{total_docs}\n"
        output += f"   Documents with date: {docs_with_date}/{total_docs}\n\n"

        models = {}
        for doc in docs_by_source.values():
            model = doc['model'] or 'unknown'
            models[model] = models.get(model, 0) + 1

        output += f"MODEL DISTRIBUTION:\n"
        for model, count in sorted(models.items()):
            pct = (count / total_docs) * 100
            output += f"   {model:30} {count:3d} docs ({pct:5.1f}%)\n"

        output += f"\n" + "=" * 70 + "\n"
        output += "RETRIEVAL MODE: HYBRID (BM25 + Semantic + Reranking)\n"
        output += "=" * 70

        return output

    except Exception as e:
        return f"ERROR: {str(e)}"


if __name__ == "__main__":
    mcp.run()
