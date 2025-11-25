#!/usr/bin/env python3
"""
MCP Server for RAGDOC
Contextualized embeddings + BM25 hybrid search with Cohere reranking.
"""

import os
import sys
import hashlib
import logging
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure src is in path if running as script
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from fastmcp import FastMCP
import chromadb
import voyageai
import cohere

# Import internal modules
from src.config import (
    RAGDOC_MODE,
    COLLECTION_NAME,
    ACTIVE_DB_PATH,
    CONTEXT_WINDOW_SIZE,
    VOYAGE_API_KEY,
    COHERE_API_KEY,
    LOG_LEVEL
)
from src.hybrid_retriever import HybridRetriever

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Initialize MCP server (single contextualized mode)
mcp = FastMCP(f"ragdoc-{RAGDOC_MODE}")

# Global clients (initialized on first use)
voyage_client = None
chroma_client = None
cohere_client = None
hybrid_retriever = None

def init_clients():
    """Initialize API clients with auto-detection of ChromaDB server"""
    global voyage_client, chroma_client, cohere_client, hybrid_retriever

    if not voyage_client:
        if not os.getenv("VOYAGE_API_KEY"):
             logging.warning("VOYAGE_API_KEY not set. Semantic search will fail.")
        voyage_client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

    if not chroma_client:
        # Try HttpClient (server mode) first, fallback to PersistentClient
        try:
            test_client = chromadb.HttpClient(host="localhost", port=8000)
            test_client.heartbeat()
            chroma_client = test_client
            logging.info(f"[OK] MCP: Connected to ChromaDB server (localhost:8000) - Collection: {COLLECTION_NAME}")
        except Exception:
            logging.info(f"[INFO] MCP: ChromaDB server not available, using local mode: {ACTIVE_DB_PATH}")
            chroma_client = chromadb.PersistentClient(path=str(ACTIVE_DB_PATH))

    if not cohere_client:
        if not os.getenv("COHERE_API_KEY"):
            logging.warning("COHERE_API_KEY not set. Reranking will fail.")
        cohere_client = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))

    # Initialize retriever (contextualized embeddings + BM25 fusion)
    if not hybrid_retriever:
        try:
            collection = chroma_client.get_collection(name=COLLECTION_NAME)

            # Contextualized embedding function (only mode supported)
            def voyage_contextualized_embed(texts):
                results = []
                for text in texts:
                    result = voyage_client.contextualized_embed(
                        inputs=[[text]],
                        model="voyage-context-3",
                        input_type="query"
                    )
                    results.append(result.results[0].embeddings[0])
                return results

            embed_fn = voyage_contextualized_embed
            logging.info("[OK] Retriever initialized (Contextualized Mode)")

            hybrid_retriever = HybridRetriever(
                collection=collection,
                embedding_function=embed_fn
            )
        except Exception as e:
            logging.error(f"Failed to initialize HybridRetriever: {e}")
            # Don't crash, just allow other tools to work


def _fetch_document_chunks(collection, source: str) -> dict:
    """
    Fetch all chunks for a given document source.
    """
    for field in ("source", "filename"):
        try:
            results = collection.get(
                where={field: source},
                include=["documents", "metadatas"]
            )
        except Exception as fetch_error:
            logging.warning(f"Failed to fetch chunks for {field}={source}: {fetch_error}")
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
        logging.warning(f"Error retrieving adjacent chunks for {source}: {e}")
        return []


def _perform_search_hybrid(
    query: str,
    top_k: int = 10,
    alpha: float = 0.5,
    where: dict = None,
    where_document: dict = None
) -> str:
    """
    Unified search (contextualized embeddings + BM25 + Cohere rerank).
    """
    try:
        init_clients()
        
        if not hybrid_retriever:
            return "ERROR: Hybrid retriever not initialized. Check database connection."

        collection = chroma_client.get_collection(name=COLLECTION_NAME)

        # 1. Retrieval (BM25 + contextualized semantic with RRF)
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
        output = f"SEARCH RESULTS ({RAGDOC_MODE.upper()} MODE): {query}\n"
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

            output += f"[{i}] Rerank Score: {score:.4f} | Fusion: {hybrid_score:.4f}\n"
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
                    # Longer preview for main chunk
                    preview_length = 800 if is_main else 400
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
        logging.exception("Error during hybrid search")
        return f"ERROR: {str(e)}"


@mcp.tool()
def semantic_search_hybrid(query: str, top_k: int = 10, alpha: float = 0.5) -> str:
    """
    Hybrid search with BM25 + Vector + Cohere v3.5 reranking.

    Args:
        query: Search query about the indexed knowledge base.
        top_k: Number of results to return (default: 10)
        alpha: Semantic weight (0.5 = balanced hybrid). Use 0.3 for BM25-heavy, 0.7 for semantic-heavy.

    Returns:
        Formatted search results with hybrid ranking scores and source information.
    """
    return _perform_search_hybrid(query, top_k, alpha)


@mcp.tool()
def search_by_source(query: str, sources: list, top_k: int = 10, alpha: float = 0.5) -> str:
    """
    Hybrid search limited to specific documents.

    Args:
        query: Search query about the indexed knowledge base.
        sources: List of document filenames to search in (e.g., ["1982_RGSP.md", "2009_RSE_Painter.md"])
        top_k: Number of results to return (default: 10)
        alpha: Semantic weight (0.7 = 70% semantic, 30% BM25). Use 0.5 for equal weight.

    Returns:
        Formatted search results from specified documents only.
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
        collection = chroma_client.get_collection(name=COLLECTION_NAME)

        all_docs = collection.get(include=["metadatas"])

        sources = {}
        for metadata in all_docs['metadatas']:
            source = metadata.get('source', metadata.get('filename', 'unknown'))
            if source not in sources:
                sources[source] = metadata

        output = f"INDEXED DOCUMENTS ({len(sources)} papers)\n"
        output += f"Mode: {RAGDOC_MODE}\n"
        output += "=" * 70 + "\n\n"

        for i, (source, metadata) in enumerate(sorted(sources.items()), 1):
            output += f"[{i}] {source}\n"
            output += f"    Title: {metadata.get('title', 'No title')}\n"
            output += f"    Chunks: {metadata.get('total_chunks', 'N/A')}\n\n"

        return output

    except Exception as e:
        return f"ERROR: {str(e)}"


@mcp.tool()
def get_document_content(source: str, format: str = "markdown", max_length: int = 80000) -> str:
    """
    Get complete document content by reconstructing from chunks.

    Args:
        source: Document source filename (from list_documents)
        format: Output format - "markdown" (default), "text", or "chunks"
        max_length: Maximum characters to return (default: 80000 chars ≈ 20K tokens)
    """
    try:
        init_clients()
        collection = chroma_client.get_collection(name=COLLECTION_NAME)

        doc_data = _fetch_document_chunks(collection, source)

        if not doc_data['documents']:
            return f"ERROR: Document '{source}' not found in the database."

        documents = doc_data['documents']
        metadatas = doc_data['metadatas']

        combined = list(zip(documents, metadatas))
        combined.sort(key=lambda x: x[1].get('chunk_index', 0))

        first_meta = combined[0][1]
        total_chunks = len(combined)
        doc_hash = first_meta.get('doc_hash', 'N/A')
        indexed_date = first_meta.get('indexed_date', 'N/A')
        model = first_meta.get('model', 'N/A')
        title = first_meta.get('title', source)

        output = ""

        if format == "chunks":
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
            full_text = "\n\n".join([chunk for chunk, _ in combined])
            output = full_text

        else:  # markdown (default)
            output += f"# {title}\n\n"
            output += f"**Source:** {source}  \n"
            output += f"**Total chunks:** {total_chunks}  \n"
            output += f"**Indexed:** {indexed_date}  \n"
            output += f"**Model:** {model}  \n"
            output += f"**Hash:** {doc_hash}  \n"
            output += "\n" + "=" * 70 + "\n\n"

            full_text = "\n\n".join([chunk for chunk, _ in combined])
            output += full_text

        if max_length and len(output) > max_length:
            original_length = len(output)
            estimated_tokens = original_length // 4
            output = output[:max_length]
            output += f"\n\n... (truncated: showing {max_length:,} of {original_length:,} chars, ~{estimated_tokens:,} tokens total)"
            output += f"\n\n[i] Use max_length parameter to adjust limit or retrieve in chunks via semantic_search_hybrid"

        return output

    except Exception as e:
        return f"ERROR: {str(e)}"


@mcp.tool()
def get_chunk_with_context(chunk_id: str, context_size: int = 2, highlight: bool = True) -> str:
    """
    Show a chunk with surrounding chunks for context.

    Args:
        chunk_id: ID of the target chunk (from search results) or in format "source_name_XX"
        context_size: Number of chunks before and after (default: 2)
        highlight: Highlight the matched chunk (default: True)
    """
    try:
        init_clients()
        collection = chroma_client.get_collection(name=COLLECTION_NAME)

        target = collection.get(
            ids=[chunk_id],
            include=["documents", "metadatas"]
        )

        if not target['documents']:
            # Backward compatibility: parse source_name_XX
            if '_' in chunk_id and chunk_id.count('_') >= 1:
                parts = chunk_id.rsplit('_', 1)
                if len(parts) == 2:
                    source = parts[0]
                    if not source.endswith('.md'):
                        source = source + '.md'
                    try:
                        chunk_index = int(parts[1])
                        all_chunks = collection.get(
                            where={
                                "$and": [
                                    {"source": {"$eq": source}},
                                    {"chunk_index": {"$eq": chunk_index}}
                                ]
                            },
                            include=["documents", "metadatas"]
                        )
                        if all_chunks['documents']:
                            target = all_chunks
                            chunk_id = all_chunks['ids'][0]
                    except (ValueError, KeyError):
                        pass

        if not target['documents']:
            return f"ERROR: Chunk '{chunk_id}' not found in the database."

        target_text = target['documents'][0]
        target_meta = target['metadatas'][0]

        source = target_meta.get('source', target_meta.get('filename', 'unknown'))
        chunk_index = target_meta.get('chunk_index', 0)
        total_chunks = target_meta.get('total_chunks')

        try:
            total_chunks = int(total_chunks)
        except (TypeError, ValueError):
            doc_data = _fetch_document_chunks(collection, source)
            total_chunks = len(doc_data['metadatas']) if doc_data['metadatas'] else 1

        adjacent_chunks = _get_adjacent_chunks(
            collection,
            source,
            chunk_index,
            total_chunks,
            window_size=context_size
        )

        if not adjacent_chunks:
            output = f"CHUNK CONTEXT: {source}\n"
            output += "=" * 70 + "\n"
            output += f"[!] No context chunks found. Showing target chunk only.\n\n"
            output += f"[Chunk {chunk_index}/{total_chunks}]\n"
            output += f"{target_text}\n"
            return output

        output = f"CHUNK CONTEXT: {source}\n"
        output += "=" * 70 + "\n"
        output += f"Showing chunk {chunk_index}/{total_chunks} "
        output += f"with ±{context_size} chunks of context\n"
        output += "=" * 70 + "\n\n"

        for chunk_text, chunk_meta in adjacent_chunks:
            current_idx = chunk_meta.get('chunk_index', 0)
            is_target = current_idx == chunk_index

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
    """
    try:
        init_clients()
        collection = chroma_client.get_collection(name=COLLECTION_NAME)

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

        output = f"INDEXATION STATUS - {RAGDOC_MODE.upper()} MODE\n"
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

        return output

    except Exception as e:
        return f"ERROR: {str(e)}"


def main():
    """Entry point for CLI execution"""
    parser = argparse.ArgumentParser(description="Ragdoc MCP Server")
    parser.add_argument("--mode", choices=["hybrid", "contextualized"], 
                        help="Override operation mode")
    args, unknown = parser.parse_known_args()

    # If mode is passed via CLI, warn user it might not persist for MCP stdio
    if args.mode:
        logging.info(f"Starting in {args.mode} mode (CLI override)")
        # We can't easily change the global constant here because it's imported
        # but we could set os.environ and re-exec, or refactor config loading.
        # For now, rely on env vars.
    
    mcp.run()


if __name__ == "__main__":
    main()
