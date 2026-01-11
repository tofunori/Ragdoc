#!/usr/bin/env python3
"""
HTTP server version of ragdoc MCP (SSE transport).

Usage:
    python src/server_http.py

This runs ragdoc as an HTTP/SSE server instead of stdio.
"""
import sys
import os
import re
import unicodedata

# Set working directory
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

from dotenv import load_dotenv
load_dotenv()

from fastmcp import FastMCP
import chromadb
import voyageai
import cohere

from src.config import (
    RAGDOC_MODE,
    COLLECTION_NAME,
    ACTIVE_DB_PATH,
)
from src.hybrid_retriever import HybridRetriever

# Initialize MCP server
mcp = FastMCP(f"ragdoc-{RAGDOC_MODE}")

# Global clients
voyage_client = None
chroma_client = None
cohere_client = None
hybrid_retriever = None

def _normalize_query_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", text).strip()


def _generate_query_variants(query: str, n_queries: int = 3) -> list[str]:
    q0 = query.strip() if query else ""
    if not q0:
        return []

    n_queries = int(n_queries) if n_queries is not None else 3
    n_queries = max(1, min(n_queries, 5))

    if len(q0) > 500:
        return [q0]

    q_norm = _normalize_query_text(q0)
    q_lower = q_norm.lower()

    replacements: list[tuple[str, str]] = [
        ("albédo", "albedo"),
        ("télédétection", "remote sensing"),
        ("carbone noir", "black carbon"),
        ("neige", "snow"),
        ("glaciers", "glacier"),
        (" bc ", " black carbon "),
        (" ssa ", " specific surface area "),
        (" modis ", " moderate resolution imaging spectroradiometer modis "),
        (" lst ", " land surface temperature "),
        (" firn ", " firn snow "),
    ]

    expanded = f" {q_lower} "
    for src, dst in replacements:
        expanded = expanded.replace(src, dst)
    expanded = re.sub(r"\s+", " ", expanded).strip()

    stop = {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "from", "by", "as", "is", "are",
        "this", "that", "these", "those", "what", "how", "why", "which", "who",
        "le", "la", "les", "un", "une", "des", "du", "de", "d", "et", "ou", "dans", "sur", "pour", "avec",
        "par", "en", "au", "aux", "ce", "cet", "cette", "ces", "quoi", "comment", "pourquoi", "quel", "quelle",
    }
    tokens = re.findall(r"[a-z0-9_]+", expanded)
    keywords = [t for t in tokens if t not in stop and len(t) >= 3]
    keyword_variant = " ".join(dict.fromkeys(keywords))

    candidates = [q0, q_norm]
    if expanded and expanded not in candidates:
        candidates.append(expanded)
    if keyword_variant and keyword_variant not in candidates:
        candidates.append(keyword_variant)

    out: list[str] = []
    seen = set()
    for q in candidates:
        q = q.strip()
        if not q or q in seen:
            continue
        seen.add(q)
        out.append(q)
        if len(out) >= n_queries:
            break

    return out or [q0]


def _multiquery_rrf_fuse(results_by_query: list[list[dict]], rrf_k: int = 60) -> list[dict]:
    scores: dict[str, float] = {}
    payload: dict[str, dict] = {}
    for res_list in results_by_query:
        for rank, r in enumerate(res_list):
            doc_id = r.get("id")
            if not doc_id:
                continue
            scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank + 1))
            if doc_id not in payload:
                payload[doc_id] = r

    merged = []
    for doc_id, s in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        item = dict(payload[doc_id])
        item["score"] = float(s)
        merged.append(item)
    return merged


def init_clients():
    """Initialize API clients with auto-detection of ChromaDB server"""
    global voyage_client, chroma_client, cohere_client, hybrid_retriever

    if not voyage_client:
        voyage_client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

    if not chroma_client:
        try:
            test_client = chromadb.HttpClient(host="localhost", port=8000)
            test_client.heartbeat()
            chroma_client = test_client
        except Exception:
            chroma_client = chromadb.PersistentClient(path=str(ACTIVE_DB_PATH))

    if not cohere_client:
        cohere_client = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))

    if not hybrid_retriever:
        try:
            collection = chroma_client.get_collection(name=COLLECTION_NAME)

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

            hybrid_retriever = HybridRetriever(
                collection=collection,
                embedding_function=voyage_contextualized_embed
            )
        except Exception as e:
            print(f"Warning: Failed to initialize HybridRetriever: {e}")


@mcp.tool()
def semantic_search_hybrid(
    query: str,
    top_k: int = 10,
    alpha: float = 0.5,
    multi_query: bool = False,
    n_queries: int = 3
) -> str:
    """
    Hybrid search with BM25 + Vector + Cohere v3.5 reranking.

    Args:
        query: Search query about the indexed knowledge base.
        top_k: Number of results to return (default: 10)
        alpha: Semantic weight (0.5 = balanced hybrid).

    Returns:
        Formatted search results with hybrid ranking scores.
    """
    try:
        init_clients()
        if not hybrid_retriever:
            return "ERROR: Hybrid retriever not initialized."

        queries = _generate_query_variants(query, n_queries=n_queries) if multi_query else [query]
        per_query_results: list[list[dict]] = []
        for q in queries:
            res = hybrid_retriever.search(query=q, top_k=50, alpha=alpha)
            if res:
                per_query_results.append(res)

        if not per_query_results:
            return "No results found."

        hybrid_results = per_query_results[0] if len(per_query_results) == 1 else _multiquery_rrf_fuse(per_query_results)
        hybrid_results = hybrid_results[:100]

        if not hybrid_results:
            return "No results found."

        documents_for_rerank = [r['text'] for r in hybrid_results]
        metadatas = [r['metadata'] for r in hybrid_results]

        rerank_results = cohere_client.rerank(
            model="rerank-v4.0-pro",
            query=query,
            documents=documents_for_rerank,
            top_n=top_k
        )

        output = f"SEARCH RESULTS: {query}\n{'=' * 70}\n\n"
        if multi_query and len(queries) > 1:
            output += f"[i] Multi-query enabled ({len(queries)} variants)\n\n"
        for i, result in enumerate(rerank_results.results, 1):
            idx = result.index
            score = result.relevance_score
            metadata = metadatas[idx]
            source = metadata.get('source', 'unknown')
            output += f"[{i}] Score: {score:.4f} | Source: {source}\n"
            output += f"    {documents_for_rerank[idx][:500]}...\n\n"

        return output
    except Exception as e:
        return f"ERROR: {str(e)}"


@mcp.tool()
def search_by_source(
    query: str,
    sources: list,
    top_k: int = 10,
    alpha: float = 0.5,
    multi_query: bool = False,
    n_queries: int = 3
) -> str:
    """
    Hybrid search limited to specific documents.

    Args:
        query: Search query about the indexed knowledge base.
        sources: List of document filenames to search in
        top_k: Number of results to return (default: 10)
        alpha: Semantic weight (0.7 = 70% semantic, 30% BM25).

    Returns:
        Formatted search results from specified documents only.
    """
    try:
        init_clients()
        if not hybrid_retriever:
            return "ERROR: Hybrid retriever not initialized."

        queries = _generate_query_variants(query, n_queries=n_queries) if multi_query else [query]
        per_query_results: list[list[dict]] = []
        for q in queries:
            res = hybrid_retriever.search(query=q, top_k=100, alpha=alpha)
            if res:
                per_query_results.append(res)

        if not per_query_results:
            return "No results found."

        hybrid_results = per_query_results[0] if len(per_query_results) == 1 else _multiquery_rrf_fuse(per_query_results)
        hybrid_results = hybrid_results[:200]

        if not hybrid_results:
            return "No results found."

        # Filter by sources
        filtered_results = [
            r for r in hybrid_results
            if r['metadata'].get('source', '') in sources
        ]

        if not filtered_results:
            return f"No results found in specified sources: {sources}"

        documents_for_rerank = [r['text'] for r in filtered_results]
        metadatas = [r['metadata'] for r in filtered_results]

        rerank_results = cohere_client.rerank(
            model="rerank-v4.0-pro",
            query=query,
            documents=documents_for_rerank,
            top_n=top_k
        )

        output = f"SEARCH RESULTS (filtered): {query}\n{'=' * 70}\n\n"
        for i, result in enumerate(rerank_results.results, 1):
            idx = result.index
            score = result.relevance_score
            metadata = metadatas[idx]
            source = metadata.get('source', 'unknown')
            output += f"[{i}] Score: {score:.4f} | Source: {source}\n"
            output += f"    {documents_for_rerank[idx][:500]}...\n\n"

        return output
    except Exception as e:
        return f"ERROR: {str(e)}"


@mcp.tool()
def list_documents() -> str:
    """List all indexed documents."""
    try:
        init_clients()
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
        all_docs = collection.get(include=["metadatas"])

        sources = {}
        for metadata in all_docs['metadatas']:
            source = metadata.get('source', 'unknown')
            if source not in sources:
                sources[source] = metadata

        output = f"INDEXED DOCUMENTS ({len(sources)} papers)\n{'=' * 70}\n\n"
        for i, (source, metadata) in enumerate(sorted(sources.items()), 1):
            output += f"[{i}] {source}\n"
            output += f"    Title: {metadata.get('title', 'No title')}\n\n"

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
        max_length: Maximum characters to return (default: 80000)
    """
    try:
        init_clients()
        collection = chroma_client.get_collection(name=COLLECTION_NAME)

        results = collection.get(
            where={"source": source},
            include=["documents", "metadatas"]
        )

        if not results['documents']:
            return f"Document not found: {source}"

        # Sort by chunk_index if available
        chunks = list(zip(results['documents'], results['metadatas']))
        chunks.sort(key=lambda x: x[1].get('chunk_index', 0))

        if format == "chunks":
            output = f"DOCUMENT CHUNKS: {source}\n{'=' * 70}\n\n"
            for i, (doc, meta) in enumerate(chunks):
                output += f"--- Chunk {i} ---\n{doc}\n\n"
        else:
            content = "\n\n".join([doc for doc, _ in chunks])
            output = f"DOCUMENT: {source}\n{'=' * 70}\n\n{content}"

        if len(output) > max_length:
            output = output[:max_length] + f"\n\n... [Truncated at {max_length} chars]"

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
    """
    try:
        init_clients()
        collection = chroma_client.get_collection(name=COLLECTION_NAME)

        # Get the target chunk
        result = collection.get(ids=[chunk_id], include=["documents", "metadatas"])

        if not result['documents']:
            return f"Chunk not found: {chunk_id}"

        source = result['metadatas'][0].get('source', '')
        chunk_index = result['metadatas'][0].get('chunk_index', 0)

        # Get all chunks from same document
        all_chunks = collection.get(
            where={"source": source},
            include=["documents", "metadatas"]
        )

        chunks = list(zip(all_chunks['ids'], all_chunks['documents'], all_chunks['metadatas']))
        chunks.sort(key=lambda x: x[2].get('chunk_index', 0))

        # Find surrounding chunks
        target_idx = None
        for i, (cid, _, _) in enumerate(chunks):
            if cid == chunk_id:
                target_idx = i
                break

        if target_idx is None:
            return f"Could not locate chunk in document: {chunk_id}"

        start = max(0, target_idx - context_size)
        end = min(len(chunks), target_idx + context_size + 1)

        output = f"CHUNK WITH CONTEXT: {source}\n{'=' * 70}\n\n"
        for i in range(start, end):
            cid, doc, meta = chunks[i]
            if i == target_idx and highlight:
                output += f">>> MATCHED CHUNK (index {meta.get('chunk_index', i)}) <<<\n"
                output += f"{doc}\n"
                output += f">>> END MATCHED CHUNK <<<\n\n"
            else:
                output += f"--- Chunk {meta.get('chunk_index', i)} ---\n{doc}\n\n"

        return output
    except Exception as e:
        return f"ERROR: {str(e)}"


@mcp.tool()
def get_indexation_status() -> str:
    """Get current indexation database statistics."""
    try:
        init_clients()
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
        all_docs = collection.get(include=["metadatas"])

        docs_by_source = {}
        for metadata in all_docs['metadatas']:
            source = metadata.get('source', 'unknown')
            if source not in docs_by_source:
                docs_by_source[source] = {'chunks': 0}
            docs_by_source[source]['chunks'] += 1

        total_chunks = len(all_docs['ids'])
        total_docs = len(docs_by_source)

        output = f"INDEXATION STATUS\n{'=' * 70}\n\n"
        output += f"Documents: {total_docs}\n"
        output += f"Total chunks: {total_chunks}\n"
        if total_docs > 0:
            output += f"Avg chunks/doc: {total_chunks / total_docs:.1f}\n"

        return output
    except Exception as e:
        return f"ERROR: {str(e)}"


if __name__ == "__main__":
    print("=" * 60)
    print("RAGDOC MCP HTTP SERVER")
    print("=" * 60)
    print(f"Starting on http://127.0.0.1:8321/sse")
    print("Press Ctrl+C to stop")
    print("=" * 60)

    # Run as SSE server (compatible with most MCP clients)
    mcp.run(transport="sse", host="127.0.0.1", port=8321)
