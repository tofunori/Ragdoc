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
import re
import unicodedata
from pathlib import Path
from dotenv import load_dotenv

# Windows: normalize newlines for stdio transports (avoids CRLF issues in some MCP clients)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(newline="\n", encoding="utf-8", errors="replace")
    except Exception:
        pass
    try:
        sys.stderr.reconfigure(newline="\n", encoding="utf-8", errors="replace")
    except Exception:
        pass
    try:
        sys.stdin.reconfigure(newline="\n")
    except Exception:
        pass

# Load environment variables
load_dotenv()

# Ensure src is in path if running as script
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from fastmcp import FastMCP

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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr,  # never write logs to stdout (stdout is reserved for MCP JSON-RPC)
)

# Initialize MCP server (single contextualized mode)
mcp = FastMCP(f"ragdoc-{RAGDOC_MODE}")

# Global clients (initialized on first use)
voyage_client = None
chroma_client = None
cohere_client = None
hybrid_retriever = None
_chromadb = None
_voyageai = None
_cohere = None


def _get_chromadb():
    global _chromadb
    if _chromadb is None:
        import chromadb as _chromadb_mod
        _chromadb = _chromadb_mod
    return _chromadb


def _get_voyageai():
    global _voyageai
    if _voyageai is None:
        import voyageai as _voyageai_mod
        _voyageai = _voyageai_mod
    return _voyageai


def _get_cohere():
    global _cohere
    if _cohere is None:
        import cohere as _cohere_mod
        _cohere = _cohere_mod
    return _cohere

def _normalize_query_text(text: str) -> str:
    """Best-effort query normalization (diacritics, whitespace)."""
    if not text:
        return ""
    # Normalize unicode accents (albédo -> albedo)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _generate_query_variants(query: str, n_queries: int = 3) -> list[str]:
    """
    Heuristic query rewriting/expansion.
    Produces up to n_queries queries INCLUDING the original query.

    Goal: help short/noisy queries (acronyms, FR/EN variants, key synonyms).
    """
    q0 = query.strip() if query else ""
    if not q0:
        return []

    n_queries = int(n_queries) if n_queries is not None else 3
    n_queries = max(1, min(n_queries, 5))  # keep it cheap & predictable

    # Prefer short queries; for long chunk-like queries, keep only the original.
    if len(q0) > 500:
        return [q0]

    q_norm = _normalize_query_text(q0)
    q_lower = q_norm.lower()

    # Domain-oriented acronym/synonym expansions (FR/EN) – small but high-signal.
    replacements: list[tuple[str, str]] = [
        # French -> English (common in papers)
        ("albédo", "albedo"),
        ("télédétection", "remote sensing"),
        ("carbone noir", "black carbon"),
        ("neige", "snow"),
        ("glaciers", "glacier"),
        # Acronyms
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

    # Add a “keyword-only” variant (helps BM25 when query is verbose)
    stop = {
        # EN
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "from", "by", "as", "is", "are",
        "this", "that", "these", "those", "what", "how", "why", "which", "who",
        # FR
        "le", "la", "les", "un", "une", "des", "du", "de", "d", "et", "ou", "dans", "sur", "pour", "avec",
        "par", "en", "au", "aux", "ce", "cet", "cette", "ces", "quoi", "comment", "pourquoi", "quel", "quelle",
    }
    tokens = re.findall(r"[a-z0-9_]+", expanded)
    keywords = [t for t in tokens if t not in stop and len(t) >= 3]
    keyword_variant = " ".join(dict.fromkeys(keywords))  # de-dupe, keep order

    # Build final list (unique, preserve order)
    candidates = [q0, q_norm]
    if expanded and expanded not in candidates:
        candidates.append(expanded)
    if keyword_variant and keyword_variant not in candidates:
        candidates.append(keyword_variant)

    out: list[str] = []
    seen = set()
    for q in candidates:
        q = q.strip()
        if not q:
            continue
        if q in seen:
            continue
        seen.add(q)
        out.append(q)
        if len(out) >= n_queries:
            break

    return out or [q0]


def _multiquery_rrf_fuse(results_by_query: list[list[dict]], rrf_k: int = 60) -> list[dict]:
    """
    Fuse multiple ranked result lists using Reciprocal Rank Fusion over doc ids.
    Keeps the first payload (text/metadata) seen for each id.
    """
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
        item["score"] = float(s)  # overwrite score with multi-query fused score
        merged.append(item)
    return merged


def init_chroma_client():
    """Initialize Chroma client (server mode if available, otherwise local persistent)."""
    global chroma_client

    if not chroma_client:
        chromadb = _get_chromadb()
        # Try HttpClient (server mode) first, fallback to PersistentClient
        try:
            test_client = chromadb.HttpClient(host="localhost", port=8000)
            test_client.heartbeat()
            chroma_client = test_client
            logging.info(f"[OK] MCP: Connected to ChromaDB server (localhost:8000) - Collection: {COLLECTION_NAME}")
        except Exception:
            logging.info(f"[INFO] MCP: ChromaDB server not available, using local mode: {ACTIVE_DB_PATH}")
            chroma_client = chromadb.PersistentClient(path=str(ACTIVE_DB_PATH))

    return chroma_client

def init_voyage_client():
    """Initialize Voyage client (used for semantic embedding)."""
    global voyage_client
    if not voyage_client:
        voyageai = _get_voyageai()
        if not os.getenv("VOYAGE_API_KEY"):
            logging.warning("VOYAGE_API_KEY not set. Semantic search will fail.")
        voyage_client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
    return voyage_client


def init_cohere_client():
    """Initialize Cohere client (used for reranking)."""
    global cohere_client
    if not cohere_client:
        cohere = _get_cohere()
        if not os.getenv("COHERE_API_KEY"):
            logging.warning("COHERE_API_KEY not set. Reranking will fail.")
        cohere_client = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
    return cohere_client


def init_retriever():
    """
    Initialize HybridRetriever.
    IMPORTANT: HybridRetriever BM25 index is lazy (and may build in background),
    so this should be fast and not cause MCP timeouts.
    """
    global hybrid_retriever

    if hybrid_retriever:
        return hybrid_retriever

    try:
        init_chroma_client()
        init_voyage_client()
        init_cohere_client()

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

    return hybrid_retriever


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
    where_document: dict = None,
    multi_query: bool = False,
    n_queries: int = 3,
    format: str = "verbose",
    preview_chars: int | None = None,
    context_window: int | None = None,
) -> str:
    """
    Unified search (contextualized embeddings + BM25 + Cohere rerank).
    """
    try:
        init_retriever()
        
        if not hybrid_retriever:
            return "ERROR: Hybrid retriever not initialized. Check database connection."

        collection = chroma_client.get_collection(name=COLLECTION_NAME)

        # 1. Retrieval (BM25 + contextualized semantic with RRF)
        # Optional: multi-query rewrite/expansion (heuristic), fused with RRF.
        if multi_query:
            queries = _generate_query_variants(query, n_queries=n_queries)
        else:
            queries = [query]

        per_query_results: list[list[dict]] = []
        for q in queries:
            res = hybrid_retriever.search(
                query=q,
                top_k=50,  # candidates per query
                alpha=alpha,
                bm25_top_n=100,
                semantic_top_n=100,
                where=where,
                where_document=where_document,
            )
            if res:
                per_query_results.append(res)

        if not per_query_results:
            return "No results found for your search."

        hybrid_results = per_query_results[0] if len(per_query_results) == 1 else _multiquery_rrf_fuse(per_query_results)
        # Cap rerank candidates (Cohere side) to keep latency/cost bounded
        hybrid_results = hybrid_results[:100]

        if not hybrid_results:
            return "No results found for your search."

        # 2. Prepare for reranking
        documents_for_rerank = [r['text'] for r in hybrid_results]
        metadatas = [r['metadata'] for r in hybrid_results]

        # 3. Rerank with Cohere v4.0 Pro
        rerank_results = cohere_client.rerank(
            model="rerank-v4.0-pro",
            query=query,
            documents=documents_for_rerank,
            top_n=top_k
        )

        # 4. Normalize output controls (presentation-only; retrieval/rerank unchanged)
        output_format = (format or "compact").strip().lower()
        if output_format not in {"compact", "verbose"}:
            output_format = "compact"

        if context_window is None:
            context_window = 0 if output_format == "compact" else CONTEXT_WINDOW_SIZE
        else:
            try:
                context_window = int(context_window)
            except (TypeError, ValueError):
                context_window = 0
            context_window = max(0, context_window)

        if preview_chars is None:
            preview_chars = 120 if output_format == "compact" else 800
        else:
            try:
                preview_chars = int(preview_chars)
            except (TypeError, ValueError):
                preview_chars = 200
            preview_chars = max(0, preview_chars)

        preview_context_chars = max(0, preview_chars // 2)

        # 5. Materialize ranked hits once, then format them (ensures same hits across formats)
        ranked_hits: list[dict] = []
        for result in rerank_results.results:
            idx = result.index
            metadata = metadatas[idx]

            doc_id = hybrid_results[idx].get('id')
            source = metadata.get('source', metadata.get('filename', 'unknown'))
            chunk_index = metadata.get('chunk_index', 0)

            total_chunks = metadata.get('total_chunks')
            try:
                total_chunks = int(total_chunks)
            except (TypeError, ValueError):
                total_chunks = None

            ranked_hits.append({
                "id": doc_id,
                "rerank_score": float(result.relevance_score),
                "fusion_score": float(hybrid_results[idx].get('score', 0.0)),
                "bm25_rank": hybrid_results[idx].get('bm25_rank'),
                "semantic_rank": hybrid_results[idx].get('semantic_rank'),
                "source": source,
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
                "text": documents_for_rerank[idx] if idx < len(documents_for_rerank) else "",
                "metadata": metadata,
            })

        output = f"SEARCH RESULTS ({RAGDOC_MODE.upper()} MODE): {query}\n"
        if multi_query and len(queries) > 1:
            output += f"[i] Multi-query enabled ({len(queries)} variants)\n"
        output += "=" * 70 + "\n\n"

        if output_format == "compact":
            for i, hit in enumerate(ranked_hits, 1):
                line = f"[{i}] rerank={hit['rerank_score']:.4f} fusion={hit['fusion_score']:.4f}"
                line += f" source={hit['source']}"
                if hit.get('total_chunks') is not None:
                    line += f" chunk={hit['chunk_index']}/{hit['total_chunks']}"
                else:
                    line += f" chunk={hit['chunk_index']}"
                if hit.get('id'):
                    line += f" id={hit['id']}"

                bm25_rank = hit.get('bm25_rank')
                semantic_rank = hit.get('semantic_rank')
                if bm25_rank is not None or semantic_rank is not None:
                    line += f" bm25_rank={bm25_rank} semantic_rank={semantic_rank}"

                output += line + "\n"

                text = hit.get('text') or ""
                text = re.sub(r"\s+", " ", text).strip()
                if preview_chars > 0:
                    preview = text[:preview_chars]
                    if len(text) > preview_chars:
                        preview += "..."
                    output += f"    {preview}\n\n"
                else:
                    output += "\n"

            return output

        # verbose output (backward-compatible) with optional context control
        doc_cache = {}
        for i, hit in enumerate(ranked_hits, 1):
            source = hit['source']
            chunk_index = hit['chunk_index']

            total_chunks = hit.get('total_chunks')
            if total_chunks is None:
                doc_entry = _get_document_cache_entry(collection, source, doc_cache)
                meta_list = doc_entry.get('metadatas') if doc_entry else []
                total_chunks = len(meta_list) if meta_list else 1

            output += f"[{i}] Rerank Score: {hit['rerank_score']:.4f} | Fusion: {hit['fusion_score']:.4f}\n"
            output += f"    Source: {source}\n"
            output += f"    Position: chunk {chunk_index}/{total_chunks}\n"
            output += f"    Rankings: BM25 #{hit.get('bm25_rank')}, Semantic #{hit.get('semantic_rank')}\n"
            if hit.get('id'):
                output += f"    Chunk ID: {hit['id']}\n"
            output += "\n"

            adjacent_chunks = _get_adjacent_chunks(
                collection,
                source,
                chunk_index,
                total_chunks,
                window_size=context_window,
                doc_cache=doc_cache,
            )

            if adjacent_chunks:
                for chunk_content, chunk_meta in adjacent_chunks:
                    is_main = chunk_meta['chunk_index'] == chunk_index
                    marker = "[*]" if is_main else "[ ]"

                    output += f"{marker} [Chunk {chunk_meta['chunk_index']}]"
                    if is_main:
                        output += " <-- MAIN RESULT"
                    output += "\n"

                    preview_length = preview_chars if is_main else preview_context_chars
                    preview = chunk_content[:preview_length]
                    if len(chunk_content) > preview_length:
                        preview += "..."
                    output += f"    {preview}\n\n"
            else:
                fallback = (hit.get('text') or "")[:200]
                if len(hit.get('text') or "") > 200:
                    fallback += "..."
                output += f"    Content: {fallback}\n\n"

            output += "-" * 70 + "\n\n"

        return output

    except Exception as e:
        logging.exception("Error during hybrid search")
        return f"ERROR: {str(e)}"


@mcp.tool()
def semantic_search_hybrid(
    query: str,
    top_k: int = 10,
    alpha: float = 0.5,
    multi_query: bool = False,
    n_queries: int = 3,
    format: str = "compact",
    preview_chars: int | None = None,
    context_window: int | None = None,
) -> str:
    """
    Hybrid search with BM25 + Vector + Cohere v3.5 reranking.

    Args:
        query: Search query about the indexed knowledge base.
        top_k: Number of results to return (default: 10)
        alpha: Semantic weight (0.5 = balanced hybrid). Use 0.3 for BM25-heavy, 0.7 for semantic-heavy.
        multi_query: If True, generate multiple query variants (rewrite/expansion) and fuse results (default: False).
        n_queries: Total number of query variants to use INCLUDING the original (1-5, default: 3).
        format: Output format ("compact" or "verbose").
        preview_chars: Character count for the main snippet/preview (defaults: 200 compact, 800 verbose).
        context_window: Adjacent chunks to include on each side in verbose mode (defaults: 0 compact, CONTEXT_WINDOW_SIZE verbose).

    Returns:
        Formatted search results with hybrid ranking scores and source information.
    """
    return _perform_search_hybrid(
        query=query,
        top_k=top_k,
        alpha=alpha,
        multi_query=multi_query,
        n_queries=n_queries,
        format=format,
        preview_chars=preview_chars,
        context_window=context_window,
    )


@mcp.tool()
def search_by_source(
    query: str,
    sources: list,
    top_k: int = 10,
    alpha: float = 0.5,
    multi_query: bool = False,
    n_queries: int = 3,
    format: str = "compact",
    preview_chars: int | None = None,
    context_window: int | None = None,
) -> str:
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

    return _perform_search_hybrid(
        query=query,
        top_k=top_k,
        alpha=alpha,
        where=where,
        multi_query=multi_query,
        n_queries=n_queries,
        format=format,
        preview_chars=preview_chars,
        context_window=context_window,
    )


@mcp.tool()
def list_documents() -> str:
    """
    List all indexed documents.

    Returns:
        List of available papers with metadata.
    """
    try:
        init_chroma_client()
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
        init_chroma_client()
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
        init_chroma_client()
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
        init_chroma_client()
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
    
    # IMPORTANT: keep stdout clean (JSON-RPC only) for stdio MCP clients.
    # FastMCP banner (if printed) can break some clients, so disable it.
    mcp.run(show_banner=False)


if __name__ == "__main__":
    main()
