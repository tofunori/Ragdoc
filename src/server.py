#!/usr/bin/env python3
"""
MCP Server for Glacier Research Search
Voyage-Context-3 embeddings + Cohere v3.5 reranking
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
mcp = FastMCP("ragdoc")

# Global clients (initialized on first use)
voyage_client = None
chroma_client = None
cohere_client = None


def init_clients():
    """Initialize API clients with auto-detection of ChromaDB server"""
    global voyage_client, chroma_client, cohere_client

    if not voyage_client:
        voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)

    if not chroma_client:
        # Try HttpClient (server mode) first, fallback to PersistentClient
        try:
            test_client = chromadb.HttpClient(host="localhost", port=8000)
            test_client.heartbeat()
            chroma_client = test_client
            logging.info("[OK] MCP: Connecte au serveur ChromaDB (localhost:8000)")
        except Exception as e:
            logging.info("[INFO] MCP: Serveur ChromaDB non disponible, utilisation du mode local")
            chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_HYBRID_PATH))

    if not cohere_client:
        cohere_client = cohere.ClientV2(api_key=COHERE_API_KEY)


def _compute_doc_hash(content: str) -> str:
    """Compute MD5 hash of document content"""
    return hashlib.md5(content.encode()).hexdigest()


def _chunk_markdown(content: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    """Split markdown content into overlapping chunks"""
    chunks = []
    start = 0
    while start < len(content):
        end = min(start + chunk_size, len(content))
        chunk = content[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start >= len(content) - overlap:
            break
    return chunks


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
                "Echec recuperation chunks pour %s=%s : %s",
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
            "Erreur lors de la recuperation des chunks adjacents pour %s: %s",
            source,
            e
        )
        return []


def _perform_search(query: str, top_k: int = 10) -> str:
    """
    Internal search implementation (shared by multiple tools)
    Voyage-Context-3 + Cohere v3.5 reranking pipeline with context window expansion
    """
    try:
        init_clients()

        # 1. Embed query with Voyage-Context-3
        query_result = voyage_client.contextualized_embed(
            inputs=[[query]],
            model="voyage-context-3",
            input_type="query"
        )
        query_embedding = query_result.results[0].embeddings[0]

        # 2. Search Chroma (top-50 candidates)
        collection = chroma_client.get_collection(name=COLLECTION_HYBRID_NAME)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=50
        )

        if not results or not results['documents'] or len(results['documents'][0]) == 0:
            return "Aucun resultat trouve pour votre recherche."

        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        doc_cache: dict[str, dict] = {}

        # 3. Rerank with Cohere v3.5
        rerank_results = cohere_client.rerank(
            model="rerank-v3.5",
            query=query,
            documents=documents,
            top_n=top_k
        )

        # 4. Format results with context window expansion
        output = f"RESULTATS DE RECHERCHE: {query}\n"
        output += "=" * 70 + "\n\n"

        for i, result in enumerate(rerank_results.results, 1):
            idx = result.index
            score = result.relevance_score
            metadata = metadatas[idx]
            # Utiliser 'source' en priorité, puis 'filename' pour compatibilité
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

            output += f"[{i}] Score: {score:.4f}\n"
            output += f"    Source: {source}\n"
            output += f"    Position: chunk {chunk_index}/{total_chunks}\n\n"

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
                        output += " <-- RESULTAT PRINCIPAL"
                    output += "\n"

                    # Show content preview
                    preview_length = 300 if is_main else 150
                    preview = chunk_content[:preview_length]
                    if len(chunk_content) > preview_length:
                        preview += "..."
                    output += f"    {preview}\n\n"
            else:
                # Fallback if adjacent chunks not found
                output += f"    Contenu: {documents[idx][:200]}...\n\n"

            output += "-" * 70 + "\n\n"

        return output

    except Exception as e:
        return f"ERREUR: {str(e)}"


@mcp.tool()
def semantic_search(query: str, top_k: int = 10) -> str:
    """
    Search glacier research papers with Voyage-Context-3 + Cohere v3.5 reranking.

    Args:
        query: Search query about the indexed knowledge base.
        top_k: Number of results to return (default: 10)

    Returns:
        Formatted search results with relevance scores and source information.
    """
    return _perform_search(query, top_k)


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
            # Utiliser 'filename' au lieu de 'source' pour la base hybride
            source = metadata.get('source', metadata.get('filename', 'unknown'))
            if source not in sources:
                sources[source] = metadata

        output = f"DOCUMENTS INDEXES: {len(sources)} papiers\n"
        output += "=" * 70 + "\n\n"

        for i, (source, metadata) in enumerate(sorted(sources.items()), 1):
            output += f"[{i}] {source}\n"
            output += f"    Titre: {metadata.get('title', 'Sans titre')}\n"
            output += f"    Chunks: {metadata.get('total_chunks', 'N/A')}\n\n"

        return output

    except Exception as e:
        return f"ERREUR: {str(e)}"


@mcp.tool()
def topic_search(topic: str, top_k: int = 5) -> str:
    """
    Quick search by topic shortcut.

    Args:
        topic: Research topic (e.g., "black carbon", "snow albedo", "glacier melt")
        top_k: Number of results to return

    Returns:
        Top papers for the topic with relevance scores.
    """

    # Expand topic to full query
    topic_queries = {
        "black carbon": "black carbon soot impurities glacier albedo",
        "snow albedo": "snow surface albedo reflectance glaciers",
        "glacier melt": "glacier melt ablation surface energy budget",
        "dust": "mineral dust impurities glacier darkening",
        "cryoconite": "cryoconite holes algae glacier surface",
        "remote sensing": "remote sensing satellite albedo glaciers"
    }

    query = topic_queries.get(topic.lower(), topic)
    return _perform_search(query, top_k=top_k)


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
            # Utiliser 'filename' au lieu de 'source' pour la base hybride
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

        output = "ETAT DE L'INDEXATION - CHROMA DB\n"
        output += "=" * 70 + "\n\n"

        output += f"STATISTIQUES GLOBALES:\n"
        output += f"   Nombre de documents: {total_docs}\n"
        output += f"   Nombre total de chunks: {total_chunks}\n"
        if total_docs > 0:
            output += f"   Moyenne chunks/doc: {total_chunks / total_docs:.1f}\n\n"

        docs_with_hash = sum(1 for d in docs_by_source.values() if d['hash'])
        docs_with_date = sum(1 for d in docs_by_source.values() if d['indexed_date'])

        output += f"VERIFICATION DES METADONNEES:\n"
        output += f"   Documents avec hash MD5: {docs_with_hash}/{total_docs}\n"
        output += f"   Documents avec date: {docs_with_date}/{total_docs}\n\n"

        models = {}
        for doc in docs_by_source.values():
            model = doc['model'] or 'unknown'
            models[model] = models.get(model, 0) + 1

        output += f"REPARTITION PAR MODELE:\n"
        for model, count in sorted(models.items()):
            pct = (count / total_docs) * 100
            output += f"   {model:30} {count:3d} docs ({pct:5.1f}%)\n"

        output += f"\n" + "=" * 70 + "\n"
        if docs_with_hash == total_docs and docs_with_date > 0:
            output += "STATUS: READY FOR PRODUCTION"
        else:
            output += "STATUS: ATTENTION - Missing metadata"
        output += "\n" + "=" * 70

        return output

    except Exception as e:
        return f"ERREUR: {str(e)}"


@mcp.tool()
def reindex_documents(force: bool = False) -> str:
    """
    Trigger document reindexation from the MCP server.

    Args:
        force: If True, force complete reindex (remove duplicates). If False, only add missing documents.

    Returns:
        Indexation report with document and chunk statistics.
    """
    try:
        import subprocess
        init_clients()

        script_path = Path(__file__).parent.parent / "index_hybrid_collection.py"

        cmd = [
            "python",
            str(script_path)
        ]

        if force:
            cmd.append("--force")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode == 0:
            output = "REINDEXATION EFFECTUEE\n"
            output += "=" * 70 + "\n\n"
            output += result.stdout
            output += "\n" + "=" * 70
            output += "\nPour verifier l'etat, utilisez: get_indexation_status()"
            return output
        else:
            return f"ERREUR lors de la reindexation:\n{result.stderr}"

    except subprocess.TimeoutExpired:
        return "ERREUR: Reindexation timeout (> 5 minutes)"
    except Exception as e:
        return f"ERREUR: {str(e)}"


if __name__ == "__main__":
    mcp.run()
