#!/usr/bin/env python3
"""
Hybrid Retriever: BM25 + Semantic Search + Reciprocal Rank Fusion
Pour ChromaDB (qui n'a pas de hybrid search natif)
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, TYPE_CHECKING
from collections import defaultdict
from rank_bm25 import BM25Okapi
import threading

if TYPE_CHECKING:
    import chromadb

logger = logging.getLogger(__name__)

try:
    # New relative import
    from .bm25_tokenizers.advanced_tokenizer import AdvancedTokenizer
except ImportError:
    # Fallback for running as script
    try:
        from bm25_tokenizers.advanced_tokenizer import AdvancedTokenizer
    except ImportError:
        AdvancedTokenizer = None

class HybridRetriever:
    """
    Hybrid retrieval combining BM25 (lexical) + Vector (semantic)

    Usage:
        retriever = HybridRetriever(chroma_collection)
        results = retriever.search("black carbon albedo", top_k=10)
    """

    def __init__(self, collection: "chromadb.Collection", embedding_function=None,
                 use_advanced_tokenizer: bool = True):
        """
        Args:
            collection: ChromaDB collection
            embedding_function: Function to embed queries (e.g., voyage_client.embed)
            use_advanced_tokenizer: If True, use advanced tokenization with stemming,
                                   stopwords removal, and n-grams (default: True)
                                   If False, use simple tokenization (backward compatible)
        """
        self.collection = collection
        self.embedding_function = embedding_function

        # Initialize tokenizer
        self.tokenizer = None
        if use_advanced_tokenizer and AdvancedTokenizer:
            try:
                self.tokenizer = AdvancedTokenizer()
                logger.info("[OK] Advanced tokenization enabled (stemming + stopwords + scientific terms)")
            except Exception as e:
                logger.warning(f"Advanced tokenizer initialization failed ({e}), using simple tokenization")
                self.tokenizer = None
        else:
            logger.info("Simple tokenization mode (backward compatible)")

        # BM25 index is heavy to build on large corpora (can exceed MCP client timeouts).
        # We build it lazily on first need and (by default) in the background.
        self.docs: List[str] = []
        self.ids: List[str] = []
        self.metadatas: List[dict] = []
        self.bm25: BM25Okapi | None = None
        self._id_to_idx: dict[str, int] = {}
        self._bm25_lock = threading.Lock()
        self._bm25_building = False

    def _build_bm25_index(self):
        """Build BM25 index from ChromaDB collection"""
        # Fetch all documents from ChromaDB
        # TODO: Optimize for large collections (lazy loading or caching)
        all_data = self.collection.get(include=["documents", "metadatas"])

        self.docs = all_data['documents']
        self.ids = all_data['ids']
        self.metadatas = all_data['metadatas']
        self._id_to_idx = {doc_id: i for i, doc_id in enumerate(self.ids)}

        # Tokenize corpus for BM25
        # Simple whitespace tokenization (can be improved with stemming/lemmatization)
        tokenized_corpus = [self._tokenize(doc) for doc in self.docs]

        # Initialize BM25
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _build_bm25_index_worker(self) -> None:
        """Worker that builds the BM25 index and clears the building flag."""
        try:
            logger.info("Building BM25 index (lazy init)...")
            self._build_bm25_index()
            logger.info(f"BM25 index built: {len(self.docs)} documents")
        except Exception as e:
            logger.exception(f"BM25 index build failed: {e}")
            # Leave bm25 as None
        finally:
            with self._bm25_lock:
                self._bm25_building = False

    def ensure_bm25_index(self, background: bool = True) -> bool:
        """
        Ensure BM25 index is available.

        Returns:
            True if BM25 is ready now, False if build is in progress or failed.
        """
        if self.bm25 is not None:
            return True

        with self._bm25_lock:
            if self.bm25 is not None:
                return True
            if self._bm25_building:
                return False
            self._bm25_building = True

        if background:
            t = threading.Thread(target=self._build_bm25_index_worker, daemon=True)
            t.start()
            return False

        # Blocking build (may take time on large corpora)
        self._build_bm25_index_worker()
        return self.bm25 is not None

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text using advanced or simple tokenizer.
        """
        if self.tokenizer:
            # Advanced tokenization
            return self.tokenizer.tokenize(text)
        else:
            # Fallback: simple tokenization
            return text.lower().split()

    def search(
        self,
        query: str,
        top_k: int = 10,
        alpha: float = 0.5,
        bm25_top_n: int = 100,
        semantic_top_n: int = 100,
        rrf_k: int = 60,
        where: dict = None,
        where_document: dict = None
    ) -> List[Dict]:
        """
        Hybrid search with Reciprocal Rank Fusion

        Args:
            query: Search query
            top_k: Number of final results
            alpha: Weight for semantic (0.5 = equal BM25/semantic)
            bm25_top_n: Number of BM25 candidates
            semantic_top_n: Number of semantic candidates
            rrf_k: RRF constant (typically 60)
            where: Optional metadata filter (e.g., {"source": "doc.md"})
            where_document: Optional document content filter (e.g., {"$contains": "text"})

        Returns:
            List of dicts with keys: id, text, metadata, score, ranks
        """

        # 1. BM25 search (optional / lazy)
        bm25_results: List[Tuple[str, float, int]] = []
        if alpha < 1.0:
            if self.bm25 is None:
                # If user asked for pure BM25, we must build synchronously.
                # Otherwise start in background and proceed with semantic-only results.
                if alpha == 0.0:
                    self.ensure_bm25_index(background=False)
                else:
                    self.ensure_bm25_index(background=True)
            if self.bm25 is not None:
                bm25_results = self._bm25_search(
                    query,
                    top_n=bm25_top_n,
                    where=where,
                    where_document=where_document,
                )

        # 2. Semantic search (ChromaDB) with filtering
        semantic_results, semantic_payload = self._semantic_search(
            query,
            top_n=semantic_top_n,
            where=where,
            where_document=where_document
        )

        # 3. Reciprocal Rank Fusion
        fused_results = self._reciprocal_rank_fusion(
            bm25_results,
            semantic_results,
            k=rrf_k,
            alpha=alpha,
            semantic_payload=semantic_payload
        )

        return fused_results[:top_k]

    def _bm25_search(
        self,
        query: str,
        top_n: int,
        where: dict = None,
        where_document: dict = None,
    ) -> List[Tuple[str, float, int]]:
        """
        BM25 search

        Returns:
            List of (doc_id, bm25_score, rank)
        """
        if self.bm25 is None:
            return []

        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Apply optional filtering (metadata/content) to keep BM25 consistent with Chroma queries.
        eligible_indices = list(range(len(bm25_scores)))
        if where is not None or where_document is not None:
            eligible_indices = []
            for idx, (doc, meta) in enumerate(zip(self.docs, self.metadatas)):
                if where is not None and not self._match_where(meta, where):
                    continue
                if where_document is not None and not self._match_where_document(doc, where_document):
                    continue
                eligible_indices.append(idx)

        if not eligible_indices:
            return []

        # Get top-N by score within eligible indices
        actual_top_n = min(top_n, len(eligible_indices))
        scores_subset = np.asarray(bm25_scores)[eligible_indices]
        top_local = np.argsort(scores_subset)[::-1][:actual_top_n]

        results: List[Tuple[str, float, int]] = []
        for rank, local_idx in enumerate(top_local):
            global_idx = eligible_indices[int(local_idx)]
            doc_id = self.ids[global_idx]
            score = float(scores_subset[int(local_idx)])
            results.append((doc_id, score, rank))

        return results

    def _match_where(self, metadata: dict, where: dict) -> bool:
        """
        Best-effort evaluator for a subset of Chroma `where` syntax.
        Supports:
          - {"field": "value"} (equality)
          - {"field": {"$eq": value}}
          - {"field": {"$in": [v1, v2, ...]}}
          - {"$and": [cond1, cond2, ...]}
          - {"$or":  [cond1, cond2, ...]}
        """
        if where is None:
            return True
        if metadata is None:
            return False

        if "$and" in where:
            clauses = where.get("$and") or []
            return all(self._match_where(metadata, clause) for clause in clauses)
        if "$or" in where:
            clauses = where.get("$or") or []
            return any(self._match_where(metadata, clause) for clause in clauses)

        for key, condition in where.items():
            if key in ("$and", "$or"):
                continue

            value = metadata.get(key)

            # Direct equality
            if not isinstance(condition, dict):
                if value != condition:
                    return False
                continue

            # Operator dict
            if "$eq" in condition:
                if value != condition["$eq"]:
                    return False
            elif "$in" in condition:
                allowed = condition.get("$in") or []
                if value not in allowed:
                    return False
            else:
                # Unknown operator → best effort: fail closed to avoid leaking out-of-scope docs
                return False

        return True

    def _match_where_document(self, document: str, where_document: dict) -> bool:
        """
        Best-effort evaluator for Chroma `where_document`.
        Supports:
          - {"$contains": "text"}
          - {"$not_contains": "text"}
        """
        if where_document is None:
            return True
        if not document:
            return False

        if "$contains" in where_document:
            needle = where_document.get("$contains")
            if needle is None:
                return True
            return str(needle).lower() in document.lower()
        if "$not_contains" in where_document:
            needle = where_document.get("$not_contains")
            if needle is None:
                return True
            return str(needle).lower() not in document.lower()

        # Unknown operator: fail closed
        return False

    def _semantic_search(
        self,
        query: str,
        top_n: int,
        where: dict = None,
        where_document: dict = None
    ) -> Tuple[List[Tuple[str, float, int]], Dict[str, Tuple[str, dict]]]:
        """
        Semantic search via ChromaDB
        """
        if self.embedding_function is None:
            raise ValueError("embedding_function required for semantic search")

        # Embed query
        query_embedding = self.embedding_function([query])[0]

        # Query ChromaDB with optional filters
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_n,
            where=where,
            where_document=where_document,
            include=["documents", "metadatas", "distances"]
        )
        
        # Check if results are empty
        if not results['ids'] or not results['ids'][0]:
            return [], {}

        ids = results['ids'][0]
        distances = results['distances'][0]
        documents = results.get('documents', [[]])[0] or []
        metadatas = results.get('metadatas', [[]])[0] or []

        # Format results
        semantic_results: List[Tuple[str, float, int]] = []
        semantic_payload: Dict[str, Tuple[str, dict]] = {}

        for rank, (doc_id, distance) in enumerate(zip(ids, distances)):
            # Convert distance to similarity (cosine distance → similarity)
            similarity = 1 - distance
            semantic_results.append((doc_id, similarity, rank))
            # Payload (best effort)
            if rank < len(documents) and rank < len(metadatas):
                semantic_payload[doc_id] = (documents[rank], metadatas[rank])

        return semantic_results, semantic_payload

    def _reciprocal_rank_fusion(
        self,
        bm25_results: List[Tuple[str, float, int]],
        semantic_results: List[Tuple[str, float, int]],
        k: int = 60,
        alpha: float = 0.5,
        semantic_payload: Dict[str, Tuple[str, dict]] | None = None
    ) -> List[Dict]:
        """
        Reciprocal Rank Fusion (RRF)

        Score(doc) = alpha * RRF_semantic + (1-alpha) * RRF_bm25
        RRF(doc) = sum(1 / (k + rank))
        """

        # Calculate RRF scores
        rrf_scores = defaultdict(lambda: {'bm25': 0, 'semantic': 0, 'combined': 0})

        # BM25 contribution
        for doc_id, bm25_score, rank in bm25_results:
            rrf_scores[doc_id]['bm25'] = 1 / (k + rank + 1)
            rrf_scores[doc_id]['bm25_raw'] = bm25_score
            rrf_scores[doc_id]['bm25_rank'] = rank

        # Semantic contribution
        for doc_id, sem_score, rank in semantic_results:
            rrf_scores[doc_id]['semantic'] = 1 / (k + rank + 1)
            rrf_scores[doc_id]['semantic_raw'] = sem_score
            rrf_scores[doc_id]['semantic_rank'] = rank

        # Combined score (weighted)
        for doc_id in rrf_scores:
            rrf_scores[doc_id]['combined'] = (
                alpha * rrf_scores[doc_id]['semantic'] +
                (1 - alpha) * rrf_scores[doc_id]['bm25']
            )

        # Sort by combined score
        sorted_results = sorted(
            rrf_scores.items(),
            key=lambda x: x[1]['combined'],
            reverse=True
        )

        # Format output
        final_results = []
        for doc_id, scores in sorted_results:
            # Get document text and metadata
            text = None
            metadata = None

            # Prefer semantic payload (available even when BM25 index isn't built)
            if semantic_payload and doc_id in semantic_payload:
                text, metadata = semantic_payload[doc_id]

            # Fast path: cached BM25 index payload
            if text is None or metadata is None:
                idx = self._id_to_idx.get(doc_id)
                if idx is not None and idx < len(self.docs) and idx < len(self.metadatas):
                    text = self.docs[idx]
                    metadata = self.metadatas[idx]

            # Fallback: fetch from collection
            if text is None or metadata is None:
                try:
                    result = self.collection.get(
                        ids=[doc_id],
                        include=["documents", "metadatas"]
                    )
                    if result.get('documents'):
                        text = result['documents'][0]
                        metadata = result['metadatas'][0]
                    else:
                        continue
                except Exception:
                    continue

            final_results.append({
                'id': doc_id,
                'text': text,
                'metadata': metadata,
                'score': scores['combined'],
                'bm25_score': scores.get('bm25_raw', 0),
                'semantic_score': scores.get('semantic_raw', 0),
                'bm25_rank': scores.get('bm25_rank', None),
                'semantic_rank': scores.get('semantic_rank', None)
            })

        return final_results
