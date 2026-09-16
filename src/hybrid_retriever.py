#!/usr/bin/env python3
"""
Hybrid Retriever: BM25 + Semantic Search + Reciprocal Rank Fusion
Pour ChromaDB (qui n'a pas de hybrid search natif)
"""

import logging
import os
import numpy as np
from typing import List, Dict, Tuple, TYPE_CHECKING
from collections import defaultdict
from rank_bm25 import BM25Okapi
import threading
from contextvars import ContextVar
try:
    from .chroma_reads import read_collection
except ImportError:
    from chroma_reads import read_collection

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
                 use_advanced_tokenizer: bool = True, revision_provider=None,
                 lexical_index=None):
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
        self.revision_provider = revision_provider or (lambda: self.collection.metadata or {})
        self.lexical_index = lexical_index
        self._revision = None
        self._search_lock = threading.RLock()
        self._status = ContextVar(f"retrieval_status_{id(self)}", default={})

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
        self._token_sets = []
        self._bm25_max_chunks = int(os.getenv("RAGDOC_BM25_MAX_CHUNKS", "50000"))
        self._bm25_disabled_reason = None

    def _build_bm25_index(self):
        """Build BM25 index from ChromaDB collection"""
        # Fetch all documents from ChromaDB
        # TODO: Optimize for large collections (lazy loading or caching)
        all_data = read_collection(self.collection, include=["documents", "metadatas"])

        self.docs = all_data['documents']
        self.ids = all_data['ids']
        self.metadatas = all_data['metadatas']
        self._id_to_idx = {doc_id: i for i, doc_id in enumerate(self.ids)}

        # Tokenize corpus for BM25
        # Simple whitespace tokenization (can be improved with stemming/lemmatization)
        tokenized_corpus = [self._tokenize(doc) for doc in self.docs]
        self._token_sets = [set(tokens) for tokens in tokenized_corpus]

        # Initialize BM25
        self.bm25 = BM25Okapi(tokenized_corpus) if any(tokenized_corpus) else None

    @property
    def last_status(self):
        return dict(self._status.get())

    def _check_revision(self):
        metadata = self.revision_provider() or {}
        if metadata.get("ragdoc_write_state", "ready") != "ready" or metadata.get("ragdoc_repairing", False):
            raise RuntimeError("Index write incomplete or in progress; retry after indexing/repair")
        revision = metadata.get("ragdoc_revision")
        if revision != self._revision:
            self.bm25 = None
            self.docs, self.ids, self.metadatas, self._token_sets = [], [], [], []
            self._id_to_idx = {}
            self._revision = revision
        return revision

    def _build_bm25_index_worker(self) -> None:
        """Worker that builds the BM25 index and clears the building flag."""
        try:
            logger.info("Building BM25 index (lazy init)...")
            with self._search_lock:
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
        if self._bm25_disabled_reason is not None:
            return False

        chunk_count = self.collection.count()
        if chunk_count > self._bm25_max_chunks:
            self._bm25_disabled_reason = (
                f"corpus has {chunk_count} chunks; safe limit is {self._bm25_max_chunks}"
            )
            logger.warning(
                "BM25 disabled to protect server memory: %s. Semantic search remains available.",
                self._bm25_disabled_reason,
            )
            return False

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

    def search(self, query: str, top_k: int = 10, alpha: float = 0.5,
               bm25_top_n: int = 100, semantic_top_n: int = 100, rrf_k: int = 60,
               where: dict = None, where_document: dict = None) -> List[Dict]:
        if not query or not query.strip():
            raise ValueError("query must not be empty")
        if not 0 <= alpha <= 1 or not 1 <= top_k <= 1000:
            raise ValueError("alpha must be in [0, 1] and internal top_k in [1, 1000]")
        with self._search_lock:
            revision = self._check_revision()
            result = self._search(query, top_k, alpha, bm25_top_n, semantic_top_n, rrf_k, where, where_document)
            if self._check_revision() != revision:
                raise RuntimeError("Index changed during search; retry against the new version")
            return result

    def _search(
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

        warnings = []
        bm25_results: List[Tuple[str, float, int]] = []
        lexical_payload: Dict[str, Tuple[str, dict]] = {}
        lexical_ready = False
        if alpha < 1.0:
            bm25_results, lexical_payload, lexical_ready = self._lexical_search(
                query, bm25_top_n, where, where_document
            )
            if not lexical_ready:
                warnings.append("lexical_unavailable")

        # 2. Semantic search (ChromaDB) with filtering
        semantic_results, semantic_payload = [], {}
        semantic_available = False
        effective_alpha = alpha
        if alpha > 0:
            try:
                semantic_results, semantic_payload = self._semantic_search(
                    query, top_n=semantic_top_n, where=where, where_document=where_document)
                semantic_available = True
            except Exception:
                logger.warning("Semantic search unavailable; falling back to lexical search", exc_info=True)
                warnings.append("semantic_unavailable")
                if not lexical_ready:
                    bm25_results, lexical_payload, lexical_ready = self._lexical_search(
                        query, bm25_top_n, where, where_document
                    )
                effective_alpha = 0.0
        if not lexical_ready and not semantic_available:
            if self.collection.count() > 0:
                raise RuntimeError("No search channel available")
        if alpha < 1 and not lexical_ready:
            effective_alpha = 1.0
        lexical_available = lexical_ready and effective_alpha < 1
        self._status.set({"mode": "hybrid" if semantic_available and lexical_available else
                          "semantic" if semantic_available else "lexical",
                          "bm25_ready": self.bm25 is not None,
                          "persistent_lexical_ready": bool(self.lexical_index) and lexical_ready,
                          "tokenizer": "advanced" if self.tokenizer else "simple",
                          "index_revision": self._revision, "warnings": warnings})

        # 3. Reciprocal Rank Fusion
        fused_results = self._reciprocal_rank_fusion(
            bm25_results,
            semantic_results,
            k=rrf_k,
            alpha=effective_alpha,
            semantic_payload=semantic_payload,
            lexical_payload=lexical_payload,
        )

        return fused_results[:top_k]

    def _lexical_search(self, query: str, top_n: int, where: dict, where_document: dict):
        """Use the persistent FTS sidecar, retaining in-memory BM25 for small tests."""
        if self.lexical_index is not None:
            try:
                exact_source = (
                    where.get("source")
                    if isinstance(where, dict) and set(where) == {"source"}
                    else None
                )
                results, payload = self.lexical_index.search(
                    query,
                    top_n=top_n,
                    revision=self._revision,
                    chunk_count=self.collection.count(),
                    source=exact_source,
                )
                if where is not None or where_document is not None:
                    filtered = []
                    for doc_id, score, _ in results:
                        text, metadata = payload[doc_id]
                        if where is not None and not self._match_where(metadata, where):
                            continue
                        if where_document is not None and not self._match_where_document(text, where_document):
                            continue
                        filtered.append((doc_id, score, len(filtered)))
                    results = filtered
                return results, payload, True
            except Exception:
                logger.warning("Persistent lexical search unavailable", exc_info=True)
                return [], {}, False

        if self.bm25 is None:
            self.ensure_bm25_index(background=False)
        if self.bm25 is None:
            return [], {}, False
        return self._bm25_search(query, top_n, where, where_document), {}, True

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
        query_tokens = set(tokenized_query)
        eligible_indices = [i for i, tokens in enumerate(self._token_sets) if query_tokens.intersection(tokens)]
        if where is not None or where_document is not None:
            filtered = []
            for idx in eligible_indices:
                doc, meta = self.docs[idx], self.metadatas[idx]
                if where is not None and not self._match_where(meta, where):
                    continue
                if where_document is not None and not self._match_where_document(doc, where_document):
                    continue
                filtered.append(idx)
            eligible_indices = filtered

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
            elif "$ne" in condition:
                if value == condition["$ne"]:
                    return False
            elif "$nin" in condition:
                if value in condition["$nin"]:
                    return False
            elif any(op in condition for op in ("$gt", "$gte", "$lt", "$lte")):
                if value is None:
                    return False
                try:
                    for op, target in condition.items():
                        matches = {"$gt": lambda: value > target, "$gte": lambda: value >= target,
                                   "$lt": lambda: value < target, "$lte": lambda: value <= target}
                        if op not in matches or not matches[op]():
                            return False
                except TypeError:
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
            return str(needle) in document
        if "$not_contains" in where_document:
            needle = where_document.get("$not_contains")
            if needle is None:
                return True
            return str(needle) not in document

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
        semantic_payload: Dict[str, Tuple[str, dict]] | None = None,
        lexical_payload: Dict[str, Tuple[str, dict]] | None = None,
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

            if (text is None or metadata is None) and lexical_payload and doc_id in lexical_payload:
                text, metadata = lexical_payload[doc_id]

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
