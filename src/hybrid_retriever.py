#!/usr/bin/env python3
"""
Hybrid Retriever: BM25 + Semantic Search + Reciprocal Rank Fusion
Pour ChromaDB (qui n'a pas de hybrid search natif)
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
from rank_bm25 import BM25Okapi
import chromadb


class HybridRetriever:
    """
    Hybrid retrieval combining BM25 (lexical) + Vector (semantic)

    Usage:
        retriever = HybridRetriever(chroma_collection)
        results = retriever.search("black carbon albedo", top_k=10)
    """

    def __init__(self, collection: chromadb.Collection, embedding_function=None):
        """
        Args:
            collection: ChromaDB collection
            embedding_function: Function to embed queries (e.g., voyage_client.embed)
        """
        self.collection = collection
        self.embedding_function = embedding_function

        # Build BM25 index
        print("Building BM25 index...")
        self._build_bm25_index()
        print(f"BM25 index built: {len(self.docs)} documents")

    def _build_bm25_index(self):
        """Build BM25 index from ChromaDB collection"""
        # Fetch all documents from ChromaDB
        all_data = self.collection.get(include=["documents", "metadatas"])

        self.docs = all_data['documents']
        self.ids = all_data['ids']
        self.metadatas = all_data['metadatas']

        # Tokenize corpus for BM25
        # Simple whitespace tokenization (can be improved with stemming/lemmatization)
        tokenized_corpus = [self._tokenize(doc) for doc in self.docs]

        # Initialize BM25
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization (lowercase + split)

        TODO: Améliorer avec:
        - Stemming (SnowballStemmer)
        - Stopwords removal
        - N-grams pour termes composés
        """
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

        # 1. BM25 search
        bm25_results = self._bm25_search(query, top_n=bm25_top_n)

        # 2. Semantic search (ChromaDB) with filtering
        semantic_results = self._semantic_search(
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
            alpha=alpha
        )

        return fused_results[:top_k]

    def _bm25_search(self, query: str, top_n: int) -> List[Tuple[str, float, int]]:
        """
        BM25 search

        Returns:
            List of (doc_id, bm25_score, rank)
        """
        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Get top-N by score
        top_indices = np.argsort(bm25_scores)[::-1][:top_n]

        results = []
        for rank, idx in enumerate(top_indices):
            doc_id = self.ids[idx]
            score = bm25_scores[idx]
            results.append((doc_id, score, rank))

        return results

    def _semantic_search(
        self,
        query: str,
        top_n: int,
        where: dict = None,
        where_document: dict = None
    ) -> List[Tuple[str, float, int]]:
        """
        Semantic search via ChromaDB

        Args:
            query: Search query
            top_n: Number of results
            where: Optional metadata filter
            where_document: Optional document content filter

        Returns:
            List of (doc_id, distance, rank)
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
            include=["distances"]
        )

        # Format results
        semantic_results = []
        for rank, (doc_id, distance) in enumerate(zip(results['ids'][0], results['distances'][0])):
            # Convert distance to similarity (cosine distance → similarity)
            similarity = 1 - distance
            semantic_results.append((doc_id, similarity, rank))

        return semantic_results

    def _reciprocal_rank_fusion(
        self,
        bm25_results: List[Tuple[str, float, int]],
        semantic_results: List[Tuple[str, float, int]],
        k: int = 60,
        alpha: float = 0.5
    ) -> List[Dict]:
        """
        Reciprocal Rank Fusion (RRF)

        Score(doc) = alpha * RRF_semantic + (1-alpha) * RRF_bm25
        RRF(doc) = sum(1 / (k + rank))

        Args:
            bm25_results: BM25 ranked results
            semantic_results: Semantic ranked results
            k: RRF constant (typically 60)
            alpha: Weight for semantic vs BM25

        Returns:
            Sorted list of documents with combined scores
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
            try:
                # Try to get from cached index (fast path)
                idx = self.ids.index(doc_id)
                text = self.docs[idx]
                metadata = self.metadatas[idx]
            except ValueError:
                # Fallback: refetch from collection if doc_id not in index
                # This can happen if collection was updated after retriever initialization
                try:
                    result = self.collection.get(
                        ids=[doc_id],
                        include=["documents", "metadatas"]
                    )
                    if result['documents']:
                        text = result['documents'][0]
                        metadata = result['metadatas'][0]
                    else:
                        # Skip this document if it can't be found
                        continue
                except Exception:
                    # Skip documents that can't be retrieved
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


def test_hybrid_retriever():
    """Test du hybrid retriever"""
    import chromadb
    from pathlib import Path

    # Load ChromaDB
    client = chromadb.PersistentClient(path=str(Path(__file__).parent.parent / "chroma_db_new"))
    collection = client.get_collection(name="zotero_research_context_hybrid_v3")

    # Dummy embedding function (remplacer par Voyage)
    def dummy_embed(texts):
        return [[0.1] * 1024 for _ in texts]

    # Initialize retriever
    retriever = HybridRetriever(collection, embedding_function=dummy_embed)

    # Test query
    query = "black carbon impact on glacier albedo"
    results = retriever.search(query, top_k=5, alpha=0.7)

    print(f"\nHybrid Search Results for: '{query}'")
    print("=" * 80)

    for i, result in enumerate(results, 1):
        print(f"\n[{i}] Score: {result['score']:.4f} "
              f"(BM25: {result['bm25_score']:.3f}, Semantic: {result['semantic_score']:.3f})")
        print(f"    Source: {result['metadata'].get('source', 'unknown')}")
        print(f"    BM25 rank: {result['bm25_rank']}, Semantic rank: {result['semantic_rank']}")
        print(f"    Text: {result['text'][:200]}...")


if __name__ == "__main__":
    test_hybrid_retriever()
