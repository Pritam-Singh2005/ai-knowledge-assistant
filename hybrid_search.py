# =============================================================================
# Module: Hybrid Search (BM25 + Dense ChromaDB Retrieval)
# Role: Merges keyword matching with semantic vector search for higher recall.
# =============================================================================

from typing import List, Dict, Any
from rank_bm25 import BM25Okapi

def compute_rrf_scores(
    vector_results: List[Dict[str, Any]], 
    bm25_results: List[Dict[str, Any]], 
    k: int = 60
) -> List[Dict[str, Any]]:
    """
    Combines two ranked lists using Reciprocal Rank Fusion (RRF).
    RRF Score = sum(1 / (k + rank_i))
    """
    scores = {}
    doc_map = {}

    # Helper to process candidate list
    def add_rankings(items):
        for rank, item in enumerate(items):
            doc_id = item["id"]
            if doc_id not in scores:
                scores[doc_id] = 0.0
                doc_map[doc_id] = item
            scores[doc_id] += 1.0 / (k + (rank + 1))

    add_rankings(vector_results)
    add_rankings(bm25_results)

    # Sort merged documents by fused RRF score
    sorted_doc_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [doc_map[doc_id] for doc_id in sorted_doc_ids]


class BM25Retriever:
    def __init__(self, documents: List[str], metadatas: List[Dict[str, Any]], ids: List[str]):
        """Builds an in-memory BM25 index over vector store document chunks."""
        self.documents = documents
        self.metadatas = metadatas
        self.ids = ids
        
        # Tokenize documents for BM25
        tokenized_corpus = [doc.lower().split(" ") for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None

    def search(self, query: str, top_k: int = 6) -> List[Dict[str, Any]]:
        """Performs sparse keyword matching."""
        if not self.bm25:
            return []
        
        tokenized_query = query.lower().split(" ")
        scores = self.bm25.get_scores(tokenized_query)
        
        # Zip scores with documents
        ranked = sorted(
            zip(scores, self.documents, self.metadatas, self.ids), 
            key=lambda x: x[0], 
            reverse=True
        )[:top_k]

        return [
            {"document": doc, "metadata": meta, "id": doc_id} 
            for score, doc, meta, doc_id in ranked if score > 0
        ]


def hybrid_retrieve(
    query: str, 
    vector_docs: List[str], 
    vector_metas: List[Dict[str, Any]], 
    vector_ids: List[str],
    all_collection_docs: List[str],
    all_collection_metas: List[Dict[str, Any]],
    all_collection_ids: List[str],
    top_k: int = 6
) -> tuple[List[str], List[Dict[str, Any]]]:
    """
    Executes hybrid search combining dense vectors and sparse BM25.
    """
    # 1. Format dense results
    dense_candidates = [
        {"document": d, "metadata": m, "id": i} 
        for d, m, i in zip(vector_docs, vector_metas, vector_ids)
    ]

    # 2. Perform BM25 sparse search across corpus
    bm25_retriever = BM25Retriever(all_collection_docs, all_collection_metas, all_collection_ids)
    sparse_candidates = bm25_retriever.search(query, top_k=top_k)

    # 3. Fuse scores with RRF
    fused_candidates = compute_rrf_scores(dense_candidates, sparse_candidates)[:top_k]

    fused_docs = [item["document"] for item in fused_candidates]
    fused_metas = [item["metadata"] for item in fused_candidates]

    return fused_docs, fused_metas