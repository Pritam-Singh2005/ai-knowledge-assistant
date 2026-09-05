# =============================================================================
# Module: Cross-Encoder Reranker
# Role: Reranks candidate document chunks based on semantic relevance.
# =============================================================================

from typing import List, Dict, Any, Tuple
from sentence_transformers import CrossEncoder

# Lazy global model loading
_reranker_instance = None


def get_reranker_model(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> CrossEncoder:
    """Initializes and caches the CrossEncoder model instance."""
    global _reranker_instance
    if _reranker_instance is None:
        print(f"Loading Reranker Model: {model_name}...")
        _reranker_instance = CrossEncoder(model_name)
    return _reranker_instance


def rerank_documents(
    query: str,
    documents: List[str],
    metadatas: List[Dict[str, Any]],
    top_k: int = 3,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Reranks retrieved candidate chunks using a Cross-Encoder.
    Returns top_k (reranked_documents, reranked_metadatas).
    """
    if not documents:
        return [], []

    model = get_reranker_model(model_name)

    # Form query-document sentence pairs
    pairs = [[query, doc] for doc in documents]

    # Predict relevance scores
    scores = model.predict(pairs)

    # Pair scores with documents and metadata
    scored_results = list(zip(scores, documents, metadatas))

    # Sort in descending order of relevance score
    scored_results.sort(key=lambda x: x[0], reverse=True)

    # Select top_k results
    top_results = scored_results[:top_k]

    reranked_docs = [doc for _, doc, _ in top_results]
    reranked_metas = [meta for _, _, meta in top_results]

    return reranked_docs, reranked_metas


if __name__ == "__main__":
    # Test Run
    sample_query = "What is deep learning?"
    sample_docs = [
        "Deep learning is a subset of machine learning using neural networks.",
        "Photosynthesis is the process by which plants turn sunlight into energy.",
        "Artificial intelligence covers machine learning and neural architectures."
    ]
    sample_metas = [{"source": "A.pdf"}, {"source": "B.pdf"}, {"source": "C.pdf"}]

    ranked_docs, ranked_metas = rerank_documents(sample_query, sample_docs, sample_metas, top_k=2)
    print("Reranked Top Result:", ranked_docs[0])