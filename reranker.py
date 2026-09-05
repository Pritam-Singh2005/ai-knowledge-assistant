# ============================================================
# reranker.py
# CrossEncoder Reranking
# ============================================================

from sentence_transformers import CrossEncoder


DEFAULT_RERANKER_MODEL = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


_reranker_models = {}


# ============================================================
# LOAD RERANKER
# ============================================================

def get_reranker(
    model_name=DEFAULT_RERANKER_MODEL
):

    if model_name not in _reranker_models:

        _reranker_models[model_name] = (
            CrossEncoder(model_name)
        )

    return _reranker_models[model_name]


# ============================================================
# RERANK DOCUMENTS
# ============================================================

def rerank_documents(
    query,
    documents,
    metadatas,
    top_k=3,
    model_name=DEFAULT_RERANKER_MODEL
):

    if not documents:

        return [], []

    # Keep metadata aligned
    pairs = []

    valid_documents = []

    valid_metadatas = []

    for document, metadata in zip(
        documents,
        metadatas
    ):

        if not document:
            continue

        valid_documents.append(
            document
        )

        valid_metadatas.append(
            metadata or {}
        )

        pairs.append(
            [
                query,
                document
            ]
        )

    if not pairs:

        return [], []

    model = get_reranker(
        model_name
    )

    scores = model.predict(
        pairs
    )

    ranked = sorted(
        zip(
            scores,
            valid_documents,
            valid_metadatas
        ),
        key=lambda x: float(x[0]),
        reverse=True
    )

    limit = min(
        int(top_k),
        len(ranked)
    )

    reranked_documents = [
        item[1]
        for item in ranked[:limit]
    ]

    reranked_metadatas = [
        item[2]
        for item in ranked[:limit]
    ]

    return (
        reranked_documents,
        reranked_metadatas
    )