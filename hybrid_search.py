# ============================================================
# hybrid_search.py
# Lightweight Hybrid Retrieval
# Vector Retrieval + Keyword Matching
# ============================================================

import re
from collections import defaultdict


# ============================================================
# TOKENIZER
# ============================================================

def tokenize(text):

    if not text:
        return []

    return re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text.lower()
    )


# ============================================================
# KEYWORD SCORE
# ============================================================

def keyword_score(
    query,
    document
):

    query_tokens = set(
        tokenize(query)
    )

    document_tokens = set(
        tokenize(document)
    )

    if not query_tokens:

        return 0.0

    intersection = (
        query_tokens &
        document_tokens
    )

    return (
        len(intersection) /
        len(query_tokens)
    )


# ============================================================
# HYBRID RETRIEVE
# ============================================================

def hybrid_retrieve(
    query,
    vector_docs,
    vector_metas,
    vector_ids,
    all_collection_docs,
    all_collection_metas,
    all_collection_ids,
    top_k=6
):

    # --------------------------------------------------------
    # Create lookup for vector results
    # --------------------------------------------------------

    vector_scores = {}

    for rank, doc_id in enumerate(
        vector_ids
    ):

        if doc_id:

            vector_scores[doc_id] = (
                1.0 / (rank + 1)
            )

    # --------------------------------------------------------
    # Score every document
    # --------------------------------------------------------

    candidates = []

    for i, document in enumerate(
        all_collection_docs
    ):

        if not document:
            continue

        metadata = (
            all_collection_metas[i]
            if i < len(all_collection_metas)
            else {}
        )

        doc_id = (
            all_collection_ids[i]
            if i < len(all_collection_ids)
            else str(i)
        )

        vector_score = vector_scores.get(
            doc_id,
            0.0
        )

        keyword = keyword_score(
            query,
            document
        )

        # ----------------------------------------------------
        # Hybrid score
        # ----------------------------------------------------

        score = (
            0.65 * vector_score
            +
            0.35 * keyword
        )

        candidates.append(
            (
                score,
                document,
                metadata,
                doc_id
            )
        )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    candidates.sort(
        key=lambda x: x[0],
        reverse=True
    )

    selected = candidates[
        :min(
            int(top_k),
            len(candidates)
        )
    ]

    documents = [
        item[1]
        for item in selected
    ]

    metadatas = [
        item[2]
        for item in selected
    ]

    ids = [
        item[3]
        for item in selected
    ]

    return (
        documents,
        metadatas,
        ids
    )