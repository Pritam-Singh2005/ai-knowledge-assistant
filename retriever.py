# =============================================================================
# Module: Vector Retriever with Multi-Query Rewriting
# Role: Performs ChromaDB document retrieval with query expansion.
# =============================================================================

import requests
import chromadb
from typing import List, Dict, Any, Tuple

# Initialize Persistent ChromaDB Client
chroma_client = chromadb.PersistentClient(path="./chroma_db")


def rewrite_query(query: str, model_name: str = "llama3.2:1b") -> List[str]:
    """
    Generates alternative search queries to improve retrieval recall.
    """
    prompt = f"""Generate 3 alternative search queries for retrieving relevant context based on: "{query}".
Return ONLY the queries, one per line, without numbers or extra formatting."""

    try:
        res = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model_name, "prompt": prompt, "stream": False},
            timeout=10
        )
        if res.status_code == 200:
            raw_lines = res.json().get("response", "").split("\n")
            lines = [l.strip() for l in raw_lines if l.strip()]
            # Ensure the original user query is included first
            if query not in lines:
                lines.insert(0, query)
            return lines
    except Exception as e:
        print(f"Query rewriting skipped ({e}), defaulting to original prompt.")
    
    return [query]


def retrieve_documents(
    query: str, 
    collection_name: str = "pdf_collection", 
    initial_top_k: int = 6,
    model_name: str = "llama3.2:1b"
) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
    """
    Retrieves document chunks, metadata, and chunk IDs from ChromaDB.
    Returns:
        (documents_list, metadatas_list, ids_list)
    """
    try:
        collection = chroma_client.get_collection(name=collection_name)
    except Exception as e:
        print(f"Warning: Could not connect to collection '{collection_name}': {e}")
        return [], [], []

    # Rewrite query into multiple variants
    queries = rewrite_query(query, model_name)

    # Perform query against ChromaDB
    try:
        results = collection.query(
            query_texts=queries, 
            n_results=initial_top_k
        )
    except Exception as e:
        print(f"ChromaDB Query Error: {e}")
        return [], [], []

    documents_list, metadatas_list, ids_list = [], [], []
    seen_ids = set()

    raw_docs = results.get("documents", [])
    raw_metas = results.get("metadatas", [])
    raw_ids = results.get("ids", [])

    # Process and deduplicate chunks across rewritten query results
    for q_idx in range(len(queries)):
        q_docs = raw_docs[q_idx] if q_idx < len(raw_docs) else []
        q_metas = raw_metas[q_idx] if q_idx < len(raw_metas) else [{}] * len(q_docs)
        q_ids = raw_ids[q_idx] if q_idx < len(raw_ids) else [f"doc_{i}" for i in range(len(q_docs))]

        for doc, meta, doc_id in zip(q_docs, q_metas, q_ids):
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                documents_list.append(doc)
                metadatas_list.append(meta or {})
                ids_list.append(doc_id)

    return documents_list, metadatas_list, ids_list


if __name__ == "__main__":
    # Test Run
    test_docs, test_metas, test_ids = retrieve_documents("What is machine learning?")
    print(f"Retrieved {len(test_docs)} chunks.")