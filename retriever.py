# ============================================================
# AETHERAI - RETRIEVER
# ============================================================

import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

CHROMA_PATH = "./chroma_db"

COLLECTION_NAME = "aether_knowledge_base"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


# ============================================================
# CHROMA CLIENT
# ============================================================

def get_chroma_client():

    return chromadb.PersistentClient(
        path=CHROMA_PATH
    )


# ============================================================
# COLLECTION
# ============================================================

def get_collection():

    client = get_chroma_client()

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine"
        }
    )

    return collection


# ============================================================
# EMBEDDING MODEL
# ============================================================

_model = None


def get_embedding_model():

    global _model

    if _model is None:

        print(
            "Loading embedding model..."
        )

        _model = SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )

        print(
            "Embedding model loaded."
        )

    return _model


# ============================================================
# DATABASE COUNT
# ============================================================

def get_collection_count():

    try:

        collection = get_collection()

        return collection.count()

    except Exception as e:

        print(
            f"ChromaDB error: {e}"
        )

        return 0


# ============================================================
# RETRIEVE DOCUMENTS
# ============================================================

def retrieve_documents(
    query,
    top_k=6
):

    collection = get_collection()

    count = collection.count()

    print(
        f"\nChromaDB documents: {count}"
    )

    if count == 0:

        print(
            "WARNING: ChromaDB is empty."
        )

        return {
            "documents": [],
            "metadatas": [],
            "ids": [],
            "distances": []
        }

    # --------------------------------------------------------
    # Create query embedding
    # --------------------------------------------------------

    model = get_embedding_model()

    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    # --------------------------------------------------------
    # Search ChromaDB
    # --------------------------------------------------------

    results = collection.query(
        query_embeddings=[
            query_embedding
        ],
        n_results=min(
            top_k,
            count
        ),
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    documents = results.get(
        "documents",
        [[]]
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]

    distances = results.get(
        "distances",
        [[]]
    )[0]

    ids = results.get(
        "ids",
        [[]]
    )[0]

    print(
        f"Retrieved: {len(documents)} chunks"
    )

    return {
        "documents": documents,
        "metadatas": metadatas,
        "ids": ids,
        "distances": distances
    }


# ============================================================
# SIMPLE SEARCH
# ============================================================

def search_documents(
    query,
    top_k=6
):

    result = retrieve_documents(
        query,
        top_k
    )

    return result["documents"]


# ============================================================
# GET ALL DOCUMENTS
# ============================================================

def get_all_documents():

    collection = get_collection()

    count = collection.count()

    if count == 0:

        return {
            "documents": [],
            "metadatas": [],
            "ids": []
        }

    results = collection.get(
        include=[
            "documents",
            "metadatas"
        ]
    )

    return {
        "documents": results.get(
            "documents",
            []
        ),
        "metadatas": results.get(
            "metadatas",
            []
        ),
        "ids": results.get(
            "ids",
            []
        )
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print(
        "\n" + "=" * 60
    )

    print(
        "AetherAI Retriever Test"
    )

    print(
        "=" * 60
    )

    count = get_collection_count()

    print(
        f"\nChromaDB path:"
        f" {CHROMA_PATH}"
    )

    print(
        f"Collection:"
        f" {COLLECTION_NAME}"
    )

    print(
        f"Embedding model:"
        f" {EMBEDDING_MODEL_NAME}"
    )

    print(
        f"Documents indexed:"
        f" {count}"
    )

    if count > 0:

        result = retrieve_documents(
            "What is machine learning?",
            top_k=3
        )

        print(
            "\nRetrieved chunks:"
        )

        for i, doc in enumerate(
            result["documents"],
            start=1
        ):

            print(
                f"\n--- Chunk {i} ---"
            )

            print(
                doc[:500]
            )

    else:

        print(
            "\nNo documents indexed."
        )

        print(
            "Run: python ingest.py"
        )