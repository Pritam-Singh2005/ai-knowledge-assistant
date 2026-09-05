# ============================================================
# AETHERAI - PDF INGESTION
# ============================================================

import os
import glob
import hashlib
from pathlib import Path

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

CHROMA_PATH = "./chroma_db"

COLLECTION_NAME = "aether_knowledge_base"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

DOCUMENTS_DIR = "./data_ml/documents"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


# ============================================================
# DIRECTORY
# ============================================================

Path(CHROMA_PATH).mkdir(
    parents=True,
    exist_ok=True
)

Path(DOCUMENTS_DIR).mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CHROMA CLIENT
# ============================================================

def get_chroma_client():

    return chromadb.PersistentClient(
        path=CHROMA_PATH
    )


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
            "\nLoading embedding model..."
        )

        _model = SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )

        print(
            "Embedding model loaded."
        )

    return _model


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    if not text:

        return ""

    return " ".join(
        text.split()
    ).strip()


# ============================================================
# CHUNKING
# ============================================================

def create_chunks(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):

    text = clean_text(
        text
    )

    if not text:

        return []

    chunks = []

    start = 0

    while start < len(text):

        end = min(
            start + chunk_size,
            len(text)
        )

        chunk = text[
            start:end
        ].strip()

        if chunk:

            chunks.append(
                chunk
            )

        if end >= len(text):

            break

        start = end - overlap

    return chunks


# ============================================================
# READ PDF
# ============================================================

def extract_pdf_pages(
    pdf_path
):

    print(
        f"\nReading PDF: {pdf_path}"
    )

    reader = PdfReader(
        pdf_path
    )

    print(
        f"Pages found: {len(reader.pages)}"
    )

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        try:

            text = page.extract_text()

        except Exception as e:

            print(
                f"Page {page_number} error: {e}"
            )

            text = ""

        text = clean_text(
            text
        )

        if text:

            pages.append(
                {
                    "page": page_number,
                    "text": text
                }
            )

    return pages


# ============================================================
# INDEX ONE PDF
# ============================================================

def index_pdf(
    pdf_path
):

    pdf_path = str(
        pdf_path
    )

    if not os.path.exists(
        pdf_path
    ):

        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    print(
        "\n" + "=" * 60
    )

    print(
        "INDEXING PDF"
    )

    print(
        "=" * 60
    )

    # --------------------------------------------------------
    # Read PDF
    # --------------------------------------------------------

    pages = extract_pdf_pages(
        pdf_path
    )

    if not pages:

        raise ValueError(
            "No readable text was extracted from this PDF."
        )

    # --------------------------------------------------------
    # Prepare chunks
    # --------------------------------------------------------

    documents = []

    metadatas = []

    ids = []

    file_name = Path(
        pdf_path
    ).name

    file_hash = hashlib.md5(
        Path(pdf_path).read_bytes()
    ).hexdigest()[:12]

    chunk_number = 0

    for page_data in pages:

        page_number = page_data[
            "page"
        ]

        page_text = page_data[
            "text"
        ]

        chunks = create_chunks(
            page_text
        )

        for chunk in chunks:

            chunk_id = (
                f"{file_hash}_"
                f"page_{page_number}_"
                f"chunk_{chunk_number}"
            )

            documents.append(
                chunk
            )

            metadatas.append(
                {
                    "source": file_name,
                    "page": page_number,
                    "chunk": chunk_number,
                    "file_hash": file_hash
                }
            )

            ids.append(
                chunk_id
            )

            chunk_number += 1

    print(
        f"Chunks created: {len(documents)}"
    )

    if not documents:

        raise ValueError(
            "PDF was read but zero chunks were created."
        )

    # --------------------------------------------------------
    # Embeddings
    # --------------------------------------------------------

    model = get_embedding_model()

    print(
        "\nCreating embeddings..."
    )

    embeddings = model.encode(
        documents,
        normalize_embeddings=True,
        show_progress_bar=True
    ).tolist()

    print(
        f"Embeddings created: {len(embeddings)}"
    )

    # --------------------------------------------------------
    # ChromaDB
    # --------------------------------------------------------

    collection = get_collection()

    before_count = collection.count()

    print(
        f"\nChromaDB before indexing: {before_count}"
    )

    # --------------------------------------------------------
    # Insert
    # --------------------------------------------------------

    print(
        "Inserting chunks into ChromaDB..."
    )

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    after_count = collection.count()

    print(
        f"ChromaDB after indexing: {after_count}"
    )

    if after_count <= 0:

        raise RuntimeError(
            "Indexing finished but ChromaDB still contains 0 chunks."
        )

    print(
        "\nSUCCESS!"
    )

    return {
        "file": file_name,
        "chunks_added": len(documents),
        "total_chunks": after_count
    }


# ============================================================
# INDEX ALL PDFs IN DATA DIRECTORY
# ============================================================

def index_all_pdfs():

    pdf_files = glob.glob(
        os.path.join(
            DOCUMENTS_DIR,
            "*.pdf"
        )
    )

    if not pdf_files:

        print(
            "\nNo PDF files found in:"
        )

        print(
            DOCUMENTS_DIR
        )

        return []

    results = []

    for pdf_file in pdf_files:

        try:

            result = index_pdf(
                pdf_file
            )

            results.append(
                result
            )

        except Exception as e:

            print(
                f"\nFAILED: {pdf_file}"
            )

            print(
                str(e)
            )

    return results


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    print(
        "\nAetherAI PDF Ingestion"
    )

    print(
        f"Documents folder: {DOCUMENTS_DIR}"
    )

    print(
        f"ChromaDB: {CHROMA_PATH}"
    )

    results = index_all_pdfs()

    print(
        "\n" + "=" * 60
    )

    print(
        "FINAL RESULT"
    )

    print(
        "=" * 60
    )

    if results:

        for result in results:

            print(
                f"\n{result['file']}"
            )

            print(
                f"Chunks added: "
                f"{result['chunks_added']}"
            )

            print(
                f"Total ChromaDB chunks: "
                f"{result['total_chunks']}"
            )

    else:

        print(
            "No files were indexed."
        )