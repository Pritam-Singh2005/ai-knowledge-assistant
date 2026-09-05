import os
import fitz
import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

DOCUMENTS_DIR = "data_ml/documents"
CHROMA_DIR = "data_ml/chroma_db"
COLLECTION_NAME = "knowledge_base"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(DOCUMENTS_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("Embedding model loaded successfully.")


# ============================================================
# CHROMADB
# ============================================================

client = chromadb.PersistentClient(
    path=CHROMA_DIR
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)


# ============================================================
# TEXT CHUNKING
# ============================================================

def split_text(
    text,
    chunk_size=500,
    overlap=100
):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(
            words[start:end]
        ).strip()

        if chunk:

            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# ============================================================
# PROCESS ONE PDF
# ============================================================

def process_pdf(pdf_path):

    print("\n" + "=" * 60)

    print(
        f"Processing: {os.path.basename(pdf_path)}"
    )

    print("=" * 60)


    document = fitz.open(pdf_path)

    all_chunks = []
    all_metadata = []
    all_ids = []


    # --------------------------------------------------------
    # PROCESS EACH PAGE
    # --------------------------------------------------------

    for page_number, page in enumerate(
        document,
        start=1
    ):

        text = page.get_text()

        if not text.strip():

            print(
                f"Page {page_number}: No text found."
            )

            continue


        chunks = split_text(text)


        print(
            f"Page {page_number}: "
            f"{len(chunks)} chunks"
        )


        for chunk_number, chunk in enumerate(
            chunks,
            start=1
        ):

            chunk_id = (
                f"{os.path.basename(pdf_path)}"
                f"_page_{page_number}"
                f"_chunk_{chunk_number}"
            )


            # Make ID safe
            chunk_id = (
                chunk_id
                .replace(" ", "_")
                .replace("/", "_")
                .replace("\\", "_")
            )


            all_chunks.append(chunk)


            all_metadata.append(
                {
                    "source": os.path.basename(
                        pdf_path
                    ),
                    "page": page_number,
                    "chunk": chunk_number
                }
            )


            all_ids.append(chunk_id)


    document.close()


    # --------------------------------------------------------
    # CHECK CONTENT
    # --------------------------------------------------------

    if not all_chunks:

        print(
            "\n❌ No readable text found in this PDF."
        )

        return


    print(
        f"\nTotal chunks created: "
        f"{len(all_chunks)}"
    )


    # --------------------------------------------------------
    # CREATE EMBEDDINGS
    # --------------------------------------------------------

    print(
        "\nCreating embeddings..."
    )


    embeddings = embedding_model.encode(
        all_chunks,
        show_progress_bar=True
    ).tolist()


    print(
        "Embeddings created successfully."
    )


    # --------------------------------------------------------
    # STORE IN CHROMADB
    # --------------------------------------------------------

    print(
        "\nStoring chunks in ChromaDB..."
    )


    collection.upsert(
        ids=all_ids,
        documents=all_chunks,
        embeddings=embeddings,
        metadatas=all_metadata
    )


    print(
        "\n✅ PDF successfully indexed."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n🚀 Starting PDF ingestion..."
    )


    pdf_files = [
        file
        for file in os.listdir(
            DOCUMENTS_DIR
        )
        if file.lower().endswith(".pdf")
    ]


    if not pdf_files:

        print(
            "\n❌ No PDF files found in:"
        )

        print(
            DOCUMENTS_DIR
        )

        return


    print(
        f"\nFound {len(pdf_files)} PDF(s):"
    )


    for pdf_file in pdf_files:

        print(
            f"📄 {pdf_file}"
        )


    # --------------------------------------------------------
    # PROCESS PDFs
    # --------------------------------------------------------

    for pdf_file in pdf_files:

        pdf_path = os.path.join(
            DOCUMENTS_DIR,
            pdf_file
        )


        try:

            process_pdf(
                pdf_path
            )

        except Exception as e:

            print(
                f"\n❌ Error processing "
                f"{pdf_file}:"
            )

            print(e)


    # --------------------------------------------------------
    # FINAL COUNT
    # --------------------------------------------------------

    print("\n" + "=" * 60)

    print(
        "🎉 INGESTION COMPLETED"
    )

    print("=" * 60)


    print(
        f"Total chunks in ChromaDB: "
        f"{collection.count()}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()