from pathlib import Path
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# =========================
# Paths
# =========================

DOCUMENTS_DIR = Path("data_ml/documents")
CHROMA_DIR = "data_ml/chroma_db"


# =========================
# Embedding Model
# =========================

model = SentenceTransformer("all-MiniLM-L6-v2")


# =========================
# ChromaDB
# =========================

client = chromadb.PersistentClient(path=CHROMA_DIR)

collection = client.get_or_create_collection(
    name="knowledge_base"
)


# =========================
# Extract text from PDF
# =========================

def extract_text(pdf_path):

    reader = PdfReader(str(pdf_path))

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# =========================
# Create text chunks
# =========================

def create_chunks(text, chunk_size=500, overlap=50):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# =========================
# Add documents to ChromaDB
# =========================

def add_documents():

    if not DOCUMENTS_DIR.exists():

        print(f"Documents folder not found: {DOCUMENTS_DIR}")

        return

    pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:

        print(f"No PDF files found in {DOCUMENTS_DIR}")

        return

    for pdf_file in pdf_files:

        print(f"\nReading: {pdf_file.name}")

        text = extract_text(pdf_file)

        print(f"Characters extracted: {len(text)}")

        chunks = create_chunks(text)

        print(f"Chunks created: {len(chunks)}")

        embeddings = model.encode(
            chunks,
            convert_to_numpy=True
        ).tolist()

        ids = [
            f"{pdf_file.stem}_{i}"
            for i in range(len(chunks))
        ]

        metadatas = [
            {
                "source": pdf_file.name,
                "chunk": i
            }
            for i in range(len(chunks))
        ]

        collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )

        print(f"Added {len(chunks)} chunks from {pdf_file.name}")


# =========================
# Run ingestion
# =========================

if __name__ == "__main__":

    add_documents()

    print(
        f"\nTotal documents in collection: "
        f"{collection.count()}"
    )