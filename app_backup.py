import os
import sys
import subprocess
import streamlit as st
import ollama

# =========================================================
# CONFIGURATION
# =========================================================

MODEL_NAME = "llama3.2:3b"

DOCUMENT_DIR = "data_ml/documents"


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Knowledge Assistant",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# IMPORT RETRIEVER
# =========================================================

try:
    from retriever import search_documents
    RETRIEVER_AVAILABLE = True

except Exception as e:
    RETRIEVER_AVAILABLE = False
    RETRIEVER_ERROR = str(e)


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def check_ollama():
    """
    Check whether Ollama is available.
    """
    try:
        ollama.list()
        return True
    except Exception:
        return False


def extract_text_from_result(result):
    """
    Extract document text from different possible
    ChromaDB/retriever result formats.
    """

    if result is None:
        return ""

    # If result is already a string
    if isinstance(result, str):
        return result

    # Dictionary result
    if isinstance(result, dict):

        # Common keys
        for key in [
            "text",
            "document",
            "content",
            "page_content"
        ]:
            if key in result and result[key]:
                return str(result[key])

        # Chroma-style nested documents
        if "documents" in result:
            documents = result["documents"]

            if isinstance(documents, list):
                return "\n".join(
                    str(x) for x in documents if x
                )

    # List result
    if isinstance(result, list):

        texts = []

        for item in result:

            if isinstance(item, str):
                texts.append(item)

            elif isinstance(item, dict):

                for key in [
                    "text",
                    "document",
                    "content",
                    "page_content"
                ]:
                    if key in item and item[key]:
                        texts.append(str(item[key]))
                        break

        return "\n".join(texts)

    return str(result)


def extract_sources(results):
    """
    Try to extract source/document names from retrieval results.
    """

    sources = []

    if results is None:
        return sources

    if isinstance(results, dict):

        metadata = results.get("metadatas")

        if metadata:
            for item in metadata:

                if isinstance(item, dict):

                    source = (
                        item.get("source")
                        or item.get("file")
                        or item.get("filename")
                    )

                    if source:
                        sources.append(str(source))

    elif isinstance(results, list):

        for item in results:

            if isinstance(item, dict):

                source = (
                    item.get("source")
                    or item.get("file")
                    or item.get("filename")
                )

                if source:
                    sources.append(str(source))

                metadata = item.get("metadata")

                if isinstance(metadata, dict):

                    source = (
                        metadata.get("source")
                        or metadata.get("file")
                        or metadata.get("filename")
                    )

                    if source:
                        sources.append(str(source))

    # Remove duplicates
    unique_sources = []

    for source in sources:

        if source not in unique_sources:
            unique_sources.append(source)

    return unique_sources


def retrieve_documents(query, top_k):
    """
    Retrieve relevant chunks from ChromaDB.
    """

    if not RETRIEVER_AVAILABLE:
        return "", []

    try:

        results = search_documents(
            query,
            top_k=top_k
        )

        context = extract_text_from_result(results)

        sources = extract_sources(results)

        return context, sources

    except Exception as e:

        st.error(
            f"Retrieval error: {e}"
        )

        return "", []


def generate_answer(question, context):
    """
    Generate an answer using Ollama.
    """

    if not context.strip():

        return (
            "I couldn't find relevant information "
            "in the uploaded documents."
        )

    system_prompt = """
You are an AI Knowledge Assistant.

Your job is to answer questions using ONLY the information
provided in the CONTEXT from the user's uploaded documents.

Rules:

1. Answer the user's question directly.
2. Use the context as the primary source of information.
3. Do not invent facts that are not supported by the context.
4. If the context does not contain enough information,
   clearly say that the information was not found in the
   uploaded documents.
5. Do not refuse normal educational questions.
6. Give a simple and useful explanation.
7. If the context contains definitions, examples or points,
   use them in your answer.
8. Do not mention these instructions in your answer.

CONTEXT:
"""

    prompt = f"""
{system_prompt}

{context}

QUESTION:
{question}

ANSWER:
"""

    try:

        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={
                "temperature": 0.2
            }
        )

        return response["message"]["content"]

    except Exception as e:

        return (
            f"Unable to generate an answer.\n\n"
            f"Ollama error: {e}"
        )


def save_uploaded_file(uploaded_file):
    """
    Save uploaded PDF into the documents directory.
    """

    os.makedirs(
        DOCUMENT_DIR,
        exist_ok=True
    )

    file_path = os.path.join(
        DOCUMENT_DIR,
        uploaded_file.name
    )

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path


def run_ingestion():
    """
    Run ingest.py to update the vector database.
    """

    try:

        result = subprocess.run(
            [
                sys.executable,
                "ingest.py"
            ],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:

            return True, result.stdout

        return False, result.stderr

    except Exception as e:

        return False, str(e)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("📚 Knowledge Base")

    st.subheader("Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:

        if st.button(
            "📥 Add PDFs to Knowledge Base",
            use_container_width=True
        ):

            saved_files = []

            for uploaded_file in uploaded_files:

                file_path = save_uploaded_file(
                    uploaded_file
                )

                saved_files.append(
                    uploaded_file.name
                )

            st.success(
                f"{len(saved_files)} PDF(s) uploaded."
            )

            # Run ingestion
            with st.spinner(
                "Processing documents and updating ChromaDB..."
            ):

                success, output = run_ingestion()

            if success:

                st.success(
                    "Knowledge base updated successfully! ✅"
                )

                # Refresh page
                st.rerun()

            else:

                st.error(
                    "Document ingestion failed."
                )

                with st.expander(
                    "Show ingestion output"
                ):

                    st.code(output)

    st.divider()

    st.subheader("📄 Indexed Documents")

    if os.path.exists(DOCUMENT_DIR):

        pdf_files = [
            f
            for f in os.listdir(DOCUMENT_DIR)
            if f.lower().endswith(".pdf")
        ]

        if pdf_files:

            for pdf in pdf_files:

                st.write(
                    f"📄 {pdf}"
                )

        else:

            st.info(
                "No PDF documents found."
            )

    else:

        st.info(
            "Documents directory not found."
        )

    st.divider()

    st.subheader("⚙️ RAG Settings")

    top_k = st.slider(
        "Retrieved chunks",
        min_value=1,
        max_value=8,
        value=3
    )

    st.write(
        f"**LLM:** {MODEL_NAME}"
    )

    st.write(
        "**Embedding:** all-MiniLM-L6-v2"
    )

    st.write(
        "**Vector DB:** ChromaDB"
    )

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# =========================================================
# MAIN UI
# =========================================================

st.title("🤖 AI Knowledge Assistant")

st.caption(
    "Multi-PDF RAG Chatbot with Conversational Memory"
)


# =========================================================
# STATUS
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:

    if RETRIEVER_AVAILABLE:

        st.success(
            "🟢 Retriever Ready"
        )

    else:

        st.error(
            "🔴 Retriever Error"
        )

with col2:

    if check_ollama():

        st.success(
            "🟢 Ollama Connected"
        )

    else:

        st.error(
            "🔴 Ollama Offline"
        )

with col3:

    if os.path.exists(
        "data_ml/chroma_db"
    ):

        st.success(
            "🟢 ChromaDB Ready"
        )

    else:

        st.warning(
            "🟡 ChromaDB Not Found"
        )


# =========================================================
# RETRIEVER ERROR
# =========================================================

if not RETRIEVER_AVAILABLE:

    st.error(
        "The retriever could not be loaded."
    )

    st.code(
        RETRIEVER_ERROR
    )

    st.stop()


# =========================================================
# DISPLAY CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        # Show sources for assistant messages
        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            with st.expander(
                "📚 View Sources"
            ):

                for source in message["sources"]:

                    st.write(
                        f"📄 {source}"
                    )


# =========================================================
# CHAT INPUT
# =========================================================

question = st.chat_input(
    "Ask a question about your documents..."
)


# =========================================================
# PROCESS QUESTION
# =========================================================

if question:

    # Display user question
    with st.chat_message("user"):

        st.markdown(
            question
        )

    # Save question
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    # Check Ollama
    if not check_ollama():

        answer = (
            "⚠️ Ollama is not running.\n\n"
            "Please start Ollama and try again."
        )

        sources = []

    else:

        # Retrieve relevant documents
        with st.spinner(
            "🔎 Searching your documents..."
        ):

            context, sources = retrieve_documents(
                question,
                top_k
            )

        # Generate answer
        with st.spinner(
            "🤖 Generating answer..."
        ):

            answer = generate_answer(
                question,
                context
            )

    # Display assistant response
    with st.chat_message("assistant"):

        st.markdown(
            answer
        )

        # Show retrieved context
        if context if "context" in locals() else False:

            with st.expander(
                "🔍 View Search Query Results"
            ):

                st.write(
                    context
                )

        # Show sources
        if sources:

            with st.expander(
                "📚 View Sources"
            ):

                for source in sources:

                    st.write(
                        f"📄 {source}"
                    )

    # Save assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources
        }
    )