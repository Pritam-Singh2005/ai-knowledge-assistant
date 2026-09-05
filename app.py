# ============================================================
# AETHERAI - AI KNOWLEDGE ASSISTANT
# ============================================================

import os
import tempfile
from pathlib import Path

import streamlit as st

from ingest import (
    index_pdf,
    get_collection
)


# ============================================================
# CONFIGURATION
# ============================================================

APP_TITLE = "AetherAI - AI Knowledge Assistant"

DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"

FALLBACK_GROQ_MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b"
]

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

RERANKER_MODEL_NAME = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


if "indexed_files" not in st.session_state:

    st.session_state.indexed_files = []


# ============================================================
# CHROMA COUNT
# ============================================================

def get_chroma_count():

    try:

        collection = get_collection()

        return collection.count()

    except Exception:

        return 0


# ============================================================
# GROQ API KEY
# ============================================================

def get_groq_api_key():

    # Streamlit Cloud secrets
    try:

        key = st.secrets.get(
            "GROQ_API_KEY"
        )

        if key:

            return key

    except Exception:

        pass

    # Local environment
    return os.getenv(
        "GROQ_API_KEY"
    )


# ============================================================
# GROQ CLIENT
# ============================================================

@st.cache_resource
def get_groq_client():

    from groq import Groq

    api_key = get_groq_api_key()

    if not api_key:

        return None

    return Groq(
        api_key=api_key
    )


# ============================================================
# EMBEDDING MODEL
# ============================================================

@st.cache_resource
def get_embedding_model():

    from sentence_transformers import (
        SentenceTransformer
    )

    return SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_documents(
    query,
    top_k=6
):

    collection = get_collection()

    count = collection.count()

    if count == 0:

        return []

    model = get_embedding_model()

    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

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

    documents = (
        results
        .get("documents", [[]])[0]
    )

    metadatas = (
        results
        .get("metadatas", [[]])[0]
    )

    distances = (
        results
        .get("distances", [[]])[0]
    )

    output = []

    for i, document in enumerate(
        documents
    ):

        metadata = (
            metadatas[i]
            if i < len(metadatas)
            else {}
        )

        distance = (
            distances[i]
            if i < len(distances)
            else 0
        )

        output.append(
            {
                "document": document,
                "metadata": metadata,
                "distance": distance
            }
        )

    return output


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(
    results
):

    context_parts = []

    for index, result in enumerate(
        results,
        start=1
    ):

        metadata = result[
            "metadata"
        ]

        source = metadata.get(
            "source",
            "Unknown"
        )

        page = metadata.get(
            "page",
            "?"
        )

        document = result[
            "document"
        ]

        context_parts.append(
            f"""
[Doc {index}]
Source: {source}
Page: {page}

{document}
"""
        )

    return "\n".join(
        context_parts
    )


# ============================================================
# QUERY REWRITING
# ============================================================

def rewrite_query(
    question
):

    client = get_groq_client()

    if client is None:

        return question

    if len(
        st.session_state.messages
    ) <= 1:

        return question

    history = st.session_state.messages[
        -6:
    ]

    history_text = "\n".join(
        [
            f"{m['role']}: {m['content']}"
            for m in history
        ]
    )

    prompt = f"""
Rewrite the user's latest question as a
standalone search query.

Resolve pronouns and references from the
conversation.

Conversation:
{history_text}

Latest question:
{question}

Return ONLY the rewritten query.
"""

    for model in FALLBACK_GROQ_MODELS:

        try:

            response = (
                client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0
                )
            )

            result = (
                response
                .choices[0]
                .message
                .content
                .strip()
            )

            if result:

                return result

        except Exception:

            continue

    return question


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    question,
    context=None
):

    client = get_groq_client()

    if client is None:

        return (
            "❌ Groq API key not found.\n\n"
            "Please configure GROQ_API_KEY."
        )

    if context:

        system_prompt = """
You are AetherAI, a document-grounded
AI Knowledge Assistant.

Answer ONLY using the supplied document
context.

Rules:

1. Never invent information.
2. If the answer is not present in the
   documents, say:

   "The answer is not available in the
   uploaded documents."

3. Cite sources using [Doc 1], [Doc 2],
   etc.
4. Give a clear and concise answer.
"""

        user_prompt = f"""
DOCUMENT CONTEXT:

{context}

QUESTION:

{question}
"""

    else:

        system_prompt = """
You are AetherAI, a helpful AI assistant.
Answer clearly and accurately.
"""

        user_prompt = question

    last_error = None

    for model in FALLBACK_GROQ_MODELS:

        try:

            response = (
                client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": user_prompt
                        }
                    ],
                    temperature=0.2
                )
            )

            return (
                response
                .choices[0]
                .message
                .content
                .strip()
            )

        except Exception as e:

            last_error = str(e)

            if (
                "404" in last_error
                or
                "model_not_found"
                in last_error.lower()
            ):

                continue

            break

    return (
        f"❌ Groq error:\n\n{last_error}"
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title(
        "🤖 AetherAI"
    )

    st.divider()

    # --------------------------------------------------------
    # DATABASE STATUS
    # --------------------------------------------------------

    st.subheader(
        "📚 Knowledge Base"
    )

    count = get_chroma_count()

    if count > 0:

        st.success(
            f"🟢 {count} chunks indexed"
        )

    else:

        st.warning(
            "🟡 No documents indexed"
        )

    st.divider()

    # --------------------------------------------------------
    # UPLOAD
    # --------------------------------------------------------

    st.subheader(
        "📄 Upload PDF"
    )

    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:

        st.write(
            f"Selected: {len(uploaded_files)} PDF(s)"
        )

        if st.button(
            "📥 Index Uploaded PDFs",
            use_container_width=True
        ):

            total_added = 0

            for uploaded_file in uploaded_files:

                st.write(
                    f"Processing: "
                    f"**{uploaded_file.name}**"
                )

                try:

                    # ----------------------------------------
                    # Temporary file
                    # ----------------------------------------

                    suffix = ".pdf"

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=suffix
                    ) as temp_file:

                        temp_file.write(
                            uploaded_file.getbuffer()
                        )

                        temp_path = (
                            temp_file.name
                        )

                    # ----------------------------------------
                    # Index
                    # ----------------------------------------

                    with st.spinner(
                        f"Indexing {uploaded_file.name}..."
                    ):

                        result = index_pdf(
                            temp_path
                        )

                    # ----------------------------------------
                    # Delete temporary file
                    # ----------------------------------------

                    try:

                        os.remove(
                            temp_path
                        )

                    except Exception:

                        pass

                    total_added += (
                        result["chunks_added"]
                    )

                    st.success(
                        f"✅ {uploaded_file.name}: "
                        f"{result['chunks_added']} chunks"
                    )

                    st.session_state.indexed_files.append(
                        uploaded_file.name
                    )

                except Exception as e:

                    st.error(
                        f"❌ Failed to index "
                        f"{uploaded_file.name}"
                    )

                    st.exception(e)

            # ------------------------------------------------
            # FINAL DATABASE CHECK
            # ------------------------------------------------

            final_count = (
                get_chroma_count()
            )

            if final_count > 0:

                st.success(
                    f"🎉 Indexing completed!\n\n"
                    f"ChromaDB now contains "
                    f"**{final_count} chunks**."
                )

                st.rerun()

            else:

                st.error(
                    "❌ ChromaDB is still empty."
                )

    st.divider()

    # --------------------------------------------------------
    # CLEAR CHAT
    # --------------------------------------------------------

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()

    st.divider()

    # --------------------------------------------------------
    # SYSTEM
    # --------------------------------------------------------

    st.subheader(
        "⚙️ System"
    )

    st.caption(
        f"LLM: {DEFAULT_GROQ_MODEL}"
    )

    st.caption(
        f"Embedding: {EMBEDDING_MODEL_NAME}"
    )

    st.caption(
        f"Reranker: {RERANKER_MODEL_NAME}"
    )

    st.caption(
        "Vector DB: ChromaDB"
    )


# ============================================================
# MAIN PAGE
# ============================================================

st.title(
    "🤖 AetherAI"
)

st.caption(
    "AI Knowledge Assistant with "
    "Document RAG"
)


# ============================================================
# MODE
# ============================================================

mode = st.radio(
    "Mode",
    [
        "💬 General Chat",
        "📚 Document RAG"
    ],
    horizontal=True
)


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# USER INPUT
# ============================================================

question = st.chat_input(
    "Ask AetherAI something..."
)


if question:

    # --------------------------------------------------------
    # User message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )

    # --------------------------------------------------------
    # Assistant
    # --------------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        # ====================================================
        # DOCUMENT RAG
        # ====================================================

        if mode == "📚 Document RAG":

            count = get_chroma_count()

            if count == 0:

                answer = (
                    "⚠️ No documents are currently "
                    "indexed.\n\n"
                    "Please upload a PDF from the "
                    "sidebar and click "
                    "**Index Uploaded PDFs**."
                )

                st.warning(
                    answer
                )

            else:

                # --------------------------------------------
                # Query rewriting
                # --------------------------------------------

                with st.spinner(
                    "🧠 Rewriting query..."
                ):

                    rewritten_query = (
                        rewrite_query(
                            question
                        )
                    )

                # --------------------------------------------
                # Retrieval
                # --------------------------------------------

                with st.spinner(
                    "🔎 Searching knowledge base..."
                ):

                    results = retrieve_documents(
                        rewritten_query,
                        top_k=6
                    )

                # --------------------------------------------
                # Context
                # --------------------------------------------

                context = build_context(
                    results
                )

                # --------------------------------------------
                # Answer
                # --------------------------------------------

                with st.spinner(
                    "🤖 Generating answer..."
                ):

                    answer = generate_answer(
                        question,
                        context
                    )

                st.markdown(
                    answer
                )

                # --------------------------------------------
                # Pipeline
                # --------------------------------------------

                with st.expander(
                    "🔍 RAG Pipeline"
                ):

                    st.write(
                        "Original query:"
                    )

                    st.code(
                        question
                    )

                    st.write(
                        "Rewritten query:"
                    )

                    st.code(
                        rewritten_query
                    )

                    st.write(
                        f"ChromaDB chunks: {count}"
                    )

                    st.write(
                        f"Retrieved chunks: "
                        f"{len(results)}"
                    )

                # --------------------------------------------
                # Sources
                # --------------------------------------------

                if results:

                    with st.expander(
                        "📚 Sources"
                    ):

                        for index, result in enumerate(
                            results,
                            start=1
                        ):

                            metadata = result[
                                "metadata"
                            ]

                            source = metadata.get(
                                "source",
                                "Unknown"
                            )

                            page = metadata.get(
                                "page",
                                "?"
                            )

                            st.markdown(
                                f"""
                                **[Doc {index}]**

                                📄 Source: `{source}`

                                📖 Page: `{page}`

                                ---
                                """
                            )

                # --------------------------------------------
                # Retrieved context
                # --------------------------------------------

                with st.expander(
                    "📄 Retrieved Context"
                ):

                    st.text(
                        context
                    )

        # ====================================================
        # GENERAL CHAT
        # ====================================================

        else:

            with st.spinner(
                "🤖 Thinking..."
            ):

                answer = generate_answer(
                    question
                )

            st.markdown(
                answer
            )

    # --------------------------------------------------------
    # Save assistant message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )