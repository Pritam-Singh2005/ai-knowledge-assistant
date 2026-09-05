# =============================================================================
# Module: AetherAI — Production General & RAG Assistant
# Features:
#   - Dual Mode (General Chat + Document RAG)
#   - Groq Cloud LLM
#   - Streaming Responses
#   - Multithreaded PDF Processing
#   - Hybrid Retrieval
#   - CrossEncoder Reranking
#   - Conversation-Aware Query Rewriting
#   - Source Citations
#   - Faithfulness / Hallucination Checks
# =============================================================================

import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Tuple

import streamlit as st
import chromadb

from groq import Groq

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# -----------------------------------------------------------------------------
# Modular RAG Components
# -----------------------------------------------------------------------------

from retriever import retrieve_documents
from reranker import rerank_documents
from hallucination_checker import check_hallucination
from hybrid_search import hybrid_retrieve


# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="AetherAI — Intelligent Assistant",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)


# -----------------------------------------------------------------------------
# 2. CUSTOM UI STYLING
# -----------------------------------------------------------------------------

st.markdown(
    """
    <style>

        .main-title {
            font-size: 2.3rem;
            font-weight: 800;
            letter-spacing: -0.5px;
            margin-bottom: 0.2rem;
        }

        .sub-title {
            font-size: 1rem;
            color: #6c757d;
            margin-bottom: 1.5rem;
        }

        .citation-card {
            background-color: #f8f9fa;
            border-left: 4px solid #6C5CE7;
            padding: 10px 14px;
            margin-top: 8px;
            border-radius: 4px;
            font-size: 0.9rem;
        }

        .badge-grounded {
            background-color: #d4edda;
            color: #155724;
            padding: 4px 10px;
            border-radius: 12px;
            font-weight: 600;
            font-size: 0.85rem;
            display: inline-block;
        }

        .badge-warning {
            background-color: #f8d7da;
            color: #721c24;
            padding: 4px 10px;
            border-radius: 12px;
            font-weight: 600;
            font-size: 0.85rem;
            display: inline-block;
        }

    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------------------------------------------------------
# 3. CONFIGURATION
# -----------------------------------------------------------------------------

MAX_WORKERS = min(
    32,
    (os.cpu_count() or 1) + 4
)

executor = ThreadPoolExecutor(
    max_workers=MAX_WORKERS
)


# Groq model
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"

# Embedding model used by your RAG retriever
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Default reranker
DEFAULT_RERANKER_MODEL = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# -----------------------------------------------------------------------------
# 4. GROQ CLIENT
# -----------------------------------------------------------------------------

@st.cache_resource
def get_groq_client():

    """
    Create a single Groq client.

    Local:
        Reads GROQ_API_KEY from environment.

    Streamlit Cloud:
        Reads GROQ_API_KEY from st.secrets.
    """

    api_key = None

    # ---------------------------------------------------------
    # Try Streamlit Secrets
    # ---------------------------------------------------------

    try:

        api_key = st.secrets.get(
            "GROQ_API_KEY"
        )

    except Exception:

        api_key = None

    # ---------------------------------------------------------
    # Try environment variable
    # ---------------------------------------------------------

    if not api_key:

        api_key = os.getenv(
            "GROQ_API_KEY"
        )

    # ---------------------------------------------------------
    # Validate
    # ---------------------------------------------------------

    if not api_key:

        return None

    return Groq(
        api_key=api_key
    )


groq_client = get_groq_client()


# -----------------------------------------------------------------------------
# 5. CHROMADB CLIENT
# -----------------------------------------------------------------------------

@st.cache_resource
def get_chroma_client() -> Any:

    """
    Returns a persistent ChromaDB client.
    """

    if chromadb is None:

        raise RuntimeError(
            "ChromaDB is not installed. "
            "Please install the chromadb package."
        )

    return chromadb.PersistentClient(
        path="./chroma_db"
    )


chroma_client = get_chroma_client()


collection = chroma_client.get_or_create_collection(
    name="aether_knowledge_base"
)


# -----------------------------------------------------------------------------
# 6. SIDEBAR CONFIGURATION
# -----------------------------------------------------------------------------

with st.sidebar:

    st.title(
        "🌌 Aether Settings"
    )

    # -------------------------------------------------------------------------
    # Model Engine
    # -------------------------------------------------------------------------

    st.subheader(
        "🤖 Model Engine"
    )

    groq_model = st.text_input(
        "Groq Model",
        value=DEFAULT_GROQ_MODEL
    )

    reranker_model = st.text_input(
        "Reranker Model",
        value=DEFAULT_RERANKER_MODEL
    )

    st.divider()

    # -------------------------------------------------------------------------
    # Knowledge Base
    # -------------------------------------------------------------------------

    st.subheader(
        "🔍 Knowledge Base"
    )

    enable_rag = st.toggle(
        "Enable Document Grounding",
        value=True
    )

    with st.expander(
        "Advanced Retrieval Parameters"
    ):

        initial_top_k = st.slider(
            "Initial Document Recall (K)",
            min_value=2,
            max_value=20,
            value=6
        )

        final_top_k = st.slider(
            "Final Reranked Context (K)",
            min_value=1,
            max_value=8,
            value=3
        )

        enable_hybrid = st.checkbox(
            "Enable Hybrid BM25 Fusion",
            value=True
        )

    # -------------------------------------------------------------------------
    # PDF Upload
    # -------------------------------------------------------------------------

    uploaded_files = st.file_uploader(
        "Upload PDFs to extend Aether's memory",
        type=["pdf"],
        accept_multiple_files=True
    )

    # -------------------------------------------------------------------------
    # PDF PARSER
    # -------------------------------------------------------------------------

    def _parse_pdf(
        uploaded_file
    ) -> Tuple[
        List[str],
        List[Dict[str, Any]],
        List[str]
    ]:

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )

        docs = []
        metas = []
        ids = []

        # -----------------------------------------------------
        # Create temporary PDF
        # -----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp:

            tmp.write(
                uploaded_file.read()
            )

            tmp_path = tmp.name

        try:

            # -------------------------------------------------
            # Load PDF
            # -------------------------------------------------

            loader = PyPDFLoader(
                tmp_path
            )

            documents = loader.load()

            # -------------------------------------------------
            # Split into chunks
            # -------------------------------------------------

            splits = text_splitter.split_documents(
                documents
            )

            # -------------------------------------------------
            # Create metadata
            # -------------------------------------------------

            for i, split in enumerate(
                splits
            ):

                docs.append(
                    split.page_content
                )

                metas.append(
                    {
                        "source": uploaded_file.name,
                        "page": split.metadata.get(
                            "page",
                            0
                        ) + 1
                    }
                )

                ids.append(
                    f"{uploaded_file.name}_chunk_{i}"
                )

        finally:

            if os.path.exists(
                tmp_path
            ):

                os.remove(
                    tmp_path
                )

        return (
            docs,
            metas,
            ids
        )

    # -------------------------------------------------------------------------
    # Index Uploaded Documents
    # -------------------------------------------------------------------------

    if uploaded_files:

        if st.button(
            "📥 Index Documents",
            use_container_width=True
        ):

            with st.spinner(
                "Ingesting files in parallel..."
            ):

                futures = [
                    executor.submit(
                        _parse_pdf,
                        f
                    )

                    for f in uploaded_files
                ]

                all_docs = []
                all_metas = []
                all_ids = []

                for future in futures:

                    d, m, i = future.result()

                    all_docs.extend(d)
                    all_metas.extend(m)
                    all_ids.extend(i)

                # -------------------------------------------------------------
                # Add to ChromaDB
                # -------------------------------------------------------------

                if all_docs:

                    collection.add(
                        documents=all_docs,
                        metadatas=all_metas,
                        ids=all_ids
                    )

                    st.success(
                        f"Added {len(all_docs)} chunk(s) "
                        "to knowledge base!"
                    )

    # -------------------------------------------------------------------------
    # Knowledge Base Metrics
    # -------------------------------------------------------------------------

    try:

        doc_count = collection.count()

    except Exception:

        doc_count = 0

    st.divider()

    col1, col2 = st.columns(2)

    col1.metric(
        "Indexed Chunks",
        doc_count
    )

    col2.metric(
        "Worker Threads",
        MAX_WORKERS
    )

    # -------------------------------------------------------------------------
    # Groq Status
    # -------------------------------------------------------------------------

    st.divider()

    st.subheader(
        "☁️ Cloud LLM"
    )

    if groq_client is not None:

        st.success(
            "Groq Connected"
        )

        st.caption(
            f"Model: {groq_model}"
        )

    else:

        st.error(
            "Groq API key not configured"
        )

        st.caption(
            "Add GROQ_API_KEY to your environment "
            "or Streamlit secrets."
        )


# -----------------------------------------------------------------------------
# 7. GROQ QUERY REWRITING
# -----------------------------------------------------------------------------

def rewrite_query_with_groq(
    question: str,
    chat_history: str
) -> str:

    """
    Convert conversational questions into standalone
    retrieval queries.

    Example:

        Previous:
        What is machine learning?

        Current:
        What are its types?

        Result:
        types of machine learning supervised
        unsupervised reinforcement learning
    """

    # ---------------------------------------------------------
    # If Groq unavailable
    # ---------------------------------------------------------

    if groq_client is None:

        return question

    prompt = f"""
You are the query rewriting component of a RAG system.

Rewrite the user's latest question into a clear,
standalone search query for retrieving information
from a document knowledge base.

Conversation history:
{chat_history}

Latest user question:
{question}

Rules:

1. Resolve pronouns such as:
   it, this, that, they, them.

2. Use conversation history when needed.

3. Preserve the original meaning.

4. Add useful keywords when appropriate.

5. Do not answer the question.

6. Do not explain anything.

7. Return ONLY the rewritten search query.

Example:

Conversation:
User: What is machine learning?
Assistant: Machine learning is...

User: What are its types?

Output:
types of machine learning supervised unsupervised reinforcement learning
"""

    try:

        response = groq_client.chat.completions.create(

            model=groq_model,

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You rewrite questions for "
                        "semantic document retrieval."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.1,

            max_tokens=120
        )

        rewritten = (
            response
            .choices[0]
            .message
            .content
            .strip()
        )

        rewritten = rewritten.strip(
            "\"'"
        )

        if not rewritten:

            return question

        return rewritten

    except Exception:

        return question


# -----------------------------------------------------------------------------
# 8. GROQ STREAMING RESPONSE
# -----------------------------------------------------------------------------

def stream_groq_response(
    messages,
    model,
    container
) -> str:

    """
    Stream Groq response token-by-token into Streamlit.
    """

    if groq_client is None:

        error_message = (
            "⚠️ Groq API is not configured.\n\n"
            "Please configure `GROQ_API_KEY`."
        )

        container.markdown(
            error_message
        )

        return error_message

    full_text = ""

    try:

        stream = groq_client.chat.completions.create(

            model=model,

            messages=messages,

            temperature=0.2,

            max_tokens=1000,

            stream=True
        )

        for chunk in stream:

            if not chunk.choices:

                continue

            delta = (
                chunk
                .choices[0]
                .delta
                .content
            )

            if delta:

                full_text += delta

                container.markdown(
                    full_text + "▌"
                )

        container.markdown(
            full_text
        )

        return full_text

    except Exception as e:

        error_message = (
            f"❌ Groq API Error:\n\n"
            f"`{str(e)}`"
        )

        container.markdown(
            error_message
        )

        return error_message


# -----------------------------------------------------------------------------
# 9. BUILD CHAT HISTORY FOR GROQ
# -----------------------------------------------------------------------------

def build_chat_history(
    messages,
    max_messages=10
):

    """
    Convert Streamlit message history into Groq messages.
    """

    history = []

    recent_messages = messages[
        -max_messages:
    ]

    for message in recent_messages:

        role = message.get(
            "role"
        )

        content = message.get(
            "content",
            ""
        )

        # -----------------------------------------------------
        # Remove source HTML from historical messages
        # -----------------------------------------------------

        if role in [
            "user",
            "assistant"
        ]:

            history.append(
                {
                    "role": role,
                    "content": content
                }
            )

    return history


# -----------------------------------------------------------------------------
# 10. MAIN APPLICATION UI
# -----------------------------------------------------------------------------

st.markdown(
    '<div class="main-title">🌌 AetherAI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'General Conversational Assistant & '
    'Specialized Knowledge Engine'
    '</div>',
    unsafe_allow_html=True
)


# -----------------------------------------------------------------------------
# 11. SESSION STATE
# -----------------------------------------------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []


# -----------------------------------------------------------------------------
# 12. DISPLAY PREVIOUS CHAT
# -----------------------------------------------------------------------------

for msg in st.session_state.messages:

    with st.chat_message(
        msg["role"]
    ):

        st.markdown(
            msg["content"],
            unsafe_allow_html=True
        )


# -----------------------------------------------------------------------------
# 13. CHAT INPUT
# -----------------------------------------------------------------------------

prompt = st.chat_input(
    "Ask Aether anything or query your uploaded documents..."
)


# -----------------------------------------------------------------------------
# 14. PROCESS USER QUESTION
# -----------------------------------------------------------------------------

if prompt:

    # -------------------------------------------------------------------------
    # Store User Message
    # -------------------------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message(
        "user"
    ):

        st.markdown(
            prompt
        )

    # -------------------------------------------------------------------------
    # Assistant Message
    # -------------------------------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        response_placeholder = st.empty()

        context = ""

        citations_list = []

        reranked_docs = []

        reranked_metas = []

        standalone_query = prompt

        # ---------------------------------------------------------------------
        # Determine if RAG should run
        # ---------------------------------------------------------------------

        should_retrieve = (
            enable_rag
            and doc_count > 0
        )

        # =====================================================================
        # RAG PIPELINE
        # =====================================================================

        if should_retrieve:

            with st.status(
                "Searching Knowledge Base...",
                expanded=False
            ) as status:

                try:

                    # ---------------------------------------------------------
                    # STEP 1: BUILD CONVERSATION HISTORY
                    # ---------------------------------------------------------

                    previous_messages = (
                        st.session_state.messages[:-1]
                    )

                    history_text_parts = []

                    for message in previous_messages:

                        history_text_parts.append(
                            f"{message['role']}: "
                            f"{message['content']}"
                        )

                    chat_history = "\n".join(
                        history_text_parts
                    )

                    # ---------------------------------------------------------
                    # STEP 2: QUERY REWRITING
                    # ---------------------------------------------------------

                    standalone_query = (
                        rewrite_query_with_groq(
                            question=prompt,
                            chat_history=chat_history
                        )
                    )

                    # ---------------------------------------------------------
                    # STEP 3: VECTOR RETRIEVAL
                    # ---------------------------------------------------------

                    v_docs, v_metas, v_ids = (
                        retrieve_documents(

                            query=standalone_query,

                            collection_name=(
                                "aether_knowledge_base"
                            ),

                            initial_top_k=initial_top_k,

                            # Important:
                            # This is the embedding model,
                            # NOT the Groq LLM.
                            model_name=EMBEDDING_MODEL
                        )
                    )

                    # ---------------------------------------------------------
                    # STEP 4: HYBRID SEARCH
                    # ---------------------------------------------------------

                    if enable_hybrid:

                        all_db = collection.get()

                        retrieved_docs, retrieved_metas = (
                            hybrid_retrieve(

                                query=standalone_query,

                                vector_docs=v_docs,

                                vector_metas=v_metas,

                                vector_ids=v_ids,

                                all_collection_docs=(
                                    all_db.get(
                                        "documents",
                                        []
                                    )
                                ),

                                all_collection_metas=(
                                    all_db.get(
                                        "metadatas",
                                        []
                                    )
                                ),

                                all_collection_ids=(
                                    all_db.get(
                                        "ids",
                                        []
                                    )
                                ),

                                top_k=initial_top_k
                            )
                        )

                    else:

                        retrieved_docs = v_docs

                        retrieved_metas = v_metas

                    # ---------------------------------------------------------
                    # STEP 5: CROSSENCODER RERANKING
                    # ---------------------------------------------------------

                    if retrieved_docs:

                        reranked_docs, reranked_metas = (
                            rerank_documents(

                                query=standalone_query,

                                documents=retrieved_docs,

                                metadatas=retrieved_metas,

                                top_k=final_top_k,

                                model_name=reranker_model
                            )
                        )

                    # ---------------------------------------------------------
                    # STEP 6: BUILD DOCUMENT CONTEXT
                    # ---------------------------------------------------------

                    if reranked_docs:

                        context_chunks = []

                        for idx, (
                            doc,
                            meta
                        ) in enumerate(
                            zip(
                                reranked_docs,
                                reranked_metas
                            ),
                            start=1
                        ):

                            source_file = meta.get(
                                "source",
                                "Document"
                            )

                            page_num = meta.get(
                                "page",
                                "N/A"
                            )

                            # -------------------------------------------------
                            # Context chunk
                            # -------------------------------------------------

                            context_chunks.append(
                                f"[Doc {idx}] "
                                f"Source: {source_file} "
                                f"(Page {page_num})\n"
                                f"{doc}"
                            )

                            # -------------------------------------------------
                            # Citation information
                            # -------------------------------------------------

                            citations_list.append(
                                {
                                    "index": idx,
                                    "source": source_file,
                                    "page": page_num,
                                    "excerpt": (
                                        doc[:180] + "..."
                                        if len(doc) > 180
                                        else doc
                                    )
                                }
                            )

                        context = "\n\n".join(
                            context_chunks
                        )

                        status.update(
                            label=(
                                "Relevant Documents Found!"
                            ),
                            state="complete"
                        )

                    else:

                        status.update(
                            label=(
                                "No document matches. "
                                "Using general knowledge."
                            ),
                            state="complete"
                        )

                except Exception as e:

                    status.update(
                        label=(
                            "Retrieval failed. "
                            "Using general mode."
                        ),
                        state="error"
                    )

                    st.warning(
                        f"Retrieval error: {e}"
                    )

        # =====================================================================
        # GENERAL CHAT / RAG PROMPT
        # =====================================================================

        if context:

            system_message = f"""
You are Aether, an expert AI assistant using
Retrieval-Augmented Generation.

Answer the user's question using ONLY the
provided document context.

IMPORTANT RULES:

1. Do not use outside knowledge for this answer.

2. Do not invent facts.

3. Do not make assumptions that are not supported
   by the provided context.

4. If the answer cannot be found in the context,
   say:

"The answer is not available in the uploaded documents."

5. Give a clear and useful explanation.

6. Cite supporting sources inline using:
[Doc 1], [Doc 2], etc.

7. Only cite documents that actually support the answer.

DOCUMENT CONTEXT:

{context}
"""

        else:

            system_message = """
You are Aether, a helpful and intelligent AI assistant.

Answer the user's question clearly, accurately,
and thoroughly using your general knowledge.

Be conversational and helpful.

Do not claim that information comes from uploaded
documents when no document context was retrieved.
"""

        # =====================================================================
        # BUILD GROQ MESSAGE LIST
        # =====================================================================

        groq_messages = [
            {
                "role": "system",
                "content": system_message
            }
        ]

        # ---------------------------------------------------------------------
        # Add previous conversation
        # ---------------------------------------------------------------------

        previous_chat = build_chat_history(
            st.session_state.messages[:-1],
            max_messages=10
        )

        for message in previous_chat:

            groq_messages.append(
                {
                    "role": message["role"],
                    "content": message["content"]
                }
            )

        # ---------------------------------------------------------------------
        # Current question
        # ---------------------------------------------------------------------

        groq_messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )

        # =====================================================================
        # STEP 7: GROQ RESPONSE STREAMING
        # =====================================================================

        raw_response = stream_groq_response(

            messages=groq_messages,

            model=groq_model,

            container=response_placeholder
        )

        # =====================================================================
        # STEP 8: HALLUCINATION / GROUNDING CHECK
        # =====================================================================

        badge_html = ""

        if context and raw_response:

            try:

                eval_result = check_hallucination(
                    context,
                    raw_response
                )

                score = float(
                    eval_result.get(
                        "score",
                        0
                    )
                )

                score_pct = int(
                    score * 100
                )

                is_grounded = eval_result.get(
                    "is_grounded",
                    False
                )

                if is_grounded:

                    badge_html = (
                        '<div class="badge-grounded">'
                        f'🟢 Grounding Confidence: '
                        f'{score_pct}%'
                        '</div>'
                    )

                else:

                    badge_html = (
                        '<div class="badge-warning">'
                        f'🔴 Grounding Warning: '
                        f'Low Confidence '
                        f'({score_pct}%)'
                        '</div>'
                    )

            except Exception as e:

                badge_html = (
                    '<div class="badge-warning">'
                    '⚠️ Grounding check unavailable'
                    '</div>'
                )

        # =====================================================================
        # STEP 9: SOURCE CITATIONS
        # =====================================================================

        citations_html = ""

        if citations_list:

            citations_html += (
                "\n\n### 📑 Source Citations\n"
            )

            for cite in citations_list:

                citations_html += f"""
<div class="citation-card">
    <strong>
        [Doc {cite['index']}] {cite['source']}
    </strong>
    (Page {cite['page']})
    <br/>
    <em style="color: #555;">
        "{cite['excerpt']}"
    </em>
</div>
"""

        # =====================================================================
        # FINAL RESPONSE
        # =====================================================================

        final_content = (
            f"{raw_response}\n\n"
            f"{badge_html}\n"
            f"{citations_html}"
        )

        response_placeholder.markdown(
            final_content,
            unsafe_allow_html=True
        )

        # =====================================================================
        # STEP 10: RETRIEVED CONTEXT INSPECTOR
        # =====================================================================

        if reranked_docs:

            with st.expander(
                "🔍 View Retrieved Context Chunks"
            ):

                for idx, (
                    doc,
                    meta
                ) in enumerate(
                    zip(
                        reranked_docs,
                        reranked_metas
                    ),
                    start=1
                ):

                    st.write(
                        f"**Chunk [{idx}]** — "
                        f"`{meta.get('source', 'Unknown')}` "
                        f"(Page "
                        f"{meta.get('page', 'N/A')})"
                    )

                    st.caption(
                        doc
                    )

                    st.divider()

        # =====================================================================
        # STEP 11: PIPELINE DETAILS
        # =====================================================================

        with st.expander(
            "⚙️ Pipeline Details"
        ):

            st.write(
                "**Original Query:**"
            )

            st.code(
                prompt
            )

            st.write(
                "**Rewritten Search Query:**"
            )

            st.code(
                standalone_query
            )

            st.write(
                "**Embedding Model:**"
            )

            st.code(
                EMBEDDING_MODEL
            )

            st.write(
                "**Groq LLM:**"
            )

            st.code(
                groq_model
            )

            st.write(
                "**Initial Retrieved Documents:**"
            )

            st.write(
                len(
                    retrieved_docs
                )
                if should_retrieve
                else 0
            )

            st.write(
                "**Final Reranked Chunks:**"
            )

            st.write(
                len(
                    reranked_docs
                )
            )

            st.write(
                "**Hybrid Search:**"
            )

            st.write(
                "Enabled"
                if enable_hybrid
                else "Disabled"
            )

            st.write(
                "**Reranker:**"
            )

            st.code(
                reranker_model
            )

        # =====================================================================
        # STEP 12: SAVE ASSISTANT RESPONSE
        # =====================================================================

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": final_content
            }
        )


# -----------------------------------------------------------------------------
# 15. FOOTER
# -----------------------------------------------------------------------------

st.divider()

st.caption(
    "AetherAI • "
    "RAG + Hybrid Search + CrossEncoder + "
    "Groq Cloud LLM + Source Attribution"
)