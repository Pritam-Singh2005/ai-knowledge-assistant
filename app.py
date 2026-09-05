# =============================================================================
# Module: AetherAI — Production General & RAG Assistant
# Features: Dual Mode (General Chat + Document RAG), Async Streaming, 
#           Multithreaded Processing, Hybrid Retrieval & Faithfulness Checks
# =============================================================================

import os
import json
import asyncio
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Tuple

try:
    import httpx  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - handled at runtime in app startup
    httpx = None

import streamlit as st  # type: ignore[import-not-found]

try:
    import chromadb  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - handled at runtime in app startup
    chromadb = None

from langchain_community.document_loaders import PyPDFLoader  # type: ignore[import-not-found]
from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore[import-not-found]

# Modular RAG Components
from retriever import retrieve_documents
from reranker import rerank_documents
from hallucination_checker import check_hallucination
from hybrid_search import hybrid_retrieve
from convo_memory import condense_question

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom UI Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AetherAI — Intelligent Assistant",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Visual styling for cards, metrics, and indicators
st.markdown("""
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
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Parallel Processing & Persistent Client
# -----------------------------------------------------------------------------
MAX_WORKERS = min(32, (os.cpu_count() or 1) + 4)
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

@st.cache_resource
def get_chroma_client() -> Any:
    """Returns a single persistent client instance for ChromaDB."""
    if chromadb is None:
        raise RuntimeError("ChromaDB is not installed. Please install the chromadb package.")
    return chromadb.PersistentClient(path="./chroma_db")

chroma_client = get_chroma_client()
collection = chroma_client.get_or_create_collection(name="aether_knowledge_base")

# -----------------------------------------------------------------------------
# 3. Sidebar Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🌌 Aether Settings")
    
    st.subheader("🤖 Model Engine")
    llm_model = st.text_input("Ollama Model", value="llama3.2:1b")
    reranker_model = st.text_input("Reranker Model", value="cross-encoder/ms-marco-MiniLM-L-6-v2")

    st.divider()
    st.subheader("🔍 Knowledge Base (Optional)")
    enable_rag = st.toggle("Enable Document Grounding", value=True)
    
    with st.expander("Advanced Retrieval Parameters"):
        initial_top_k = st.slider("Initial Document Recall (K)", min_value=2, max_value=20, value=6)
        final_top_k = st.slider("Final Reranked Context (K)", min_value=1, max_value=8, value=3)
        enable_hybrid = st.checkbox("Enable Hybrid BM25 Fusion", value=True)

    uploaded_files = st.file_uploader(
        "Upload PDFs to extend Aether's memory", 
        type=["pdf"], 
        accept_multiple_files=True
    )

    # Worker function for background parsing
    def _parse_pdf(uploaded_file) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs, metas, ids = [], [], []

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            splits = text_splitter.split_documents(documents)

            for i, split in enumerate(splits):
                docs.append(split.page_content)
                metas.append({
                    "source": uploaded_file.name,
                    "page": split.metadata.get("page", 0) + 1
                })
                ids.append(f"{uploaded_file.name}_chunk_{i}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        return docs, metas, ids

    if uploaded_files and st.button("Index Documents", use_container_width=True):
        with st.spinner("Ingesting files in parallel..."):
            futures = [executor.submit(_parse_pdf, f) for f in uploaded_files]
            
            all_docs, all_metas, all_ids = [], [], []
            for future in futures:
                d, m, i = future.result()
                all_docs.extend(d)
                all_metas.extend(m)
                all_ids.extend(i)

            if all_docs:
                collection.add(documents=all_docs, metadatas=all_metas, ids=all_ids)
                st.success(f"Added {len(all_docs)} chunk(s) to knowledge base!")

    doc_count = collection.count()
    st.divider()
    col1, col2 = st.columns(2)
    col1.metric("Indexed Chunks", doc_count)
    col2.metric("Worker Threads", MAX_WORKERS)

# -----------------------------------------------------------------------------
# 4. Async Streamer
# -----------------------------------------------------------------------------
async def async_stream_response(prompt_text: str, model: str, container) -> str:
    """Streams response generation without blocking Streamlit's event loop."""
    url = "http://localhost:11434/api/generate"
    payload = {"model": model, "prompt": prompt_text, "stream": True}
    
    full_text = ""
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            async with client.stream("POST", url, json=payload) as response:
                if response.status_code == 200:
                    async for line in response.aiter_lines():
                        if line:
                            data = json.loads(line)
                            token = data.get("response", "")
                            full_text += token
                            container.markdown(full_text + "▌")
                else:
                    full_text = f"Error: Ollama API returned status code `{response.status_code}`."
        except Exception as e:
            full_text = f"Error connecting to Ollama: {e}"

    container.markdown(full_text)
    return full_text

# -----------------------------------------------------------------------------
# 5. Application Interface & Message State
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">🌌 AetherAI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">General Conversational Assistant & Specialized Knowledge Engine</div>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 6. Response Generation Pipeline
# -----------------------------------------------------------------------------
if prompt := st.chat_input("Ask Aether anything or query your uploaded documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        context = ""
        citations_list = []
        reranked_docs, reranked_metas = [], []

        # Check if we should attempt document retrieval
        should_retrieve = enable_rag and (doc_count > 0)

        if should_retrieve:
            with st.status("Searching Knowledge Base...", expanded=False) as status:
                try:
                    # Memory condensation
                    standalone_query = condense_question(
                        chat_history=st.session_state.messages[:-1],
                        latest_question=prompt,
                        model_name=llm_model
                    )

                    # Vector retrieval
                    v_docs, v_metas, v_ids = retrieve_documents(
                        query=standalone_query,
                        collection_name="aether_knowledge_base",
                        initial_top_k=initial_top_k,
                        model_name=llm_model
                    )

                    # BM25 Hybrid search
                    if enable_hybrid:
                        all_db = collection.get()
                        retrieved_docs, retrieved_metas = hybrid_retrieve(
                            query=standalone_query,
                            vector_docs=v_docs,
                            vector_metas=v_metas,
                            vector_ids=v_ids,
                            all_collection_docs=all_db.get("documents", []),
                            all_collection_metas=all_db.get("metadatas", []),
                            all_collection_ids=all_db.get("ids", []),
                            top_k=initial_top_k
                        )
                    else:
                        retrieved_docs, retrieved_metas = v_docs, v_metas

                    if retrieved_docs:
                        # Reranking
                        reranked_docs, reranked_metas = rerank_documents(
                            query=standalone_query,
                            documents=retrieved_docs,
                            metadatas=retrieved_metas,
                            top_k=final_top_k,
                            model_name=reranker_model
                        )

                        context_chunks = []
                        for idx, (doc, meta) in enumerate(zip(reranked_docs, reranked_metas), start=1):
                            source_file = meta.get("source", "Document")
                            page_num = meta.get("page", "N/A")
                            context_chunks.append(f"[Doc {idx}] Source: {source_file} (Page {page_num})\n{doc}")
                            citations_list.append({
                                "index": idx,
                                "source": source_file,
                                "page": page_num,
                                "excerpt": doc[:180] + "..." if len(doc) > 180 else doc
                            })
                        
                        context = "\n\n".join(context_chunks)
                        status.update(label="Relevant Documents Found!", state="complete")
                    else:
                        status.update(label="No document matches. Answering using general knowledge.", state="complete")

                except Exception as e:
                    status.update(label="Retrieval failed. Falling back to general mode.", state="error")

        # Dynamic System Prompt
        if context:
            system_prompt = f"""You are Aether, an expert AI assistant.
Answer the question using the provided document context. Cite sources inline using [Doc X].

Document Context:
{context}

Question: {prompt}
Answer:"""
        else:
            system_prompt = f"""You are Aether, a helpful, intelligent AI assistant.
Answer the user's question clearly, accurately, and thoroughly using your general knowledge.

Question: {prompt}
Answer:"""

        # Stream response
        raw_response = asyncio.run(
            async_stream_response(system_prompt, llm_model, response_placeholder)
        )

        # Grounding evaluation (only run when document context was used)
        badge_html = ""
        if context and raw_response:
            eval_result = check_hallucination(context, raw_response)
            score_pct = int(eval_result["score"] * 100)
            
            if eval_result["is_grounded"]:
                badge_html = f'<div class="badge-grounded">🟢 Grounding Confidence: {score_pct}%</div>'
            else:
                badge_html = f'<div class="badge-warning">🔴 Grounding Warning: Low Confidence ({score_pct}%)</div>'

        # Render citations if documents were used
        citations_html = ""
        if citations_list:
            citations_html += "\n\n### 📑 Source Citations\n"
            for cite in citations_list:
                citations_html += f"""
<div class="citation-card">
    <strong>[Doc {cite['index']}] {cite['source']}</strong> (Page {cite['page']})<br/>
    <em style="color: #555;">"{cite['excerpt']}"</em>
</div>
"""

        final_content = f"{raw_response}\n\n{badge_html}\n{citations_html}"
        response_placeholder.markdown(final_content, unsafe_allow_html=True)

        # Reranked inspect expander
        if reranked_docs:
            with st.expander("🔍 View Retrieved Context Chunks"):
                for idx, (doc, meta) in enumerate(zip(reranked_docs, reranked_metas), start=1):
                    st.write(f"**Chunk [{idx}]** — `{meta.get('source')}` (Page {meta.get('page')})")
                    st.caption(doc)
                    st.divider()

        st.session_state.messages.append({"role": "assistant", "content": final_content})