# -----------------------------------------------------------------------------
# STEP 0: MUST BE AT THE VERY TOP (Fixes Ragas missing vertexai module crash)
# -----------------------------------------------------------------------------
import sys
import types

if "langchain_community.chat_models.vertexai" not in sys.modules:
    mock_vertex = types.ModuleType("langchain_community.chat_models.vertexai")
    class DummyChatVertexAI:
        pass
    mock_vertex.ChatVertexAI = DummyChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = mock_vertex

# -----------------------------------------------------------------------------
# STANDARD IMPORTS
# -----------------------------------------------------------------------------
import os
import requests
import pandas as pd
from datasets import Dataset

# Ragas imports
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# Import metric classes (supports both legacy and new Ragas releases)
try:
    from ragas.metrics import (
        Faithfulness,
        AnswerRelevancy,
        ContextRecall,
        ContextPrecision
    )
except ImportError:
    from ragas.metrics.collections import (
        Faithfulness,
        AnswerRelevancy,
        ContextRecall,
        ContextPrecision
    )

# LangChain local models
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

# Import custom retriever & pipeline functions
from retriever import retrieve_documents
from reranker import rerank_documents


# -----------------------------------------------------------------------------
# 1. Setup Local LLM & Embeddings Evaluator Models
# -----------------------------------------------------------------------------
print("Initializing local Ollama evaluator models for Ragas...")

eval_llm = ChatOllama(model="llama3.2:1b", base_url="http://localhost:11434")
eval_embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")

# Wrap inside Ragas wrappers
ragas_llm = LangchainLLMWrapper(eval_llm)
ragas_embeddings = LangchainEmbeddingsWrapper(eval_embeddings)


# -----------------------------------------------------------------------------
# 2. Test Dataset Definition
# -----------------------------------------------------------------------------
eval_samples = [
    {
        "question": "What is machine learning?",
        "ground_truth": "Machine learning is a branch of artificial intelligence focused on building applications that learn from data and improve accuracy over time without being explicitly programmed."
    },
    {
        "question": "What is deep learning?",
        "ground_truth": "Deep learning is a subset of machine learning based on artificial neural networks with multiple layers, used for complex pattern recognition."
    }
]


# -----------------------------------------------------------------------------
# 3. Pipeline Execution
# -----------------------------------------------------------------------------
def generate_rag_response(query: str):
    """Executes full RAG workflow and collects values for evaluation."""
    retrieved_docs, retrieved_metas = retrieve_documents(
        query=query,
        collection_name="pdf_collection",
        initial_top_k=6,
        model_name="llama3.2:1b"
    )

    reranked_docs, _ = rerank_documents(
        query=query,
        documents=retrieved_docs,
        metadatas=retrieved_metas,
        top_k=3
    )

    context_str = "\n\n".join(reranked_docs) if reranked_docs else "No context available."

    prompt = f"""Use ONLY the following context to answer the question.
Context:
{context_str}

Question: {query}
Answer:"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.2:1b", "prompt": prompt, "stream": False},
            timeout=30
        )
        answer = response.json().get("response", "Error generating response.") if response.status_code == 200 else ""
    except Exception as e:
        answer = f"Ollama execution failed: {e}"

    return answer, reranked_docs


print("Running RAG pipeline over test samples...")
dataset_dict = {
    "question": [],
    "contexts": [],
    "response": [],
    "ground_truth": []
}

for sample in eval_samples:
    q = sample["question"]
    gt = sample["ground_truth"]

    generated_answer, contexts = generate_rag_response(q)

    dataset_dict["question"].append(q)
    dataset_dict["contexts"].append(contexts)
    dataset_dict["response"].append(generated_answer)
    dataset_dict["ground_truth"].append(gt)

eval_dataset = Dataset.from_dict(dataset_dict)


# -----------------------------------------------------------------------------
# 4. Run Ragas Evaluation
# -----------------------------------------------------------------------------
print("Running Ragas evaluation metrics...")

# Instantiated metric objects
metrics = [
    Faithfulness(),
    AnswerRelevancy(),
    ContextPrecision(),
    ContextRecall()
]

results = evaluate(
    dataset=eval_dataset,
    metrics=metrics,
    llm=ragas_llm,
    embeddings=ragas_embeddings
)


# -----------------------------------------------------------------------------
# 5. Output and Analysis
# -----------------------------------------------------------------------------
print("\n=== FINAL EVALUATION SCORES ===")
print(results)

df_results = results.to_pandas()
df_results.to_csv("rag_evaluation_report.csv", index=False)
print("\nFull evaluation breakdown saved to 'rag_evaluation_report.csv'")