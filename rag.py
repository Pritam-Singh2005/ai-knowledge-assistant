import ollama

from retriever import search_documents


MODEL_NAME = "llama3.2:1b"


def generate_answer(question):

    documents, metadatas = search_documents(
        question,
        top_k=3
    )

    if not documents:

        return (
            "I could not find relevant information "
            "in the provided documents.",
            []
        )

    context_parts = []

    for document, metadata in zip(
        documents,
        metadatas
    ):

        source = metadata.get(
            "source",
            "Unknown"
        )

        page = metadata.get(
            "page",
            "Unknown"
        )

        context_parts.append(
            f"""
Source: {source}
Page: {page}

Content:
{document}
"""
        )

    context = "\n\n".join(
        context_parts
    )

    prompt = f"""
You are an AI Knowledge Assistant.

Answer the question using ONLY the provided context.

Rules:

1. Do not invent information.
2. Do not use outside knowledge.
3. If the answer is not present in the context,
   say that the information was not found.
4. Give a clear and concise answer.

Context:
-------------------------
{context}
-------------------------

Question:
{question}

Answer:
"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response["message"]["content"]

    return answer, metadatas


if __name__ == "__main__":

    question = input(
        "Ask a question: "
    )

    answer, sources = generate_answer(
        question
    )

    print("\n🤖 Answer:")
    print(answer)

    print("\n📚 Sources:")

    for source in sources:

        print(
            f"- {source.get('source')} "
            f"(Page {source.get('page')})"
        )