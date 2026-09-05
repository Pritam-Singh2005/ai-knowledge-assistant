import ollama
import re


MODEL_NAME = "llama3.2:3b"


def rewrite_query(query, chat_history=None):

    query = query.strip()

    if not query:
        return query

    history_text = ""

    if chat_history:

        recent_history = chat_history[-6:]

        for message in recent_history:

            role = message.get("role", "")
            content = message.get("content", "")

            if content:
                history_text += (
                    f"{role}: {content}\n"
                )

    prompt = f"""
You are a query rewriting system for a RAG document chatbot.

Your job is to rewrite the user's question into a concise,
document-search query.

Rules:

1. Preserve the user's original meaning.
2. Use conversation history when the question contains words
   such as "it", "they", "this", "that", "its", etc.
3. Add important keywords when useful.
4. Do not answer the question.
5. Return ONLY the rewritten search query.
6. Do not use quotes.
7. Do not add explanations.

Conversation history:
{history_text}

Current user question:
{query}

Rewritten search query:
"""

    try:

        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={
                "temperature": 0.0
            }
        )

        rewritten = response["message"]["content"].strip()

        # Remove accidental labels
        rewritten = re.sub(
            r"^(rewritten search query|search query)\s*:\s*",
            "",
            rewritten,
            flags=re.IGNORECASE
        )

        rewritten = rewritten.strip()

        if not rewritten:
            return query

        return rewritten

    except Exception as e:

        print(
            "Query rewriting error:",
            e
        )

        # Safe fallback
        return query