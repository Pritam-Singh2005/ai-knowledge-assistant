# ============================================================
# query_rewriter.py
# Conversational Query Rewriting
# ============================================================

import os

from groq import Groq


DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"

FALLBACK_GROQ_MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b"
]


# ============================================================
# API KEY
# ============================================================

def get_api_key():

    api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if not api_key:

        raise RuntimeError(
            "GROQ_API_KEY is not available."
        )

    return api_key


# ============================================================
# CLIENT
# ============================================================

def get_client():

    return Groq(
        api_key=get_api_key()
    )


# ============================================================
# REWRITE QUERY
# ============================================================

def rewrite_query(
    latest_question,
    chat_history=None,
    model=DEFAULT_GROQ_MODEL
):

    if not latest_question:

        return ""

    chat_history = (
        chat_history or []
    )

    # No history means no rewriting required
    if not chat_history:

        return latest_question.strip()

    history_text = ""

    for message in chat_history[-6:]:

        role = message.get(
            "role",
            "user"
        )

        content = message.get(
            "content",
            ""
        )

        history_text += (
            f"{role}: {content}\n"
        )

    prompt = f"""
Rewrite the latest user question into a standalone
search query for a Retrieval-Augmented Generation system.

Conversation history:
{history_text}

Latest question:
{latest_question}

Rules:

- Resolve references such as "it", "this", "that",
  "they", "them", etc.
- Preserve the exact meaning.
- Do not answer the question.
- Do not add facts.
- Keep the query concise.
- Return ONLY the rewritten query.

Rewritten query:
"""

    messages = [
        {
            "role": "system",
            "content": (
                "You are a query rewriting component "
                "for a RAG system."
            )
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    client = get_client()

    models = []

    if model:
        models.append(model)

    for fallback in FALLBACK_GROQ_MODELS:

        if fallback not in models:

            models.append(
                fallback
            )

    for current_model in models:

        try:

            response = (
                client.chat.completions.create(
                    model=current_model,
                    messages=messages,
                    temperature=0,
                    max_tokens=150
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

        except Exception as e:

            error = str(e).lower()

            if (
                "404" in error
                or "model_not_found" in error
                or "does not exist" in error
            ):

                continue

            break

    # Safe fallback
    return latest_question.strip()


# ============================================================
# COMPATIBILITY FUNCTION
# ============================================================

def condense_question(
    chat_history,
    latest_question,
    model_name=DEFAULT_GROQ_MODEL
):

    return rewrite_query(
        latest_question=latest_question,
        chat_history=chat_history,
        model=model_name
    )