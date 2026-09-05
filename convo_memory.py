# ============================================================
# convo_memory.py
# Conversation Memory Utilities
# ============================================================


MAX_HISTORY_MESSAGES = 10


# ============================================================
# ADD MESSAGE
# ============================================================

def add_message(
    history,
    role,
    content
):

    history.append(
        {
            "role": role,
            "content": content
        }
    )

    return trim_history(
        history
    )


# ============================================================
# TRIM HISTORY
# ============================================================

def trim_history(
    history,
    max_messages=MAX_HISTORY_MESSAGES
):

    if len(history) <= max_messages:

        return history

    return history[
        -max_messages:
    ]


# ============================================================
# GET CHAT HISTORY
# ============================================================

def get_recent_history(
    history,
    max_messages=MAX_HISTORY_MESSAGES
):

    return history[
        -max_messages:
    ]


# ============================================================
# FORMAT HISTORY
# ============================================================

def format_history(
    history
):

    if not history:

        return ""

    lines = []

    for message in history:

        role = message.get(
            "role",
            "user"
        )

        content = message.get(
            "content",
            ""
        )

        lines.append(
            f"{role}: {content}"
        )

    return "\n".join(
        lines
    )


# ============================================================
# CONDENSE QUESTION
# ============================================================

def condense_question(
    chat_history,
    latest_question,
    model_name=None
):

    # Import here to avoid circular imports
    from query_rewriter import rewrite_query

    return rewrite_query(
        latest_question=latest_question,
        chat_history=chat_history,
        model=model_name
        if model_name
        else "openai/gpt-oss-20b"
    )