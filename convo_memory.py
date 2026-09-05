# =============================================================================
# Module: Context-Aware Conversational Memory
# Role: Rewrites follow-up questions into standalone vector queries.
# =============================================================================

import requests
from typing import List, Dict

def condense_question(
    chat_history: List[Dict[str, str]], 
    latest_question: str, 
    model_name: str = "llama3.2:1b",
    ollama_url: str = "http://localhost:11434"
) -> str:
    """
    Converts a follow-up user query into an independent standalone query using chat history.
    """
    if not chat_history:
        return latest_question

    # Format previous turns (up to last 3 exchanges)
    formatted_history = ""
    for msg in chat_history[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        formatted_history += f"{role}: {msg['content']}\n"

    prompt = f"""Given the following conversation history and a follow-up question, rephrase the follow-up question into a standalone question that can be understood without the conversation history. Do NOT answer the question.

Chat History:
{formatted_history}

Follow-up Question: {latest_question}
Standalone Question:"""

    try:
        response = requests.post(
            f"{ollama_url}/api/generate",
            json={"model": model_name, "prompt": prompt, "stream": False},
            timeout=10
        )
        if response.status_code == 200:
            standalone = response.json().get("response", "").strip()
            return standalone if standalone else latest_question
    except Exception as e:
        print(f"Memory condensation failed ({e}), using raw query.")

    return latest_question