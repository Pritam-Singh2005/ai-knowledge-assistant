# ============================================================
# hallucination_checker.py
# Lightweight Groundedness Checker
# ============================================================

import re


STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "of",
    "to",
    "in",
    "on",
    "for",
    "and",
    "or",
    "with",
    "that",
    "this",
    "these",
    "those",
    "it",
    "its",
    "as",
    "by",
    "from",
    "at",
    "about",
    "into",
    "than",
    "then",
    "also",
    "can",
    "may",
    "will",
    "would",
    "should",
    "could",
    "do",
    "does",
    "did",
    "not"
}


# ============================================================
# TOKENIZATION
# ============================================================

def normalize_text(text):

    text = text.lower()

    text = re.sub(
        r"\[doc\s*\d+\]",
        "",
        text
    )

    return text


def get_keywords(text):

    text = normalize_text(
        text
    )

    words = re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text
    )

    return {
        word
        for word in words
        if word not in STOPWORDS
        and len(word) > 2
    }


# ============================================================
# SENTENCE SPLIT
# ============================================================

def split_sentences(text):

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text.strip()
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


# ============================================================
# SENTENCE GROUNDING
# ============================================================

def sentence_grounding_score(
    sentence,
    context
):

    sentence_words = get_keywords(
        sentence
    )

    context_words = get_keywords(
        context
    )

    if not sentence_words:

        return 1.0

    overlap = (
        sentence_words &
        context_words
    )

    return (
        len(overlap) /
        len(sentence_words)
    )


# ============================================================
# HALLUCINATION CHECK
# ============================================================

def check_hallucination(
    context,
    answer,
    threshold=0.35
):

    if not answer:

        return {
            "score": 0.0,
            "is_grounded": False,
            "sentence_scores": [],
            "unsupported_sentences": [],
            "support_ratio": 0.0
        }

    if not context:

        return {
            "score": 0.0,
            "is_grounded": False,
            "sentence_scores": [],
            "unsupported_sentences": split_sentences(answer),
            "support_ratio": 0.0
        }

    sentences = split_sentences(
        answer
    )

    scores = []

    unsupported = []

    for sentence in sentences:

        score = sentence_grounding_score(
            sentence,
            context
        )

        scores.append(
            score
        )

        if score < threshold:

            unsupported.append(
                sentence
            )

    if scores:

        overall_score = (
            sum(scores) /
            len(scores)
        )

        support_ratio = (
            sum(
                1
                for score in scores
                if score >= threshold
            )
            /
            len(scores)
        )

    else:

        overall_score = 0.0

        support_ratio = 0.0

    is_grounded = (
        overall_score >= threshold
        and
        support_ratio >= 0.5
    )

    return {
        "score": overall_score,
        "is_grounded": is_grounded,
        "sentence_scores": scores,
        "unsupported_sentences": unsupported,
        "support_ratio": support_ratio
    }