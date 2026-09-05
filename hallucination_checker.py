# =============================================================================
# Module: Hallucination & Faithfulness Evaluator
# Role: Verifies if the LLM generated response is grounded in provided context.
# =============================================================================

from sentence_transformers import CrossEncoder

class HallucinationChecker:
    def __init__(self, model_name: str = "vectara/hallucination_evaluation_model"):
        """
        Initializes an entailment/cross-encoder model trained to detect hallucinations.
        Higher score = higher alignment/factual grounding with context.
        """
        print(f"Loading Hallucination Checking Model: {model_name}...")
        # Fallback to standard cross-encoder if custom model isn't available
        try:
            self.model = CrossEncoder(model_name, max_length=512)
        except Exception as e:
            print(f"Failed loading {model_name} ({e}), falling back to default cross-encoder.")
            self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def check_faithfulness(self, context: str, response: str, threshold: float = 0.5) -> dict:
        """
        Evaluates the generated response against retrieved context.
        Returns score, pass/fail status, and a warnings list.
        """
        if not context or not response:
            return {
                "score": 0.0,
                "is_grounded": False,
                "warning": "Empty context or response provided."
            }

        # Model predicts alignment score between context and response
        pair = [[context, response]]
        score = float(self.model.predict(pair)[0])

        # Normalize score if model outputs unscaled logits
        if score < 0 or score > 1:
            import math
            score = 1 / (1 + math.exp(-score))  # Sigmoid scaling

        is_grounded = score >= threshold

        return {
            "score": round(score, 3),
            "is_grounded": is_grounded,
            "warning": None if is_grounded else "⚠️ Warning: Potential hallucination or weak context grounding detected."
        }


# Singleton pattern for cached model reuse
_checker_instance = None

def check_hallucination(context: str, response: str, threshold: float = 0.5) -> dict:
    """Convenience function interface."""
    global _checker_instance
    if _checker_instance is None:
        _checker_instance = HallucinationChecker()
    return _checker_instance.check_faithfulness(context, response, threshold)


if __name__ == "__main__":
    # Test Run
    sample_context = "Machine learning is a field of study in artificial intelligence."
    sample_answer_good = "Machine learning is a part of AI."
    sample_answer_bad = "Machine learning was invented in France during the 18th century."

    print("Grounded Test:", check_hallucination(sample_context, sample_answer_good))
    print("Hallucination Test:", check_hallucination(sample_context, sample_answer_bad))