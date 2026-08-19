from typing import List
from app.models.data_models import ChunkMetadata, GuardrailResult
from app.utils.text_cleaner import tokenize

class GroundingGuardrail:
    """Verifies that generated answer claims are grounded in retrieved source context."""
    def __init__(self, grounding_threshold: float = 0.35):
        self.threshold = grounding_threshold

    def validate(self, answer: str, contexts: List[ChunkMetadata]) -> GuardrailResult:
        if not answer or not contexts:
            return GuardrailResult(passed=False, reason="Empty answer or context")

        # Refusal answers are by definition grounded refusal behavior
        if "insufficient" in answer.lower() or "cannot assist" in answer.lower() or "do not have" in answer.lower():
            return GuardrailResult(passed=True)

        context_tokens = set()
        for c in contexts:
            context_tokens.update(tokenize(c.text))

        answer_tokens = tokenize(answer)
        if not answer_tokens:
            return GuardrailResult(passed=False, reason="Answer contained no tokens")

        # Filter out common stop words to avoid false grounding metrics
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "of", "and", "in", "to", "for", "it", "that", "this", "by", "on", "with", "as", "at"}
        substantive_answer_tokens = [t for t in answer_tokens if t not in stopwords and len(t) > 2]

        if not substantive_answer_tokens:
            return GuardrailResult(passed=True)

        matches = sum(1 for t in substantive_answer_tokens if t in context_tokens)
        grounding_score = matches / len(substantive_answer_tokens)

        if grounding_score < self.threshold:
            return GuardrailResult(
                passed=False,
                reason=f"Grounding overlap ratio ({grounding_score:.2f}) below threshold ({self.threshold})",
                refusal_message="Answer could not be strictly grounded in retrieved documents."
            )

        return GuardrailResult(passed=True)
