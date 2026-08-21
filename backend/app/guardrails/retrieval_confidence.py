from typing import List
from app.models.data_models import ChunkMetadata, GuardrailResult


class RetrievalConfidenceGuardrail:
    """Checks whether retrieved chunks are relevant enough to answer the question."""

    def __init__(self, threshold: float = 0.35):
        self.threshold = threshold

    def validate(self, chunks: List[ChunkMetadata]) -> GuardrailResult:
        # No chunks found
        if not chunks:
            return GuardrailResult(
                passed=False,
                reason="No relevant chunks found",
                refusal_message="Not Found"
            )

        # Get the best retrieval score
        max_score = max(c.score for c in chunks)

        # Reject low-confidence results
        if max_score < self.threshold:
            return GuardrailResult(
                passed=False,
                reason=(
                    f"Top retrieval score ({max_score:.3f}) "
                    f"is below threshold ({self.threshold})"
                ),
                refusal_message="Not Found"
            )

        # Relevant information was found
        return GuardrailResult(passed=True)
