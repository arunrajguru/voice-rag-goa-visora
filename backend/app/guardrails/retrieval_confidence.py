from typing import List
from app.models.data_models import ChunkMetadata, GuardrailResult

class RetrievalConfidenceGuardrail:
    """Verifies that retrieved chunks satisfy minimum confidence/similarity thresholds."""
    def __init__(self, threshold: float = 0.20):
        self.threshold = threshold

    def validate(self, chunks: List[ChunkMetadata]) -> GuardrailResult:
        if not chunks:
            return GuardrailResult(
                passed=False,
                reason="No candidate chunks retrieved",
                refusal_message="I could not find relevant context in the knowledge base to answer your question."
            )
        
        max_score = max(c.score for c in chunks)
        if max_score < self.threshold:
            return GuardrailResult(
                passed=False,
                reason=f"Top retrieval score ({max_score:.3f}) below confidence threshold ({self.threshold})",
                refusal_message="I do not have sufficient reliable context to answer this question accurately."
            )
            
        return GuardrailResult(passed=True)
