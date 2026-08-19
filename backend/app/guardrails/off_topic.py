from app.models.data_models import GuardrailResult

OFF_TOPIC_INDICATORS = [
    "predict the stock price of", "write a python script for flappy bird",
    "recipe for chocolate cake", "who won the 2026 world cup"
]

class OffTopicGuardrail:
    """Detects queries that are completely unrelated to dataset domain."""
    def validate(self, query: str) -> GuardrailResult:
        q_lower = query.lower()
        for indicator in OFF_TOPIC_INDICATORS:
            if indicator in q_lower:
                return GuardrailResult(
                    passed=False,
                    reason=f"Off-topic indicator triggered: '{indicator}'",
                    refusal_message="This query is outside the scope of the indexed dataset (MSMARCO-XI)."
                )
        return GuardrailResult(passed=True)
