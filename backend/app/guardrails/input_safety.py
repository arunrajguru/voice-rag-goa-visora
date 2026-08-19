import re
from app.models.data_models import GuardrailResult

UNSAFE_KEYWORDS = [
    "drop table", "ignore previous instructions", "system prompt", "bypass",
    "hack", "exploit", "malware", "illegal", "suicide", "bomb", "weapon", "kill"
]

class InputSafetyGuardrail:
    """Verifies query safety, detecting prompt injections and harmful content."""
    def validate(self, query: str) -> GuardrailResult:
        q_lower = query.lower()
        for kw in UNSAFE_KEYWORDS:
            if kw in q_lower:
                return GuardrailResult(
                    passed=False,
                    reason=f"Query triggered safety keyword rule: '{kw}'",
                    refusal_message="I cannot assist with requests that violate safety and security guidelines."
                )
        if len(query.strip()) < 2:
            return GuardrailResult(
                passed=False,
                reason="Query too short",
                refusal_message="Please ask a complete and meaningful question."
            )
        return GuardrailResult(passed=True)
