# Comprehensive Guardrails Architecture

The system enforces five safety, quality, and grounding guardrails:

| Guardrail | Trigger Condition | Action / Output |
| :--- | :--- | :--- |
| **1. Input Safety** | Malicious injection, jailbreak attempts, unsafe words | Immediate refusal message |
| **2. Off-Topic** | Unrelated queries out of MSMARCO-XI dataset domain | Out-of-scope refusal message |
| **3. Retrieval Confidence** | Top candidate score $< T_{sim}$ ($0.25$) | Insufficient context refusal |
| **4. Grounding Verification** | Generated answer token overlap ratio $< T_{ground}$ ($0.40$) | Refuses ungrounded claims |
| **5. Output Validation** | Empty or malformed output | Fallback structured error |
