import re

def clean_text(text: str) -> str:
    """Normalize whitespace and strip control characters from document or query text."""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def tokenize(text: str) -> list[str]:
    """Simple lowercase alphanumeric tokenization for BM25 processing."""
    cleaned = clean_text(text).lower()
    tokens = re.findall(r'\b\w+\b', cleaned)
    return tokens
