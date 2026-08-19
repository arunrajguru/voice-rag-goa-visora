import re
from typing import Dict, Any, Tuple

class AdaptiveChunkSelector:
    """Classifies queries and dynamically selects chunking parameters & weighting strategies.
    
    Query Categories:
    - exact: Specific code/ID/quote lookups (Favors lexical BM25 + sentence chunks)
    - factual: Specific facts, dates, names (Favors hybrid dense/sparse + sentence chunks)
    - semantic: Complex open-ended questions (Favors dense vector + semantic chunks)
    - broad: Wide topical overview queries (Favors fixed-size large context chunks)
    - unknown: Default hybrid balance
    """
    
    def classify_query(self, query: str) -> Tuple[str, str, float]:
        """Returns (query_category, recommended_strategy, recommended_dense_weight_alpha)"""
        q_lower = query.strip().lower()
        words = q_lower.split()
        
        # Exact lookup detection (quotes, codes, numbers)
        if '"' in query or re.search(r'\b\d{4,}\b', query) or any(term in q_lower for term in ['code', 'id', 'zip', 'exact']):
            return ("exact", "sentence", 0.3)
            
        # Factual query detection (who, when, where, what is, specific entity)
        if any(q_lower.startswith(w) for w in ['who ', 'when ', 'where ', 'what is ', 'what was ']):
            return ("factual", "sentence", 0.5)
            
        # Broad topic detection (overview, summary, list all, compare)
        if any(w in q_lower for w in ['overview', 'summary', 'compare', 'list', 'difference', 'explain']):
            return ("broad", "fixed", 0.6)

        # Semantic query detection (how to, why does, describe process)
        if any(q_lower.startswith(w) for w in ['how ', 'why ', 'explain ']) or len(words) > 8:
            return ("semantic", "semantic", 0.7)

        return ("unknown", "sentence", 0.6)
