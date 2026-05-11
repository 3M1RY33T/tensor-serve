"""
Query expansion strategies to enhance search quality dynamically.
Expands user queries before retrieval to improve recall.
"""

import re
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple


class QueryExpander(ABC):
    """Abstract base for query expansion strategies."""

    @abstractmethod
    def expand(self, query: str, top_result: Optional[str] = None) -> str:
        """
        Expand or transform a query.
        
        Args:
            query: Original user query
            top_result: Optional top-1 retrieval result for feedback
        
        Returns:
            Expanded/transformed query string
        """
        pass


class NoOpExpander(QueryExpander):
    """Return query unchanged. Default/baseline."""

    def expand(self, query: str, top_result: Optional[str] = None) -> str:
        """Return original query."""
        return query


class PseudoRelevanceFeedbackExpander(QueryExpander):
    """
    Pseudo-relevance feedback: expand query with top terms from top-1 result.
    Improves recall on reformulation queries.
    """

    def __init__(self, max_expansion_terms: int = 5):
        """
        Args:
            max_expansion_terms: Max new terms to add from top-1 result
        """
        self.max_expansion_terms = max_expansion_terms

    def expand(self, query: str, top_result: Optional[str] = None) -> str:
        """Expand query with top terms from top result."""
        if not top_result:
            return query

        query_terms = set(query.lower().split())
        result_terms = top_result.lower().split()

        # Find novel high-frequency terms from result
        term_freq = {}
        for term in result_terms:
            if term not in query_terms and len(term) > 3:  # Skip short terms and query terms
                term_freq[term] = term_freq.get(term, 0) + 1

        # Add top expansion terms
        expansion_terms = sorted(
            term_freq.items(), key=lambda x: x[1], reverse=True
        )[: self.max_expansion_terms]
        expansion_text = " ".join([term for term, _ in expansion_terms])

        return f"{query} {expansion_text}" if expansion_text else query


class EntityExtractorExpander(QueryExpander):
    """
    Extract and weight named entities in the query.
    Prioritizes entity terms for keyword search.
    """

    def __init__(self):
        # Simple patterns for common entities
        self.entity_patterns = [
            (r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", "PERSON_OR_PLACE"),
            (r"\b(?:Python|JavaScript|Java|C\+\+|Go|Rust|TypeScript)\b", "PROGRAMMING_LANGUAGE"),
            (r"\b(?:API|REST|HTTP|TCP|UDP|SSL|TLS|JWT|OAuth)\b", "TECHNOLOGY"),
            (r"\b(?:bug|error|exception|crash|failure|timeout)\b", "ISSUE_TYPE"),
        ]

    def expand(self, query: str, top_result: Optional[str] = None) -> str:
        """Extract entities and boost their weight in query."""
        entities = []

        for pattern, entity_type in self.entity_patterns:
            matches = re.findall(pattern, query)
            if matches:
                entities.extend(matches)

        if entities:
            entity_boost = " ".join(entities).lower()
            return f"{query} {entity_boost}"

        return query


def get_expander(strategy: str) -> QueryExpander:
    """
    Factory function to get query expander by strategy name.
    
    Args:
        strategy: 'none', 'prf', or 'entity'
    
    Returns:
        QueryExpander instance
    """
    if strategy == "prf":
        return PseudoRelevanceFeedbackExpander()
    elif strategy == "entity":
        return EntityExtractorExpander()
    else:
        return NoOpExpander()
