"""
Query intent analyzer — detects if a query needs RAG or can be answered
without context from the vector database.
"""

import re
from typing import Tuple


class QueryAnalyzer:
    """Analyze query to determine if RAG is needed."""

    # Common simple question patterns
    SIMPLE_PATTERNS = [
        # Math/calculation: "what is 2+2"
        r"^what\s+(?:is|are)\s+[\d+\-*/.()]+",
        # Simple dates: "what year", "when was", "what date"
        r"^(?:what|when)\s+(?:year|date|time|day)[\w\s]*\?",
        # Definition/spelling: "how to spell", "define", "what does X mean"
        r"^(?:how\s+to|define|what\s+(?:does|is)\s+\w+\s+mean)\s",
        # Current info: "who is", "what is" (single word expected)
        r"^(?:who|what)\s+is\s+\w+\?$",
        # Yes/no questions with common answers
        r"^(?:do|does|is|are|can|could|will|would|should)\s+[\w\s]+\?$",
    ]

    # Phrases that indicate domain-specific content needed
    DOMAIN_INDICATORS = [
        "how to",
        "how can i",
        "how do i",
        "explain",
        "describe",
        "what is the difference",
        "compare",
        "what are the steps",
        "tutorial",
        "guide",
        "documentation",
        "example",
        "best practices",
        "performance",
        "optimization",
        "architecture",
        "design",
        "implementation",
    ]

    @staticmethod
    def needs_rag(query: str) -> Tuple[bool, str]:
        """
        Determine if a query needs RAG (context from vector DB).

        Args:
            query: User query string

        Returns:
            Tuple of (needs_rag, reason)
            - needs_rag: True if RAG should be used, False otherwise
            - reason: Short explanation of decision
        """
        if not query or len(query.strip()) < 5:
            return False, "query_too_short"

        query_lower = query.lower().strip()

        # Check for domain-specific indicators
        for indicator in QueryAnalyzer.DOMAIN_INDICATORS:
            if indicator in query_lower:
                return True, "contains_domain_indicator"

        # Check for simple question patterns
        for pattern in QueryAnalyzer.SIMPLE_PATTERNS:
            if re.match(pattern, query_lower, re.IGNORECASE):
                return False, "matches_simple_pattern"

        # By default, assume domain-specific content is needed
        # Better to be over-inclusive with RAG than to miss context
        return True, "default_assume_domain"

    @staticmethod
    def get_query_type(query: str) -> str:
        """
        Classify query type for potential optimization hints.

        Returns:
            'definition' | 'how-to' | 'comparison' | 'factual' | 'general' | 'time-sensitive'
        """
        query_lower = query.lower()

        # Check for time-sensitive keywords first
        if QueryAnalyzer.is_time_sensitive(query):
            return "time-sensitive"

        if any(p in query_lower for p in ["define", "what does", "what is", "mean"]):
            return "definition"
        if any(p in query_lower for p in ["how to", "how can", "how do"]):
            return "how-to"
        if any(p in query_lower for p in ["compare", "difference", "vs"]):
            return "comparison"
        if any(p in query_lower for p in ["when", "where", "who", "what year"]):
            return "factual"
        return "general"

    @staticmethod
    def is_time_sensitive(query: str) -> bool:
        """
        Detect if a query is asking about time-sensitive or recent information.

        Returns:
            True if query appears to need recent data, False otherwise
        """
        query_lower = query.lower()

        time_sensitive_keywords = [
            "latest",
            "recent",
            "today",
            "yesterday",
            "this week",
            "this month",
            "this year",
            "2024",
            "2025",
            "2026",
            "current",
            "now",
            "breaking",
            "news",
            "trending",
            "upcoming",
            "next week",
            "next month",
            "latest news",
            "what's new",
            "what is new",
            "recently",
            "just happened",
            "just released",
            "just announced",
            "real-time",
            "live",
            "stock",
            "price",
            "weather",
            "today's",
            "covid",
            "pandemic",
            "election",
            "outbreak",
        ]

        for keyword in time_sensitive_keywords:
            if keyword in query_lower:
                return True

        return False

    @staticmethod
    def select_search_mode(query: str) -> str:
        """
        Select optimal search strategy for the query.

        Returns:
            'hybrid' | 'faiss' | 'bm25'
            - 'hybrid': Use both indexes (best quality, slightly slower)
            - 'faiss': Semantic search only (fast, good for conceptual queries)
            - 'bm25': Keyword search only (fast, good for code/API searches)
        """
        query_lower = query.lower()
        words = query_lower.split()

        # Indicators for keyword-heavy queries (code, API names, error messages)
        keyword_indicators = [
            r"\.",  # Dots (e.g., "asyncio.sleep")
            r"-",   # Hyphens
            r"_",   # Underscores (snake_case)
            r"`",   # Code markers
            "error",
            "exception",
            "traceback",
            "import",
            "function",
            "method",
            "class",
            "module",
            "api",
            "endpoint",
            "parameter",
        ]

        keyword_score = 0
        for indicator in keyword_indicators:
            if indicator in query:
                keyword_score += 1

        # Indicators for semantic/conceptual queries
        semantic_indicators = [
            "explain",
            "describe",
            "understand",
            "difference between",
            "what is",
            "why",
            "how does",
            "concept",
            "architecture",
            "design",
            "pattern",
            "best practice",
            "good way",
            "better way",
        ]

        semantic_score = 0
        for indicator in semantic_indicators:
            if indicator in query_lower:
                semantic_score += 1

        # Short queries are likely keyword searches (code snippets, etc.)
        if len(words) <= 3:
            keyword_score += 2

        # Decisions based on scores
        if keyword_score > semantic_score and keyword_score >= 2:
            return "bm25"  # Keyword-heavy
        elif semantic_score > keyword_score and semantic_score >= 2:
            return "faiss"  # Semantic-heavy
        else:
            return "hybrid"  # Mixed or neutral
