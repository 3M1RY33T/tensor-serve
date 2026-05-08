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
    def select_search_mode(query: str, keyword_mode: str = "auto", semantic_mode: str = "auto") -> str:
        """
        Select optimal search strategy for the query, respecting user preferences.

        Args:
            query: The search query
            keyword_mode: 'auto' (decide based on query) | 'web' (web only) | 'zim' (ZIM only) | 'off' (no keyword search)
            semantic_mode: 'auto' (decide based on query) | 'on' (force semantic search) | 'off' (no semantic search)

        Returns:
            'hybrid' | 'faiss' | 'bm25' | 'web' | None
            - 'hybrid': Use both FAISS + BM25 indexes
            - 'faiss': Semantic search only
            - 'bm25': Keyword search only
            - 'web': Web search only (keyword_mode="web")
            - None: No search (both modes are off)
        """
        # Handle explicit "off" modes first
        if keyword_mode == "off" and semantic_mode == "off":
            return None
        if keyword_mode == "web" and semantic_mode == "off":
            return "web"
        if keyword_mode == "off" and semantic_mode == "on":
            return "faiss"
        if keyword_mode == "zim" and semantic_mode == "off":
            return "bm25"

        # Helper function to auto-detect query characteristics
        def _analyze_query(q: str):
            query_lower = q.lower()
            words = query_lower.split()
            
            keyword_indicators = [r"\.", r"-", r"_", r"`", "error", "exception", "traceback",
                                  "import", "function", "method", "class", "module", "api", "endpoint", "parameter"]
            keyword_score = sum(1 for ind in keyword_indicators if ind in q)
            if len(words) <= 3:
                keyword_score += 2
            
            semantic_indicators = ["explain", "describe", "understand", "difference between", "what is",
                                   "why", "how does", "concept", "architecture", "design", "pattern",
                                   "best practice", "good way", "better way"]
            semantic_score = sum(1 for ind in semantic_indicators if ind in query_lower)
            
            return keyword_score, semantic_score

        # For "auto" modes, detect query characteristics
        if keyword_mode == "auto" and semantic_mode == "auto":
            kw_score, sem_score = _analyze_query(query)
            if kw_score > sem_score and kw_score >= 2:
                return "bm25"
            elif sem_score > kw_score and sem_score >= 2:
                return "faiss"
            else:
                return "hybrid"

        # Handle partial auto modes
        if keyword_mode == "auto":
            kw_score, _ = _analyze_query(query)
            use_keyword = kw_score >= 2
        else:
            use_keyword = keyword_mode in ("zim", "web")

        if semantic_mode == "auto":
            _, sem_score = _analyze_query(query)
            use_semantic = sem_score >= 2
        else:
            use_semantic = semantic_mode == "on"

        # Handle special keyword_mode="web"
        if keyword_mode == "web":
            if use_semantic:
                return "hybrid_web"  # Web + semantic
            else:
                return "web"  # Web only

        # Combine results based on use_keyword and use_semantic
        if use_keyword and use_semantic:
            return "hybrid"
        elif use_semantic:
            return "faiss"
        elif use_keyword:
            return "bm25"
        else:
            return None
