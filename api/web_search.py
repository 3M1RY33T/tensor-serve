"""
Web search integration for time-sensitive and recent information.
Provides pluggable search providers with caching and result formatting.
"""

import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import requests


class SearchResult:
    """Represents a single search result."""

    def __init__(self, title: str, url: str, snippet: str, source: str = "web"):
        self.title = title
        self.url = url
        self.snippet = snippet
        self.source = source
        self.timestamp = time.time()

    def to_chunk_text(self) -> str:
        """Format as text chunk for context injection."""
        return f"[{self.source.upper()}] {self.title}\nURL: {self.url}\n{self.snippet}"

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
        }


class SearchProvider(ABC):
    """Abstract base class for search providers."""

    @abstractmethod
    def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        """
        Execute a web search query.

        Args:
            query: Search query string
            num_results: Number of results to return

        Returns:
            List of SearchResult objects
        """
        pass

    @abstractmethod
    def is_configured(self) -> bool:
        """Check if the provider is properly configured."""
        pass


class DuckDuckGoProvider(SearchProvider):
    """DuckDuckGo search provider (no API key required)."""

    def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        """Search using DuckDuckGo API."""
        try:
            url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "no_redirect": 1,
                "t": "tensor-serve",
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            results = []

            # Try to get results from AbstractResults
            if "AbstractResults" in data and data["AbstractResults"]:
                for item in data["AbstractResults"][: num_results]:
                    result = SearchResult(
                        title=item.get("Result", "No title"),
                        url=item.get("FirstURL", ""),
                        snippet=item.get("Text", "")[:300],
                        source="duckduckgo",
                    )
                    results.append(result)

            # Fallback to RelatedTopics if AbstractResults empty
            if not results and "RelatedTopics" in data:
                for item in data["RelatedTopics"][: num_results]:
                    if isinstance(item, dict) and "FirstURL" in item:
                        result = SearchResult(
                            title=item.get("Text", "No title")[:100],
                            url=item.get("FirstURL", ""),
                            snippet=item.get("Text", "")[:300],
                            source="duckduckgo",
                        )
                        results.append(result)

            return results[:num_results]
        except Exception as e:
            print(f"[web_search] DuckDuckGo search failed: {e}")
            return []

    def is_configured(self) -> bool:
        return True  # DuckDuckGo needs no configuration


class BraveSearchProvider(SearchProvider):
    """Brave Search provider (requires API key)."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        """Search using Brave Search API."""
        if not self.api_key:
            return []

        try:
            url = "https://api.search.brave.com/res/v1/web/search"
            headers = {"Accept": "application/json", "X-Subscription-Token": self.api_key}
            params = {"q": query, "count": num_results}
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            results = []
            if "web" in data:
                for item in data["web"][: num_results]:
                    result = SearchResult(
                        title=item.get("title", ""),
                        url=item.get("url", ""),
                        snippet=item.get("description", "")[:300],
                        source="brave",
                    )
                    results.append(result)

            return results
        except Exception as e:
            print(f"[web_search] Brave search failed: {e}")
            return []

    def is_configured(self) -> bool:
        return self.api_key is not None


class GoogleCustomSearchProvider(SearchProvider):
    """Google Custom Search provider (requires API key and search engine ID)."""

    def __init__(self, api_key: Optional[str] = None, search_engine_id: Optional[str] = None):
        self.api_key = api_key
        self.search_engine_id = search_engine_id

    def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        """Search using Google Custom Search API."""
        if not self.api_key or not self.search_engine_id:
            return []

        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "q": query,
                "key": self.api_key,
                "cx": self.search_engine_id,
                "num": min(num_results, 10),
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            results = []
            if "items" in data:
                for item in data["items"][: num_results]:
                    result = SearchResult(
                        title=item.get("title", ""),
                        url=item.get("link", ""),
                        snippet=item.get("snippet", "")[:300],
                        source="google",
                    )
                    results.append(result)

            return results
        except Exception as e:
            print(f"[web_search] Google Custom Search failed: {e}")
            return []

    def is_configured(self) -> bool:
        return self.api_key is not None and self.search_engine_id is not None


class WebSearchManager:
    """Manages web search with provider selection and caching."""

    def __init__(self, default_provider: str = "duckduckgo"):
        self.default_provider = default_provider
        self.providers: Dict[str, SearchProvider] = {
            "duckduckgo": DuckDuckGoProvider(),
        }
        self.search_cache: Dict[str, tuple] = {}  # (results, timestamp)
        self.cache_ttl = 3600  # 1 hour

    def set_brave_api_key(self, api_key: str):
        """Configure Brave Search provider."""
        self.providers["brave"] = BraveSearchProvider(api_key)

    def set_google_api_key(self, api_key: str, search_engine_id: str):
        """Configure Google Custom Search provider."""
        self.providers["google"] = GoogleCustomSearchProvider(api_key, search_engine_id)

    def reset(self):
        """Clear configured providers and cached web search results."""
        self.default_provider = "duckduckgo"
        self.providers = {
            "duckduckgo": DuckDuckGoProvider(),
        }
        self.search_cache.clear()

    def search(
        self,
        query: str,
        num_results: int = 3,
        use_cache: bool = True,
        provider: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Execute web search with provider fallback.

        Args:
            query: Search query
            num_results: Number of results to return
            use_cache: Whether to use cached results
            provider: Specific provider to use (defaults to auto-select)

        Returns:
            List of SearchResult objects
        """
        cache_key = f"{query}:{num_results}:{provider or 'auto'}"

        # Check cache
        if use_cache and cache_key in self.search_cache:
            results, timestamp = self.search_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return results

        # Select provider
        if provider and provider in self.providers:
            selected_provider = self.providers[provider]
        else:
            # Try configured providers in order, fallback to DuckDuckGo
            for prov_name in ["brave", "google", "duckduckgo"]:
                if prov_name in self.providers and self.providers[prov_name].is_configured():
                    selected_provider = self.providers[prov_name]
                    break
            else:
                selected_provider = self.providers["duckduckgo"]

        results = selected_provider.search(query, num_results)

        # Cache results
        if results:
            self.search_cache[cache_key] = (results, time.time())

        return results

    def get_available_providers(self) -> Dict[str, bool]:
        """Get list of available and configured providers."""
        return {name: provider.is_configured() for name, provider in self.providers.items()}


# Global instance
web_search_manager = WebSearchManager()
