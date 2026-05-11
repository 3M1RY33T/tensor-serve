"""
Search profiles for different deployment scenarios and complexity levels.
Allows users to select predefined profiles or manually configure search strategy.
"""

from typing import Dict, Optional

# Pre-configured profiles for common use cases
RERANKER_MODELS = {
    "lightweight": "ms-marco-MiniLM-L-6-v2",
    "balanced": "ms-marco-MiniLM-L-12-v2",
}


PROFILES = {
    "lightweight": {
        "description": "Minimal resource usage for local/embedded deployments",
        "keyword_backend": "bm25_okapi",
        "semantic_backend": "faiss_flat",
        "query_expansion_enabled": False,
        "query_expansion_type": "none",
        "reranker_enabled": False,
        "max_search_candidates": 50,
    },
    "balanced": {
        "description": "Default profile. Good quality/speed tradeoff for general use",
        "keyword_backend": "bm25_okapi",
        "semantic_backend": "faiss_flat",
        "query_expansion_enabled": False,
        "query_expansion_type": "none",
        "reranker_enabled": True,
        "reranker_model": "lightweight",
        "max_search_candidates": 150,
    },
    "production": {
        "description": "Advanced profile for enterprise servers. Best quality/scalability",
        "keyword_backend": "bm25_plus",
        "semantic_backend": "faiss_ivf",
        "query_expansion_enabled": True,
        "query_expansion_type": "prf",
        "reranker_enabled": True,
        "reranker_model": "balanced",
        "max_search_candidates": 300,
    },
}


def get_profile(profile_name: str) -> Optional[Dict]:
    """
    Get a predefined search profile by name.
    
    Args:
        profile_name: 'lightweight', 'balanced', or 'production'
    
    Returns:
        Profile dict, or None if not found
    """
    return PROFILES.get(profile_name)


def list_profiles() -> Dict[str, Dict]:
    """Get all available profiles with descriptions."""
    return PROFILES


def validate_manual_config(config: Dict) -> bool:
    """
    Validate a manual search configuration.
    
    Args:
        config: User-provided config dict
    
    Returns:
        True if valid, False otherwise
    """
    required_keys = {"keyword_backend", "semantic_backend"}
    if not required_keys.issubset(config.keys()):
        return False

    keyword_backend = config.get("keyword_backend")
    semantic_backend = config.get("semantic_backend")

    # Validate backends exist
    from api.search_backends import KEYWORD_BACKENDS, SEMANTIC_BACKENDS

    if keyword_backend not in KEYWORD_BACKENDS:
        print(f"Warning: keyword_backend '{keyword_backend}' not found")
        return False

    if semantic_backend not in SEMANTIC_BACKENDS:
        print(f"Warning: semantic_backend '{semantic_backend}' not found")
        return False

    return True


def merge_profile_with_overrides(
    profile_name: str, overrides: Optional[Dict] = None
) -> Dict:
    """
    Create a configuration by starting with a profile and applying overrides.
    
    Args:
        profile_name: Base profile ('lightweight', 'balanced', 'production')
        overrides: Optional dict of settings to override
    
    Returns:
        Merged configuration dict
    """
    profile = get_profile(profile_name)
    if profile is None:
        raise ValueError(f"Unknown profile: {profile_name}")

    config = profile.copy()
    if overrides:
        config.update(overrides)

    return config
