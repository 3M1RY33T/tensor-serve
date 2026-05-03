import json
import os
from pathlib import Path

CONFIG_FILE = "config.json"

DEFAULT_CONFIG = {
    "ai_provider": "openai-compatible",
    "ai_endpoint": None,
    "ai_model": None,
    "ai_api_key": None,
    "ai_api_key_header": "Authorization",
    "ai_api_key_prefix": "Bearer",
    "ai_extra_headers": {},
    "context_size": 3,
    "zim_source_folder": None,
    "relevance_threshold": 0.05,
    "query_analysis_enabled": True,
    "reranker_enabled": False,
}


def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return DEFAULT_CONFIG.copy()
    return DEFAULT_CONFIG.copy()


def save_config(config):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


def get_config_value(key):
    config = load_config()
    return config.get(key, DEFAULT_CONFIG.get(key))


def set_config_value(key, value):
    config = load_config()
    config[key] = value
    save_config(config)
