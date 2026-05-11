import base64
import hashlib
import json
import os
from pathlib import Path

CONFIG_FILE = "config.json"
SECRET_KEY_FILE = ".tensor_config.key"

SECRET_CONFIG_FIELDS = {
    "ai_api_key",
    "web_search_api_key",
    "ai_extra_headers",
}

ENCRYPTED_MARKER = "__tensor_encrypted__"

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
    "reranker_model": "lightweight",  # lightweight | balanced | production
    "web_search_enabled": False,
    "web_search_provider": "duckduckgo",
    "web_search_api_key": None,
    "web_search_engine_id": None,
    "web_search_results": 3,
    "keyword_search_mode": "auto",     # auto | web | zim | off
    "semantic_search_mode": "auto",    # auto | on | off
    # Search profile and backend configuration
    "search_profile": "balanced",  # balanced | lightweight | production | manual
    "keyword_backend": "bm25_okapi",  # bm25_okapi | bm25_plus
    "semantic_backend": "faiss_flat",  # faiss_flat | faiss_ivf
    "max_search_candidates": None,  # None = use profile default
    "query_expansion_enabled": False,
    "query_expansion_type": "none",  # none | prf | entity
}


def _get_fernet():
    try:
        from cryptography.fernet import Fernet
    except ImportError as exc:
        raise RuntimeError(
            "Secret config encryption requires the 'cryptography' package. "
            "Install dependencies with 'pip install -r requirements.txt'."
        ) from exc

    env_secret = os.environ.get("TENSOR_CONFIG_KEY")
    if env_secret:
        digest = hashlib.sha256(env_secret.encode("utf-8")).digest()
        return Fernet(base64.urlsafe_b64encode(digest))

    key_file = os.environ.get("TENSOR_CONFIG_KEY_FILE")
    key_path = (
        Path(key_file) if key_file else Path(CONFIG_FILE).with_name(SECRET_KEY_FILE)
    )

    if key_path.exists():
        key = key_path.read_bytes().strip()
    else:
        key_path.parent.mkdir(parents=True, exist_ok=True)
        key = Fernet.generate_key()
        key_path.write_bytes(key)
        try:
            key_path.chmod(0o600)
        except OSError:
            pass

    return Fernet(key)


def _encrypted_value(value):
    return isinstance(value, dict) and value.get(ENCRYPTED_MARKER) is True


def _encrypt_value(value):
    if value is None or value == "" or value == {} or _encrypted_value(value):
        return value

    ciphertext = (
        _get_fernet().encrypt(json.dumps(value).encode("utf-8")).decode("utf-8")
    )
    return {
        ENCRYPTED_MARKER: True,
        "version": 1,
        "ciphertext": ciphertext,
    }


def _decrypt_value(value):
    if not _encrypted_value(value):
        return value

    try:
        from cryptography.fernet import InvalidToken
    except ImportError as exc:
        raise RuntimeError(
            "Secret config decryption requires the 'cryptography' package. "
            "Install dependencies with 'pip install -r requirements.txt'."
        ) from exc

    try:
        plaintext = _get_fernet().decrypt(value["ciphertext"].encode("utf-8"))
        return json.loads(plaintext.decode("utf-8"))
    except InvalidToken as exc:
        raise RuntimeError(
            "Could not decrypt secret config value. Check TENSOR_CONFIG_KEY "
            "or TENSOR_CONFIG_KEY_FILE."
        ) from exc


def _decrypt_config(config):
    decrypted = {**DEFAULT_CONFIG, **config}
    for key in SECRET_CONFIG_FIELDS:
        decrypted[key] = _decrypt_value(decrypted.get(key))
    return decrypted


def _encrypt_config(config):
    encrypted = {**DEFAULT_CONFIG, **_decrypt_config(config)}
    for key in SECRET_CONFIG_FIELDS:
        encrypted[key] = _encrypt_value(encrypted.get(key))
    return encrypted


def _has_plaintext_secret(config):
    for key in SECRET_CONFIG_FIELDS:
        value = config.get(key)
        if value not in (None, "", {}) and not _encrypted_value(value):
            return True
    return False


def mask_config(config):
    """Return a copy of config with secrets replaced by safe status values."""
    masked = dict(config)
    masked["ai_api_key_configured"] = bool(masked.get("ai_api_key"))
    masked["web_search_api_key_configured"] = bool(masked.get("web_search_api_key"))
    masked.pop("ai_api_key", None)
    masked.pop("web_search_api_key", None)
    masked["ai_extra_headers"] = {
        key: "<configured>" for key in (masked.get("ai_extra_headers") or {})
    }
    return masked


def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                raw_config = json.load(f)
            decrypted = _decrypt_config(raw_config)
            if _has_plaintext_secret(raw_config):
                save_config(decrypted)
            return decrypted
        except (OSError, json.JSONDecodeError):
            return DEFAULT_CONFIG.copy()
    return DEFAULT_CONFIG.copy()


def save_config(config):
    encrypted = _encrypt_config(config)
    with open(CONFIG_FILE, "w") as f:
        json.dump(encrypted, f, indent=2)


def reset_config():
    """Reset config.json back to default settings and return the new config."""
    save_config(DEFAULT_CONFIG.copy())
    return load_config()


def get_config_value(key):
    config = load_config()
    return config.get(key, DEFAULT_CONFIG.get(key))


def set_config_value(key, value):
    config = load_config()
    config[key] = value
    save_config(config)
