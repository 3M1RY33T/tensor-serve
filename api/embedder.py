import threading
import warnings

from sentence_transformers import SentenceTransformer

from api.micro_batch import MicroBatcher

# Word-pieces reserved for the model's own [CLS] / [SEP] markers.
_SPECIAL_TOKEN_BUDGET = 2

# Embedding backends, measured on all-MiniLM-L6-v2 (Apple M-series, 10 cores):
#
#   backend      ms/query   chunks/s bulk   retrieval quality vs torch
#   torch            4.00            1838   --
#   onnx             1.19            3433   identical
#   onnx-int8        0.95            2878   passage recall -3.3pp, P@1 -3.4pp
#
# So 'onnx' is free and 'onnx-int8' is not: int8 buys another 0.37ms per query
# and costs measurable recall, which is why 'auto' picks fp32 for queries.
#
# The ranking inverts for bulk work. Those figures are single short queries; on
# real 250-token chunks embedded in batches of 512, the order reverses:
#
#   backend      chunks/s on real chunks
#   torch                            323
#   onnx                             127
#
# PyTorch batches long sequences far better, so ingestion pins torch while
# serving prefers ONNX. The two produce the same vectors (cosine 1.000000), so
# an index built with one is queried correctly by the other.
#
# ONNX Runtime is pinned to the CPU execution provider. Left to choose, it
# selects CoreML on macOS, which supports only 294 of the graph's 418 nodes and
# pays partitioning overhead on every call — that measured 10.67 ms/query, two
# and a half times *slower* than PyTorch.
BACKENDS = {
    "torch": {},
    "onnx": {
        "backend": "onnx",
        "model_kwargs": {"file_name": "onnx/model.onnx", "provider": "CPUExecutionProvider"},
    },
    "onnx-int8": {
        "backend": "onnx",
        "model_kwargs": {
            "file_name": "onnx/model_qint8_arm64.onnx",
            "provider": "CPUExecutionProvider",
        },
    },
}

DEFAULT_BACKEND = "auto"


def _resolve_auto(workload: str = "query") -> str:
    """
    Pick the backend that is fastest for this kind of work.

    Queries are single short texts, where ONNX Runtime measured 3.6x faster end
    to end with retrieval quality unchanged. Bulk embedding is long chunks in
    large batches, where PyTorch is 2.5x faster. Anyone without the optional
    ONNX runtime keeps torch for both.
    """
    if workload == "bulk":
        return "torch"
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return "torch"
    return "onnx"


def _configured_backend() -> str:
    try:
        from api.config import get_config_value

        return get_config_value("embedding_backend") or DEFAULT_BACKEND
    except Exception:
        return DEFAULT_BACKEND


class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2", backend: str = None,
                 workload: str = "query"):
        """
        Args:
            model_name: HuggingFace model id.
            backend:    'torch', 'onnx' or 'onnx-int8'. Defaults to the
                        configured embedding_backend. A backend whose runtime is
                        not installed falls back to torch with a warning rather
                        than failing to start.
            workload:   'query' (single short texts) or 'bulk' (ingestion).
                        Only consulted when the backend resolves to 'auto';
                        the fastest backend differs between the two.
        """
        self.workload = workload
        self.backend = backend or _configured_backend()
        if self.backend == "auto":
            self.backend = _resolve_auto(workload)
        if self.backend not in BACKENDS:
            raise ValueError(
                f"Unknown embedding backend '{self.backend}'. "
                f"Choose one of: auto, {', '.join(BACKENDS)}"
            )

        self.model_name = model_name
        self.model = self._load(self.backend, model_name)
        self._warned_truncation = False
        # The model and its tokenizer are shared by every request. HuggingFace's
        # fast tokenizer is a Rust object behind a runtime borrow check, and
        # concurrent use raises "RuntimeError: Already borrowed" — measured at 4
        # failures in 200 encodes across 8 threads. Retrieval runs in a thread
        # pool, so two chat requests that both miss the cache land here at once.
        self._lock = threading.Lock()
        # Concurrent single-query encodes queue behind that lock, so coalesce
        # them: eight callers waiting on eight forward passes become one pass
        # over a batch of eight, which costs barely more than a pass over one.
        self._batcher = MicroBatcher(self._encode_locked)

    def _load(self, backend: str, model_name: str):
        options = BACKENDS[backend]
        if backend != "torch":
            try:
                return SentenceTransformer(model_name, **options)
            except Exception as exc:
                warnings.warn(
                    f"Embedding backend '{backend}' is unavailable ({type(exc).__name__}: "
                    f"{str(exc)[:120]}). Falling back to 'torch'. Install the optional "
                    "runtime with: pip install 'tensor-serve[onnx]'",
                    RuntimeWarning,
                    stacklevel=3,
                )
                self.backend = "torch"

        try:
            return SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            try:
                return SentenceTransformer(model_name)
            except Exception as download_error:
                raise RuntimeError(
                    f"Could not load embedding model '{model_name}' from the local cache "
                    "or download it from Hugging Face."
                ) from download_error

    @property
    def max_seq_length(self) -> int:
        """Word-pieces the model accepts, including its special tokens."""
        return int(getattr(self.model, "max_seq_length", 256) or 256)

    @property
    def max_tokens(self) -> int:
        """Word-pieces available to chunk text, once special tokens are reserved."""
        return max(1, self.max_seq_length - _SPECIAL_TOKEN_BUDGET)

    @property
    def tokenizer(self):
        """The model's tokenizer, so chunking can size chunks the model can hold."""
        return self.model.tokenizer

    @classmethod
    def for_ingest(cls, model_name="all-MiniLM-L6-v2"):
        """An embedder configured for bulk work. See _resolve_auto."""
        return cls(model_name, workload="bulk")

    def encode(self, texts, batch_size: int = 64):
        """
        Embed texts.

        sentence-transformers sorts by length within a call to limit padding
        waste, so larger accumulations before calling encode are more efficient.
        Use this for bulk work such as ingestion; single queries should go
        through :meth:`encode_query`, which shares a call with concurrent ones.
        """
        return self._encode_locked(texts, batch_size=batch_size)

    def encode_query(self, text: str):
        """
        Embed one query, sharing a model call with any concurrent queries.

        Falls back to a plain encode if batching is unavailable.
        """
        return self._batcher.encode_one(text)

    def _encode_locked(self, texts, batch_size: int = 64):
        with self._lock:
            self._warn_if_truncated(texts)
            return self.model.encode(texts, batch_size=batch_size, show_progress_bar=False)

    def _warn_if_truncated(self, texts):
        """
        Warn once if any text exceeds the model's input limit.

        The encoder truncates silently, so an oversized chunk is indexed by its
        opening fragment alone and the remainder is unreachable by semantic
        search. Chunk with ``tokenizer=`` and ``max_tokens=`` to avoid this.
        """
        if self._warned_truncation or not texts:
            return

        limit = self.max_seq_length
        for text in texts:
            try:
                length = len(self.tokenizer.encode(text, add_special_tokens=True))
            except Exception:
                return
            if length > limit:
                self._warned_truncation = True
                warnings.warn(
                    f"Text of {length} tokens exceeds the {limit}-token limit of "
                    f"'{self.model_name}' and will be truncated: "
                    f"{100 * (length - limit) / length:.0f}% of it will not reach its "
                    "vector. Re-ingest with token-aware chunking.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                return
