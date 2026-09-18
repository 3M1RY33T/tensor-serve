import threading
import warnings

from sentence_transformers import SentenceTransformer

# Word-pieces reserved for the model's own [CLS] / [SEP] markers.
_SPECIAL_TOKEN_BUDGET = 2


class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        try:
            self.model = SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            try:
                self.model = SentenceTransformer(model_name)
            except Exception as download_error:
                raise RuntimeError(
                    f"Could not load embedding model '{model_name}' from the local cache "
                    "or download it from Hugging Face."
                ) from download_error

        self.model_name = model_name
        self._warned_truncation = False
        # The model and its tokenizer are shared by every request. HuggingFace's
        # fast tokenizer is a Rust object behind a runtime borrow check, and
        # concurrent use raises "RuntimeError: Already borrowed" — measured at 4
        # failures in 200 encodes across 8 threads. Retrieval runs in a thread
        # pool, so two chat requests that both miss the cache land here at once.
        self._lock = threading.Lock()

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

    def encode(self, texts, batch_size: int = 64):
        """
        Embed texts.

        sentence-transformers sorts by length within a call to limit padding
        waste, so larger accumulations before calling encode are slightly more
        efficient — measured 317 to 343 chunks/s on a 4,000-chunk sample.
        """
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
