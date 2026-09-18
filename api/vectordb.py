import os
import warnings

from api.search_backends import get_semantic_backend

# Files each semantic backend writes, relative to the database name. Used to
# answer "is this collection already built?" without guessing at extensions:
# the startup auto-load previously probed "{name}.index" / "{name}.pkl", which
# no backend has ever written, so it never fired.
_BACKEND_FILES = {
    "faiss_flat": (".faiss_flat.index", ".chunks"),
    "faiss_ivf": (".faiss_ivf.index", ".faiss_ivf.json", ".chunks"),
}


def _configured(key, fallback):
    """Read a config value without making config a hard import dependency."""
    try:
        from api.config import get_config_value

        value = get_config_value(key)
        return fallback if value is None else value
    except Exception:
        return fallback


def _ivf_is_usable() -> bool:
    """Whether the IVF backend can train without killing the process."""
    from api.search_backends.ivf_support import ivf_training_is_safe

    return ivf_training_is_safe()


def index_exists(path: str, variant: str = None) -> bool:
    """True when a saved index for this database name is present on disk."""
    variant = variant or _configured("semantic_backend", "faiss_flat")
    suffixes = _BACKEND_FILES.get(variant)
    if not suffixes:
        return False
    return all(os.path.exists(f"{path}{suffix}") for suffix in suffixes)


def save_collection(path: str, db, bm25=None) -> None:
    """
    Write a complete collection in the order a reader can survive.

    The chunk store goes first and the vector index last, because
    ``index_exists`` gates on the vector index: a crash part-way leaves a store
    with no index beside it, which reads as "not built". The reverse order
    would leave an index promising chunks that were never written.
    """
    db.backend.store.save(path)
    if bm25 is not None:
        bm25.save(path)
    db.save_vectors_only(path)


def database_files(path: str, variant: str = None) -> dict:
    """Every file belonging to one database, whether or not it exists."""
    variant = variant or _configured("semantic_backend", "faiss_flat")
    files = {
        "vectors": f"{path}.{variant}.index",
        "chunks": f"{path}.chunks",
        "bm25": f"{path}.bm25",
    }
    if variant == "faiss_ivf":
        files["params"] = f"{path}.faiss_ivf.json"
    return files


def list_databases(directory: str = ".") -> list:
    """
    Discover built databases by the files their backends actually write.

    Discovery used to glob "*.index" and strip one suffix, which reported
    "python_docs.faiss_flat" as the database name and then looked for its
    keyword index under that name, finding none.
    """
    found = []
    for variant, suffixes in _BACKEND_FILES.items():
        marker = f".{variant}.index"
        for entry in sorted(os.listdir(directory or ".")):
            if not entry.endswith(marker):
                continue
            name = entry[: -len(marker)]
            path = os.path.join(directory, name) if directory not in ("", ".") else name
            files = database_files(path, variant)
            found.append(
                {
                    "name": name,
                    "variant": variant,
                    "files": {k: v for k, v in files.items() if os.path.exists(v)},
                    "missing": [v for v in files.values() if not os.path.exists(v)],
                    "complete": index_exists(path, variant),
                }
            )
    return found


class VectorDB:
    """Vector database with support for multiple FAISS backend variants."""

    def __init__(self, dim: int = 384, variant: str = None, nprobe: int = None):
        """
        Initialize vector database with specified backend variant.

        Args:
            dim: Embedding dimension
            variant: 'faiss_flat' (exact) or 'faiss_ivf' (approximate).
                     Defaults to the configured semantic_backend — previously
                     hardcoded to faiss_flat at every call site, which left
                     faiss_ivf unreachable however the profile was set.
            nprobe: IVF cells probed per query. Defaults to the configured value.
        """
        self.dim = dim
        self.variant = variant or _configured("semantic_backend", "faiss_flat")

        if self.variant == "faiss_ivf" and not _ivf_is_usable():
            warnings.warn(
                "faiss_ivf is unusable in this environment: FAISS and torch have "
                "conflicting OpenMP runtimes, so training the index would abort the "
                "process. Falling back to faiss_flat, which is exact and measured at "
                "4.3ms per query over one million vectors. Install a FAISS build "
                "linked against the same OpenMP runtime as torch to use IVF.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.variant = "faiss_flat"

        backend_cls = get_semantic_backend(self.variant)

        if self.variant == "faiss_ivf":
            self.backend = backend_cls(
                dim, nprobe=nprobe if nprobe is not None else _configured("faiss_nprobe", 8)
            )
        else:
            self.backend = backend_cls(dim)

    def add(self, embeddings, chunks, metadata=None):
        """Add embeddings and chunks to index."""
        self.backend.add(embeddings, chunks, metadata)

    def save(self, path="db"):
        """Save the vector index and its chunk store."""
        self.backend.save(path)

    def save_vectors_only(self, path="db"):
        """
        Save the vector index without rewriting the chunk store.

        Used by save_collection, which writes the store first so the vector
        index is the last file to appear.
        """
        self.backend.save_vectors(path)

    def load(self, path="db"):
        """Load index from disk."""
        self.backend.load(path)

    def search(self, query_embedding, top_k=5):
        """Search and return top-k text chunks."""
        return self.backend.search(query_embedding, top_k)

    def search_indices(self, query_embedding, top_k=5):
        """Return top-k chunk indices."""
        return self.backend.search_indices(query_embedding, top_k)

    @property
    def texts(self):
        """Get all indexed texts."""
        return self.backend.texts

    @property
    def metadata(self):
        """Get all metadata."""
        return self.backend.metadata
