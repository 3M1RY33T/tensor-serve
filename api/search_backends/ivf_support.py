"""
Runtime probe for whether FAISS IVF training is usable in this process.

faiss and torch each ship their own OpenMP runtime. On some installs — macOS
arm64 with pip-installed faiss-cpu and torch is the common one — loading both
aborts the process the moment FAISS runs k-means:

    OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib
    already initialized.

It is not catchable: the process dies with SIGABRT, or with SIGSEGV if
KMP_DUPLICATE_LIB_OK is set. Since the server imports torch (through
sentence-transformers) before it ever builds an index, selecting the IVF
backend on such an install would kill the process mid-ingest.

Guessing from the platform would be wrong on correctly-linked installs, so this
probes the real thing once in a subprocess and remembers the answer.
"""

import subprocess
import sys

_PROBE = (
    "import torch, numpy, faiss;"
    "x = numpy.random.rand(64, 8).astype('float32');"
    "i = faiss.IndexIVFFlat(faiss.IndexFlatL2(8), 8, 2);"
    "i.train(x)"
)

_cached: bool | None = None


def ivf_training_is_safe(timeout: int = 120) -> bool:
    """
    True when FAISS IVF training survives alongside torch in this environment.

    Probed once per process, in a subprocess, because a failure is a crash
    rather than an exception.
    """
    global _cached
    if _cached is not None:
        return _cached

    try:
        completed = subprocess.run(
            [sys.executable, "-c", _PROBE],
            capture_output=True,
            timeout=timeout,
        )
        _cached = completed.returncode == 0
    except (subprocess.SubprocessError, OSError):
        # If the probe cannot run, assume unsafe: degrading to exact search
        # costs a few milliseconds per query, and being wrong costs the process.
        _cached = False

    return _cached


def reset_cache() -> None:
    """Forget the probe result. For tests."""
    global _cached
    _cached = None
