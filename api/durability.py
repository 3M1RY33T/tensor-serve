"""
Durable writes and build locks for index files.

Two failure modes this closes:

**Torn reads.** ``open(path, "wb")`` truncates the destination before it fills,
so a reader that opens the file during a rebuild sees a partial index. A server
answering queries while an ingest runs sits squarely in that window. Writing to
a temporary file in the same directory and then ``os.replace`` makes the swap
atomic: a reader sees either the old file or the new one, never half of either.

**Interleaved builds.** Two ingests writing the same collection produce a vector
index and a keyword index that disagree about what chunk 7 is. An advisory lock
makes the second one refuse rather than corrupt.

Note for Windows: ``os.replace`` fails there if another handle holds the
destination open, so a reader mid-query can make a write fail. That surfaces as
an error rather than as corruption, which is the right way round.
"""

import contextlib
import json
import os
import tempfile
import threading
import time
from typing import Optional

LOCK_SUFFIX = ".lock"


@contextlib.contextmanager
def atomic_write(path: str, mode: str = "wb"):
    """
    Write to `path` atomically.

    Yields a file handle to a temporary file in the same directory; on clean
    exit it is flushed, fsynced and renamed over the destination. On failure the
    temporary file is removed and the destination is left untouched.
    """
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)

    fd, temp_path = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".part")
    os.close(fd)

    try:
        with open(temp_path, mode) as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temp_path)
        raise


def atomic_replace(temp_path: str, path: str) -> None:
    """Move a file already written elsewhere into place atomically."""
    os.replace(temp_path, path)


def _process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    except OSError:
        return True
    return True


# Locks held by this process, so two builds in one process — or two threads —
# conflict as surely as two processes do. A same-PID check alone would read a
# second concurrent build as re-entrancy and wave it through.
_held_locally: set = set()
_held_lock = threading.Lock()


class BuildLockError(RuntimeError):
    """Raised when another live process is already building this collection."""


class BuildLock:
    """
    Advisory lock around building one collection.

    A lock left behind by a process that died is detected and broken, so a crash
    does not require manual cleanup.
    """

    def __init__(self, path: str, purpose: str = "build"):
        self.lock_path = f"{path}{LOCK_SUFFIX}"
        self.purpose = purpose
        self._held = False

    def _read(self) -> Optional[dict]:
        try:
            with open(self.lock_path, "r") as handle:
                return json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None

    def acquire(self) -> "BuildLock":
        key = os.path.abspath(self.lock_path)
        with _held_lock:
            if key in _held_locally:
                raise BuildLockError(
                    f"'{self.lock_path}' is already held by this process."
                )
            _held_locally.add(key)

        try:
            self._acquire_file()
        except BaseException:
            with _held_lock:
                _held_locally.discard(key)
            raise

        self._held = True
        return self

    def _acquire_file(self) -> None:
        existing = self._read()
        if existing is not None:
            pid = int(existing.get("pid", -1))
            if pid != os.getpid() and _process_alive(pid):
                raise BuildLockError(
                    f"'{self.lock_path}' is held by process {pid} "
                    f"({existing.get('purpose', 'build')}, started "
                    f"{existing.get('started', 'unknown')}). Wait for it to finish, "
                    "or remove the lock file if that process is gone."
                )
            # Stale: the holder is no longer running.

        with atomic_write(self.lock_path, "w") as handle:
            json.dump(
                {
                    "pid": os.getpid(),
                    "purpose": self.purpose,
                    "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                },
                handle,
            )

    def release(self) -> None:
        if not self._held:
            return
        current = self._read()
        if current and int(current.get("pid", -1)) == os.getpid():
            with contextlib.suppress(OSError):
                os.unlink(self.lock_path)
        with _held_lock:
            _held_locally.discard(os.path.abspath(self.lock_path))
        self._held = False

    def __enter__(self) -> "BuildLock":
        return self.acquire()

    def __exit__(self, *exc) -> None:
        self.release()
