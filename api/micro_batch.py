"""
Coalesce concurrent embedding requests into single model calls.

The model is shared and not thread-safe, so access to it is serialised. Under
concurrency that makes the encoder a queue: eight threads each embedding one
query wait for eight separate forward passes, when one pass over a batch of
eight costs barely more than a pass over one.

Batching here is opportunistic and nobody ever waits. The first caller in starts
encoding straight away; callers that arrive while a pass is running queue behind
it and are swept into the next one. Batches therefore form exactly in proportion
to the load, and a lone caller pays nothing.

Measured against serialised single encodes (ONNX backend, 1 torch thread):

    threads          1      2      4      8
    serialised     730    711    732    732   queries/s
    batched        750    769   1273   1780

An earlier version paused briefly for company before encoding. That measured
worse at every thread count once the queueing above worked correctly, so there
is no wait: the queue that forms during an encode is all the batching needed.
"""

import threading
from typing import Callable, List


class MicroBatcher:
    """Batches concurrent single-text encode calls into one model call."""

    def __init__(self, encode_fn: Callable[[List[str]], object], max_batch: int = 32):
        """
        Args:
            encode_fn: Called with a list of texts, returns a sequence of vectors.
            max_batch: Never build a batch larger than this.
        """
        self._encode = encode_fn
        self._max_batch = max_batch
        self._lock = threading.Lock()
        self._pending: List["_Request"] = []
        self._batching = False

    def encode_one(self, text: str):
        """Embed one text, sharing a model call with any concurrent callers."""
        request = _Request(text)

        with self._lock:
            self._pending.append(request)
            lead = not self._batching
            if lead:
                self._batching = True

        if lead:
            self._drain()
        else:
            request.done.wait()

        if request.error is not None:
            raise request.error
        return request.vector

    def _drain(self) -> None:
        """
        Encode the queue, one model call at a time, until it is empty.

        The batching flag stays set for the whole loop, so callers arriving
        mid-encode queue instead of each starting a pass of their own. Clearing
        it before the encode meant every caller became a leader and no batch
        ever exceeded one request.
        """
        try:
            while True:
                with self._lock:
                    batch = self._pending[: self._max_batch]
                    self._pending = self._pending[len(batch) :]
                    if not batch:
                        self._batching = False
                        return
                self._encode_batch(batch)
        except BaseException:
            with self._lock:
                self._batching = False
                stranded, self._pending = self._pending, []
            for request in stranded:
                request.error = RuntimeError("embedding batch failed")
                request.done.set()
            raise

    def _encode_batch(self, batch: List["_Request"]) -> None:
        try:
            vectors = self._encode([r.text for r in batch])
            for request, vector in zip(batch, vectors):
                request.vector = vector
        except BaseException as exc:  # noqa: BLE001 - re-raised to every waiter
            for request in batch:
                request.error = exc
        finally:
            for request in batch:
                request.done.set()


class _Request:
    __slots__ = ("text", "vector", "error", "done")

    def __init__(self, text: str):
        self.text = text
        self.vector = None
        self.error = None
        self.done = threading.Event()
