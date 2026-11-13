"""EphemeralFrameBuffer: pixels live only inside a ``with`` block and are memset to 0 on exit.

Guarantees (and their limits):

* Pixels are never handed to any persistence API -- the storage layer has no
  code path that accepts an ndarray of pixels (see ``db/store.py``).
* On ``__exit__`` -- normal *or* exceptional -- the primary frame buffer and
  every scratch buffer allocated through :meth:`scratch` are overwritten with
  zeros via ``ndarray.fill(0)`` and verified with ``not buf.any()``.
* Every wipe is appended to a SHA-256 hash chain in the :class:`ZeroingLedger`
  so the privacy audit can prove how many buffers/bytes were zeroed.

What Python cannot guarantee: temporaries created by NumPy inside library
calls (e.g. ``astype`` results not requested via ``scratch``) are freed by the
allocator without being wiped. The encoder therefore routes its large
pixel-derived intermediates through ``scratch()``.
"""

from __future__ import annotations

import hashlib
import struct
import threading
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ZeroingLedger:
    buffers_zeroed: int = 0
    bytes_zeroed: int = 0
    verification_failures: int = 0
    frames_wrapped: int = 0
    chain_head: str = "0" * 64
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, nbytes: int, verified: bool) -> None:
        with self._lock:
            self.buffers_zeroed += 1
            self.bytes_zeroed += int(nbytes)
            if not verified:
                self.verification_failures += 1
            payload = bytes.fromhex(self.chain_head) + struct.pack("<QQ?", self.buffers_zeroed, int(nbytes), verified)
            self.chain_head = hashlib.sha256(payload).hexdigest()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "frames_wrapped": self.frames_wrapped,
                "buffers_zeroed": self.buffers_zeroed,
                "bytes_zeroed": self.bytes_zeroed,
                "verification_failures": self.verification_failures,
                "chain_head": self.chain_head,
            }

    def restore(self, snap: dict) -> None:
        with self._lock:
            self.frames_wrapped = int(snap.get("frames_wrapped", 0))
            self.buffers_zeroed = int(snap.get("buffers_zeroed", 0))
            self.bytes_zeroed = int(snap.get("bytes_zeroed", 0))
            self.verification_failures = int(snap.get("verification_failures", 0))
            self.chain_head = str(snap.get("chain_head", "0" * 64))


GLOBAL_LEDGER = ZeroingLedger()


class EphemeralFrameBuffer:
    """Context manager owning a writable pixel buffer for the duration of feature extraction."""

    __slots__ = ("_buf", "_scratch", "_ledger", "_closed")

    def __init__(self, pixels: np.ndarray, ledger: ZeroingLedger | None = None):
        if not isinstance(pixels, np.ndarray):
            raise TypeError("EphemeralFrameBuffer requires a numpy ndarray")
        if not pixels.flags.writeable:
            raise ValueError("EphemeralFrameBuffer requires a writable buffer (it must be able to wipe it)")
        self._buf = pixels
        self._scratch: list[np.ndarray] = []
        self._ledger = ledger or GLOBAL_LEDGER
        self._closed = False

    @property
    def pixels(self) -> np.ndarray:
        if self._closed:
            raise RuntimeError("frame buffer already wiped")
        return self._buf

    def scratch(self, shape, dtype=np.float32) -> np.ndarray:
        """Allocate a pixel-derived intermediate that is also wiped on exit."""
        arr = np.empty(shape, dtype=dtype)
        self._scratch.append(arr)
        return arr

    def track(self, arr: np.ndarray) -> np.ndarray:
        """Register an existing pixel-derived array for wiping on exit."""
        if isinstance(arr, np.ndarray) and arr.flags.writeable and arr.size:
            self._scratch.append(arr)
        return arr

    def __enter__(self) -> "EphemeralFrameBuffer":
        with self._ledger._lock:
            self._ledger.frames_wrapped += 1
        return self

    def wipe(self) -> None:
        if self._closed:
            return
        for arr in [self._buf, *self._scratch]:
            arr.fill(0)  # memset(0) over the underlying memory (views wipe their base region)
            self._ledger.record(arr.nbytes, verified=not arr.any())
        self._scratch.clear()
        self._closed = True

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.wipe()
        return False  # never swallow exceptions
