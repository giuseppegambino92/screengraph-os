"""64-bit perceptual hashes (2D-DCT pHash + dHash) and the two-stage idle gate.

pHash: area-average the frame to 32x32 luma, take the orthonormal 2D DCT-II
computed as two matrix products ``D = C @ G @ C.T``, keep the 8x8 lowest
frequencies and set bit k when coefficient k exceeds the median of the 63
AC coefficients.

dHash: area-average to 8x9 luma and set a bit wherever a pixel is brighter than
its right neighbour (horizontal gradient sign).
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Optional

import numpy as np

from screengraph.capture.models import AXNode

_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


@lru_cache(maxsize=8)
def dct_matrix(n: int) -> np.ndarray:
    """Orthonormal DCT-II basis matrix C (n x n): C[k, i] = a_k cos(pi (2i+1) k / 2n)."""
    k = np.arange(n, dtype=np.float64)[:, None]
    i = np.arange(n, dtype=np.float64)[None, :]
    c = np.sqrt(2.0 / n) * np.cos(np.pi * (2 * i + 1) * k / (2 * n))
    c[0, :] /= np.sqrt(2.0)
    return c.astype(np.float32)


@lru_cache(maxsize=128)
def _avg_matrix(length: int, bins: int) -> np.ndarray:
    """(bins x length) resampling matrix: exact area average over contiguous ranges when shrinking,
    nearest-neighbour when the window is smaller than the hash grid (tiny real windows)."""
    m = np.zeros((bins, length), np.float32)
    if bins <= length:
        edges = (np.arange(bins + 1) * length) // bins
        for b in range(bins):
            m[b, edges[b]:edges[b + 1]] = 1.0 / (edges[b + 1] - edges[b])
    else:
        m[np.arange(bins), (np.arange(bins) * length) // bins] = 1.0
    return m


@lru_cache(maxsize=128)
def _avg_luma_matrix(width: int, bins: int) -> np.ndarray:
    """(width*3 x bins): column averaging with the RGB->luma weights folded in (A_c^T kron w)."""
    return (_avg_matrix(width, bins).T[:, None, :] * _LUMA[None, :, None]).reshape(width * 3, bins).astype(np.float32)


def area_downsample(pixels: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Exact area-average downsample to luma via two BLAS products.

    RGB input:  G = A_r @ X @ (A_c^T kron w_luma), X = pixels as an (H, 3W) float matrix.
    The float copy X is the only full-resolution intermediate; it is explicitly zeroed
    before release (NumPy would otherwise free an implicit cast buffer unwiped).
    """
    h, w = pixels.shape[:2]
    if pixels.ndim == 3:
        x = pixels.reshape(h, w * 3).astype(np.float32)
        out = (_avg_matrix(h, out_h) @ x) @ _avg_luma_matrix(w, out_w)
        x.fill(0)
        return out
    lum = to_luma(pixels)
    return _avg_matrix(h, out_h) @ lum @ _avg_matrix(w, out_w).T


def to_luma(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return img @ _LUMA
    return img if img.dtype == np.float32 else img.astype(np.float32)


def _median63(ac: np.ndarray) -> float:
    """Median of the 63 AC coefficients via O(n) selection (np.median has ~10 us overhead)."""
    return float(np.partition(ac, 31)[31])


def _pack_bits(bits: np.ndarray) -> int:
    return int.from_bytes(np.packbits(bits.astype(np.uint8)).tobytes(), "big")


def phash64(pixels: np.ndarray, size: int = 32) -> int:
    g = area_downsample(pixels, size, size)
    if g.shape != (size, size):  # tiny inputs: pad to the DCT size
        pad = np.zeros((size, size), np.float32)
        pad[: g.shape[0], : g.shape[1]] = g
        g = pad
    c = dct_matrix(size)
    d = c @ g @ c.T
    g.fill(0)
    low = d[:8, :8].ravel()
    return _pack_bits(low > _median63(low[1:]))


def dhash64(pixels: np.ndarray) -> int:
    g = area_downsample(pixels, 8, 9)
    if g.shape != (8, 9):
        pad = np.zeros((8, 9), np.float32)
        pad[: g.shape[0], : g.shape[1]] = g
        g = pad
    bits = g[:, 1:] > g[:, :-1]
    g.fill(0)
    return _pack_bits(bits.ravel())


def hamming64(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def ax_tree_hash(
    title: str, url: str, doc_path: str, nodes: Iterable[AXNode], clipboard: Optional[str] = None
) -> str:
    """Stage-A structural hash of the accessibility tree (kept in RAM only, never persisted)."""
    h = hashlib.blake2b(digest_size=16)
    for part in (title, url, doc_path, clipboard or ""):
        h.update(part.encode("utf-8", "ignore"))
        h.update(b"\x1f")
    for n in nodes:
        h.update(n.role.encode())
        h.update(b"\x1e")
        h.update(n.text.encode("utf-8", "ignore"))
        h.update(b"\x01" if n.focused else b"\x00")
    return h.hexdigest()


def _phash_from_luma(g: np.ndarray, c: np.ndarray) -> int:
    d = c @ g @ c.T
    low = d[:8, :8].ravel()
    return _pack_bits(low > _median63(low[1:]))


CONFIDENCE_EPS = 0.25


def _robust_bits(low: np.ndarray, eps: float = CONFIDENCE_EPS) -> tuple[int, int]:
    """pHash bits plus a *confidence mask*: bits whose coefficient sits within eps * mean|c - median|
    of the median are unstable (flat regions flip them on sub-pixel noise) and are masked out."""
    med = np.partition(low[1:], 31)[31]
    dev = np.abs(low - med)
    mask = dev >= eps * max(float(dev[1:].mean()), 1e-3)
    mask[0] = False  # DC term carries brightness, not structure
    return _pack_bits(low > med), _pack_bits(mask)


@lru_cache(maxsize=4)
def _signature_bases(size: int) -> tuple[np.ndarray, np.ndarray]:
    """Stacked left/right bases so ONE pair of matmuls Z = L @ G2 @ R^T yields every hash input.

    rows  0..7   C8 @ M          global 8x8 low DCT of the 2x-averaged frame (M: 2size -> size averaging)
    rows  8..23  blockdiag(C8,C8) the four quadrant 8x8 low DCTs (Z[8:24, 8:24] holds the 2x2 block grid)
    rows 24..31  A8 / A9          dHash 8x9 area means (right basis has 9 rows there)
    """
    n2 = 2 * size
    c8 = dct_matrix(size)[:8].astype(np.float64)
    m = _avg_matrix(n2, size).astype(np.float64)
    bd = np.zeros((16, n2))
    bd[:8, :size], bd[8:, size:] = c8, c8
    left = np.vstack([c8 @ m, bd, _avg_matrix(n2, 8)]).astype(np.float32)
    right = np.vstack([c8 @ m, bd, _avg_matrix(n2, 9)]).astype(np.float32)
    return left, right


def perceptual_signature(pixels: np.ndarray, size: int = 32) -> tuple[tuple[int, int], int, tuple[tuple[int, int], ...]]:
    """((global pHash, mask), dHash, 2x2 quadrant (pHash, mask)) from one shared 2*size area-downsample.

    Quadrant hashes make Stage B sensitive to *localized* visual changes (a narrow
    spike in one chart panel) that barely move the global low-frequency spectrum;
    confidence masks keep them stable on near-flat regions. All five 8x8 low-frequency
    DCT blocks and the dHash means come from a single product Z = L @ G2 @ R^T.
    """
    g2 = area_downsample(pixels, 2 * size, 2 * size)
    if g2.shape != (2 * size, 2 * size):
        pad = np.zeros((2 * size, 2 * size), np.float32)
        pad[: g2.shape[0], : g2.shape[1]] = g2
        g2 = pad
    left, right = _signature_bases(size)
    z = left @ g2 @ right.T  # (32, 33)
    low = np.empty((5, 64), np.float32)
    low[0] = z[:8, :8].ravel()
    q = z[8:24, 8:24].reshape(2, 8, 2, 8).transpose(0, 2, 1, 3).reshape(4, 64)
    low[1:] = q
    med = np.partition(low[:, 1:], 31, axis=1)[:, 31:32]
    dev = np.abs(low - med)
    mask = dev >= CONFIDENCE_EPS * np.maximum(dev[:, 1:].mean(axis=1, keepdims=True), 1e-3)
    mask[:, 0] = False
    bits = np.packbits(np.concatenate([low > med, mask], axis=1), axis=1)  # (5, 16) bytes
    h = [int.from_bytes(bits[i, :8].tobytes(), "big") for i in range(5)]
    mk = [int.from_bytes(bits[i, 8:].tobytes(), "big") for i in range(5)]
    d8 = z[24:32, 24:33]
    dh = _pack_bits((d8[:, 1:] > d8[:, :-1]).ravel())
    g2.fill(0)
    z.fill(0)
    return (h[0], mk[0]), dh, ((h[1], mk[1]), (h[2], mk[2]), (h[3], mk[3]), (h[4], mk[4]))


def masked_hamming(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Disagreeing bits that are confident in *both* hashes."""
    return ((a[0] ^ b[0]) & a[1] & b[1]).bit_count()


@dataclass
class GateDecision:
    passed: bool
    reason: str
    phash: int = 0
    dhash: int = 0
    phash_distance: int = -1
    dhash_distance: int = -1
    quadrant_distance: int = -1
    elapsed_ms: float = 0.0


class PerceptualGate:
    """Two-stage idle gate.

    Stage A: AX-tree structural hash changed -> pass.
    Stage B: AX unchanged -> compare the 64-bit global pHash, dHash and the four
    quadrant pHashes with the last *accepted* frame of this window (pHash
    distances count only confidence-masked bits); drop when every distance is
    within threshold. Comparing against the last accepted
    frame (not the last seen one) lets slow drift accumulate until it crosses
    the threshold.
    """

    def __init__(self, phash_threshold: int = 4, dhash_threshold: int = 6, size: int = 32,
                 quadrant_threshold: int | None = None):
        self.phash_threshold = phash_threshold
        self.dhash_threshold = dhash_threshold
        self.quadrant_threshold = quadrant_threshold if quadrant_threshold is not None else phash_threshold
        self.size = size
        self._state: dict[str, tuple] = {}
        self.dropped = 0
        self.passed = 0

    def reset(self) -> None:
        self._state.clear()

    def check(self, window_id: str, ax_hash: str, pixels: Optional[np.ndarray]) -> GateDecision:
        t0 = time.perf_counter()
        prev = self._state.get(window_id)
        if pixels is not None:
            ph, dh, q = perceptual_signature(pixels, self.size)
        else:
            ph, dh, q = (0, 0), 0, ((0, 0),) * 4
        if prev is None:
            dec = GateDecision(True, "first_frame", ph[0], dh)
        elif prev[0] != ax_hash:
            dec = GateDecision(True, "ax_tree_changed", ph[0], dh)
        elif pixels is None:
            dec = GateDecision(False, "idle_ax_unchanged_no_pixels", ph[0], dh)
        else:
            dp, dd = masked_hamming(prev[1], ph), hamming64(prev[2], dh)
            dq = max(masked_hamming(a, b) for a, b in zip(prev[3], q))
            if dp <= self.phash_threshold and dd <= self.dhash_threshold and dq <= self.quadrant_threshold:
                dec = GateDecision(False, "idle_perceptual_match", ph[0], dh, dp, dd, dq)
            else:
                dec = GateDecision(True, "visual_change", ph[0], dh, dp, dd, dq)
        if dec.passed:
            self._state[window_id] = (ax_hash, ph, dh, q)
            self.passed += 1
        else:
            self.dropped += 1
        dec.elapsed_ms = (time.perf_counter() - t0) * 1e3
        return dec
