"""Vectorized late-interaction Patch-MaxSim scoring with a 1-bit Hamming funnel and quadrant localization.

    MaxSim(Q, W) = sum_i max_p  s_p * (q_i . w_p)

where s_p is an optional spatial prior (e.g. "top-right"). Retrieval is a
two-stage funnel:

1. Coarse: every stored patch keeps a 64-bit sign code of its 64-d Matryoshka
   prefix. Approximate MaxSim uses cos(pi * hamming / 64) per (token, patch).
2. Rescore the top-C candidates asymmetrically: float32 query tokens x int8
   patch vectors (128-d Matryoshka prefix, one symmetric scale per window),
   S = scale_n * (q . p_int8), and return the best patch (row, col).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

from screengraph.embeddings.patch_encoder import binary_quantize, hamming_to_cosine, l2n, popcount64, quantize_int8

_SPATIAL = [
    (re.compile(r"\b(?:top|upper)[\s-]*right\b"), ("top", "right")),
    (re.compile(r"\b(?:top|upper)[\s-]*left\b"), ("top", "left")),
    (re.compile(r"\b(?:bottom|lower)[\s-]*right\b"), ("bottom", "right")),
    (re.compile(r"\b(?:bottom|lower)[\s-]*left\b"), ("bottom", "left")),
    (re.compile(r"\b(?:top|upper)\b(?:\s+(?:half|area|side|part))?"), ("top", None)),
    (re.compile(r"\b(?:bottom|lower)\b(?:\s+(?:half|area|side|part))?"), ("bottom", None)),
    (re.compile(r"\bleft(?:\s+(?:half|side|part))?\b"), (None, "left")),
    (re.compile(r"\bright(?:\s+(?:half|side|part))?\b"), (None, "right")),
    (re.compile(r"\b(?:center|centre|middle)\b"), ("center", "center")),
]
_FILLER = re.compile(r"\b(?:in|at|on)\s+the\s*$|\b(?:quadrant|corner|region|panel|area)\b", re.I)


def parse_spatial(query: str, rows: int, cols: int, boost: float = 1.25) -> tuple[str, Optional[np.ndarray], Optional[str]]:
    """Strip a spatial phrase from the query and turn it into per-patch prior weights."""
    q = query.lower()
    for rx, (v, h) in _SPATIAL:
        m = rx.search(q)
        if not m:
            continue
        rest = (q[: m.start()] + " " + q[m.end():]).strip()
        rest = _FILLER.sub(" ", rest)
        rest = re.sub(r"\b(?:in|at|on)\s+(?:the\s+)?(?=\s|$)", " ", rest)
        w = np.ones((rows, cols), np.float32)
        rr = np.arange(rows)[:, None]
        cc = np.arange(cols)[None, :]
        sel = np.ones((rows, cols), bool)
        if v == "top":
            sel &= rr < rows / 2
        elif v == "bottom":
            sel &= rr >= rows / 2
        elif v == "center":
            sel &= (rr >= rows / 4) & (rr < 3 * rows / 4)
        if h == "left":
            sel &= cc < cols / 2
        elif h == "right":
            sel &= cc >= cols / 2
        elif h == "center":
            sel &= (cc >= cols / 4) & (cc < 3 * cols / 4)
        w[sel] = boost
        label = "-".join(x for x in (v, h) if x and x != "center") or "center"
        return " ".join(rest.split()), w.ravel(), label
    return query, None, None


def quadrant_label(row: int, col: int, rows: int, cols: int) -> str:
    v = "top" if row < rows / 2 else "bottom"
    h = "left" if col < cols / 2 else "right"
    return f"{v}-{h}"


def token_idf(token_max: np.ndarray, tau: float = 0.25) -> np.ndarray:
    """Query-time IDF over the candidate pool: idf_t = 1 + ln((1 + n) / (1 + df_t)),
    df_t = #windows whose best patch matches token t above tau. Ubiquitous concepts
    ("red" in every log with an error line) weigh less than rare ones ("badge")."""
    n = token_max.shape[0]
    df = (token_max > tau).sum(axis=0)
    w = 1.0 + np.log((1.0 + n) / (1.0 + df))
    return (w / w.mean()).astype(np.float32)


def maxsim(Q: np.ndarray, patches: np.ndarray, spatial: Optional[np.ndarray] = None,
           idf: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Late-interaction scores  sum_i idf_i * max_p s_p (q_i . w_p).

    Q: (q, d) query token vectors; patches: (N, P, d). Returns (scores (N,), best_patch (N,)).
    """
    S = np.einsum("qd,npd->nqp", Q, patches, optimize=True)
    if spatial is not None:
        S = S * spatial[None, None, :]
    tmax = S.max(axis=2)  # (N, q)
    w = token_idf(tmax) if idf and len(tmax) > 1 else np.ones(Q.shape[0], np.float32)
    scores = tmax @ w
    best = np.einsum("nqp,q->np", S.clip(min=0), w).argmax(axis=1)
    return scores, best


_COS_LUT: dict[int, np.ndarray] = {}


def hamming_maxsim(Qbits: np.ndarray, Pbits: np.ndarray, bits: int, spatial: Optional[np.ndarray] = None) -> np.ndarray:
    """Approximate MaxSim from 1-bit codes. Qbits: (q, W) uint64; Pbits: (N, P, W) uint64.

    cos(pi*h/bits) is monotone decreasing in h, so without a spatial prior the max over
    patches is taken on integer distances first and mapped through a (bits+1)-entry LUT.
    """
    lut = _COS_LUT.get(bits)
    if lut is None:
        lut = _COS_LUT[bits] = np.cos(np.pi * np.arange(bits + 1) / bits).astype(np.float32)
    if Pbits.shape[-1] == 1:
        x = np.bitwise_xor(Pbits[:, None, :, 0], Qbits[None, :, None, 0])  # (N, q, P)
        ham = popcount64(x)
    else:
        ham = popcount64(np.bitwise_xor(Pbits[:, None, :, :], Qbits[None, :, None, :])).sum(-1)
    if spatial is None:
        return lut[ham.min(axis=2)].sum(axis=1)
    S = lut[ham] * spatial[None, None, :]
    return S.max(axis=2).sum(axis=1)


def global_average_scores(Q: np.ndarray, patches: np.ndarray, idf: bool = True) -> np.ndarray:
    """Baseline: collapse each window to the mean of its patch vectors (single-vector retrieval).
    Uses the same query-token IDF weighting as Patch-MaxSim so the comparison isolates pooling."""
    g = l2n(patches.mean(axis=1))
    T = g @ Q.T  # (N, q)
    w = token_idf(T) if idf and len(T) > 1 else np.ones(Q.shape[0], np.float32)
    return T @ w


@dataclass
class PatchHit:
    event_id: int
    score: float
    row: int
    col: int
    quadrant: str


class PatchIndex:
    """In-memory multi-vector index (grows by doubling): int8 patches + per-window scale + 64-bit codes."""

    def __init__(self, rows: int, cols: int, store_dim: int, prefilter_dim: int = 64):
        self.rows, self.cols = rows, cols
        self.P = rows * cols
        self.store_dim = store_dim
        self.prefilter_dim = prefilter_dim
        self.W = prefilter_dim // 64
        self._ids = np.zeros(0, np.int64)
        self._q = np.zeros((0, self.P, store_dim), np.int8)
        self._s = np.zeros(0, np.float32)
        self._bits = np.zeros((0, self.P, self.W), np.uint64)
        self.n = 0

    def _grow(self, need: int) -> None:
        cap = len(self._ids)
        if need <= cap:
            return
        new = max(need, cap * 2, 64)
        ids, q = np.zeros(new, np.int64), np.zeros((new, self.P, self.store_dim), np.int8)
        s, bits = np.zeros(new, np.float32), np.zeros((new, self.P, self.W), np.uint64)
        n = self.n
        ids[:n], q[:n], s[:n], bits[:n] = self._ids[:n], self._q[:n], self._s[:n], self._bits[:n]
        self._ids, self._q, self._s, self._bits = ids, q, s, bits

    def add(self, event_id: int, patch_i8: np.ndarray, scale: Optional[float] = None,
            prefilter_bits: Optional[np.ndarray] = None) -> None:
        """Add int8 patches (+ scale). A float array is accepted too and quantized here."""
        arr = np.asarray(patch_i8).reshape(self.P, -1)[:, : self.store_dim]
        if arr.dtype != np.int8:
            arr, scale = quantize_int8(l2n(arr.astype(np.float32)))
        self._grow(self.n + 1)
        self._ids[self.n] = event_id
        self._q[self.n] = arr
        self._s[self.n] = float(scale)
        if prefilter_bits is None:
            prefilter_bits = binary_quantize(l2n(arr[:, : self.prefilter_dim].astype(np.float32)))
        self._bits[self.n] = prefilter_bits.reshape(self.P, self.W)
        self.n += 1

    def patches(self, idx: np.ndarray) -> np.ndarray:
        """Dequantized float32 patches for positions idx."""
        return self._q[idx].astype(np.float32) * self._s[idx].reshape(-1, *([1] * (self._q.ndim - 1)))

    def position(self, event_ids: np.ndarray) -> np.ndarray:
        lookup = {int(e): i for i, e in enumerate(self._ids[: self.n])}
        return np.array([lookup.get(int(e), -1) for e in event_ids], np.int64)

    def search(self, Q: np.ndarray, k: int = 10, spatial: Optional[np.ndarray] = None,
               candidates: int = 256, restrict: Optional[np.ndarray] = None) -> list[PatchHit]:
        if self.n == 0 or len(Q) == 0:
            return []
        Qs = l2n(Q[:, : self.store_dim])
        if restrict is not None:
            pool = restrict[restrict >= 0]
        elif self.n > candidates:
            Qb = binary_quantize(l2n(Q[:, : self.prefilter_dim]))
            approx = hamming_maxsim(Qb, self._bits[: self.n], self.prefilter_dim, spatial)
            pool = np.argpartition(-approx, candidates - 1)[:candidates]
        else:
            pool = np.arange(self.n)
        # asymmetric rescoring: float32 query x int8 patches, scale applied per window
        scores, best = maxsim(Qs, self.patches(pool), spatial)
        order = np.argsort(-scores)[:k]
        hits = []
        for o in order:
            r, c = divmod(int(best[o]), self.cols)
            hits.append(PatchHit(int(self._ids[pool[o]]), float(scores[o]), r, c,
                                 quadrant_label(r, c, self.rows, self.cols)))
        return hits

    def global_average_search(self, Q: np.ndarray, k: int = 10) -> list[tuple[int, float]]:
        if self.n == 0:
            return []
        s = global_average_scores(l2n(Q[:, : self.store_dim]), self.patches(np.arange(self.n)))
        order = np.argsort(-s)[:k]
        return [(int(self._ids[o]), float(s[o])) for o in order]
