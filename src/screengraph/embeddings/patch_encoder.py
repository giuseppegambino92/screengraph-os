"""Patch encoder: generic Spectral-HOG-JL visual descriptors + perceptual colour concepts + positioned AX text.

Default (zero heavy dependencies, works on any RGB/RGBA array):

* **visual descriptor** - per patch: orthonormal 2D-DCT of the opponent channels Y, R-G, B-Y
  (low/mid zig-zag coefficients), an 8-bin Scharr HOG with edge density, and basic colour-term
  fractions. Fused by a seeded Achlioptas signed random projection (Johnson-Lindenstrauss) into
  R^d, made orthogonal to the colour-word subspace. It captures layout/texture/spectrum and powers
  query-by-example, but it is *not* grounded in words.
* **perceptual colour concepts** - basic colour terms from a fixed HSV lookup (red, orange, ...);
  these, and the scrubbed accessibility-tree tokens placed in the patch where they are rendered,
  are what text queries match.
* **text tokens** - every concept/token owns a seeded Gaussian unit vector B[t]; random Gaussian
  bases are near-orthogonal (JL), so q . w_p recovers the weight of concept q in patch p.

Optional: ui_concepts (hand-written UI-object detectors, an ablation) and a learned
VisionAdapter (open_clip / SigLIP) for real word-grounded vision.
Any prefix of a Gaussian/Rademacher projection is itself a JL projection, so Matryoshka prefix
slicing (64/128/256) degrades gracefully.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional, Sequence

import numpy as np

# ----------------------------------------------------------------------------
# Tokenization & concept canonicalization
# ----------------------------------------------------------------------------
_WORD_RE = re.compile(r"<REDACTED_[A-Z]+:[0-9a-f]+>|[A-Za-z][A-Za-z0-9_]*|\d+")
_CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")

STOPWORDS = frozenset(
    "a an the of in on at to for from with by and or is are was were be been this that these those it its "
    "as into over under than then there here my your our their me we you i he she they them show find "
    "where what which who when how did do does done any some all about up out via per vs".split()
)

CANON = {
    # colours
    "crimson": "red", "scarlet": "red", "maroon": "red", "reddish": "red",
    "amber": "orange", "orangish": "orange", "gold": "yellow", "golden": "yellow",
    "teal": "cyan", "turquoise": "cyan", "aqua": "cyan", "navy": "blue", "azure": "blue", "bluish": "blue",
    "violet": "purple", "lavender": "purple", "pink": "magenta", "fuchsia": "magenta",
    "lime": "green", "greenish": "green", "grey": "gray",
    # structures
    "graph": "chart", "plot": "chart", "plotted": "chart", "graphs": "chart", "charts": "chart",
    "timeseries": "line", "sparkline": "line", "trend": "line",
    "peak": "spike", "spikes": "spike", "surge": "spike", "burst": "spike", "jump": "spike", "spiking": "spike",
    "bars": "bar", "histogram": "bar", "column": "bar", "columns": "bar",
    "dialog": "modal", "popup": "modal", "pop": "modal", "alert": "modal", "overlay": "modal",
    "modals": "modal", "dialogs": "modal",
    "pill": "badge", "chip": "badge", "tag": "badge", "label": "badge", "badges": "badge",
    "hierarchy": "tree", "trees": "tree", "flowchart": "diagram", "diagrams": "diagram", "architecture": "architecture",
    "dark": "dark", "light": "light",
    "failing": "failed", "failure": "failed", "fail": "failed", "fails": "failed",
    "errors": "error", "warnings": "warning", "mem": "memory", "ram": "memory", "oom": "oomkilled",
}


def split_identifier(tok: str) -> list[str]:
    parts = []
    for chunk in tok.split("_"):
        parts.extend(p for p in _CAMEL_RE.findall(chunk) if p)
    return parts


def canon(tok: str) -> str:
    t = tok.lower()
    return CANON.get(t, t)


def tokenize(text: str, expand_identifiers: bool = True) -> list[str]:
    """Lowercased, canonicalized tokens; identifiers also contribute their camel/snake sub-words."""
    out: list[str] = []
    for m in _WORD_RE.finditer(text or ""):
        tok = m.group(0)
        if tok.startswith("<REDACTED"):
            out.append(tok.lower().strip("<>").replace(":", "_"))
            continue
        low = canon(tok)
        if low in STOPWORDS or len(low) < 2:
            continue
        out.append(low)
        if expand_identifiers and ("_" in tok or (any(c.isupper() for c in tok[1:]) and any(c.islower() for c in tok))):
            for sub in split_identifier(tok):
                s = canon(sub)
                if s != low and len(s) >= 2 and s not in STOPWORDS:
                    out.append(s)
    return out


def _token_positions(text: str) -> list[tuple[str, float]]:
    """Tokens with their relative character position (0..1) inside the text."""
    n = max(len(text), 1)
    res = []
    for m in _WORD_RE.finditer(text):
        center = (m.start() + m.end()) / 2 / n
        for t in tokenize(m.group(0)):
            res.append((t, center))
    return res


# ----------------------------------------------------------------------------
# Concept basis (random Gaussian unit vectors, cached)
# ----------------------------------------------------------------------------
class ConceptBasis:
    def __init__(self, dim: int, seed: int, max_cache: int = 60000):
        self.dim = dim
        self.seed = seed
        self._cache: dict[str, np.ndarray] = {}
        self._max = max_cache

    def vec(self, concept: str) -> np.ndarray:
        v = self._cache.get(concept)
        if v is None:
            h = int.from_bytes(hashlib.blake2b(concept.encode(), digest_size=8).digest(), "little")
            rng = np.random.default_rng((h ^ self.seed) & 0xFFFFFFFFFFFFFFFF)
            v = rng.standard_normal(self.dim).astype(np.float32)
            v /= np.linalg.norm(v)
            if len(self._cache) >= self._max:
                self._cache.clear()
            self._cache[concept] = v
        return v

    def matrix(self, concepts: Sequence[str]) -> np.ndarray:
        if not concepts:
            return np.zeros((0, self.dim), np.float32)
        return np.stack([self.vec(c) for c in concepts])


# ----------------------------------------------------------------------------
# Vector utilities: Matryoshka, binary quantization, popcount
# ----------------------------------------------------------------------------
def l2n(x: np.ndarray, axis: int = -1) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, 1e-12)


def matryoshka(v: np.ndarray, dim: int) -> np.ndarray:
    """Prefix-slice to ``dim`` dimensions and renormalize."""
    return l2n(np.ascontiguousarray(v[..., :dim], dtype=np.float32))


def binary_quantize(v: np.ndarray) -> np.ndarray:
    """1-bit sign quantization packed into uint64 words: (..., d) float -> (..., d/64) uint64."""
    d = v.shape[-1]
    if d % 64:
        raise ValueError("dimension must be a multiple of 64 for uint64 packing")
    bits = np.packbits(v > 0, axis=-1, bitorder="little")
    return np.ascontiguousarray(bits).view(np.uint64).reshape(*v.shape[:-1], d // 64)


_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def popcount64(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    b = x.view(np.uint8).reshape(*x.shape, 8)
    return _POP8[b].sum(-1, dtype=np.uint16)


def hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamming distance over the last (uint64 word) axis with broadcasting."""
    return popcount64(np.bitwise_xor(a, b)).sum(-1, dtype=np.int32)


def hamming_to_cosine(ham: np.ndarray, bits: int) -> np.ndarray:
    """For sign-quantized Gaussian projections E[ham/bits] = theta/pi, so cos ~ cos(pi*ham/bits)."""
    return np.cos(np.pi * ham.astype(np.float32) / bits)


# ----------------------------------------------------------------------------
# Int8 symmetric scalar quantization (storage + asymmetric rescoring)
# ----------------------------------------------------------------------------
def quantize_int8(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Symmetric per-array int8 quantization: x ~ scale * q,  q in [-127, 127]."""
    m = float(np.max(np.abs(x))) if x.size else 0.0
    scale = m / 127.0 if m > 0 else 1.0
    q = np.clip(np.rint(x / scale), -127, 127).astype(np.int8)
    return q, scale


def dequantize_int8(q: np.ndarray, scale) -> np.ndarray:
    s = np.asarray(scale, np.float32)
    return q.astype(np.float32) * (s.reshape(s.shape + (1,) * (q.ndim - s.ndim)) if s.ndim else s)


# ----------------------------------------------------------------------------
# Generic perceptual colour naming (Berlin-Kay basic colour terms over HSV cells)
# ----------------------------------------------------------------------------
COLOR_NAMES = ("black", "gray", "white", "red", "orange", "yellow", "green", "cyan", "blue", "purple", "magenta")
CHROMATIC = np.array([False, False, False, True, True, True, True, True, True, True, True])
VISUAL_CONCEPTS = COLOR_NAMES + ("dark", "light")
_VC_INDEX = {c: i for i, c in enumerate(VISUAL_CONCEPTS)}
_NC = len(COLOR_NAMES)
_COLOR_SET = frozenset(COLOR_NAMES[3:])


@lru_cache(maxsize=1)
def _color_lut() -> np.ndarray:
    """32x32x32 LUT (5 bits/channel) -> basic colour term, via HSV rules on cell centres."""
    c = (np.arange(32, dtype=np.float32) * 8 + 4) / 255.0
    r, g, b = np.meshgrid(c, c, c, indexing="ij")
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    delta = mx - mn
    s = np.where(mx > 0, delta / np.maximum(mx, 1e-6), 0)
    v = mx
    dd = np.maximum(delta, 1e-6)
    hue = np.zeros_like(mx)
    hue = np.where(mx == r, ((g - b) / dd) % 6, hue)
    hue = np.where(mx == g, (b - r) / dd + 2, hue)
    hue = np.where(mx == b, (r - g) / dd + 4, hue)
    hue = hue * 60.0
    out = np.full(mx.shape, 1, np.uint8)  # gray
    out[v < 0.22] = 0
    chroma = (s >= 0.30) & (v >= 0.22) & (delta >= 0.12)
    out[(~chroma) & (v >= 0.80)] = 2
    hb = np.select([hue < 15, hue < 40, hue < 70, hue < 165, hue < 195, hue < 255, hue < 290, hue < 340],
                   [3, 4, 5, 6, 7, 8, 9, 10], default=3)
    out[chroma] = hb[chroma]
    return out.reshape(-1)


@lru_cache(maxsize=32)
def _patch_index_map(h: int, w: int, rows: int, cols: int) -> np.ndarray:
    ry = np.minimum((np.arange(h) * rows) // h, rows - 1)
    cx = np.minimum((np.arange(w) * cols) // w, cols - 1)
    return (ry[:, None] * cols + cx[None, :]).astype(np.int32).ravel()


@lru_cache(maxsize=64)
def _avg(length: int, bins: int) -> np.ndarray:
    """(bins x length) resampling matrix: exact area average when shrinking, nearest-neighbour when the
    source is smaller than the target grid (small real windows), so output shapes are always exact."""
    m = np.zeros((bins, length), np.float32)
    if bins <= length:
        e = (np.arange(bins + 1) * length) // bins
        for b in range(bins):
            m[b, e[b]:e[b + 1]] = 1.0 / (e[b + 1] - e[b])
    else:
        m[np.arange(bins), (np.arange(bins) * length) // bins] = 1.0
    return m


@lru_cache(maxsize=4)
def _dct(n: int) -> np.ndarray:
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    c = np.sqrt(2.0 / n) * np.cos(np.pi * (2 * i + 1) * k / (2 * n))
    c[0] /= np.sqrt(2.0)
    return c.astype(np.float32)


def _zigzag_mask(n: int, max_sum: int) -> np.ndarray:
    u, v = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return ((u + v) < max_sum).ravel()


# opponent colour transform (rows: Y, R-G, B-Y), applied to 0..1 RGB
_OPP = np.array([[0.299, 0.587, 0.114], [1.0, -1.0, 0.0], [-0.5, -0.5, 1.0]], np.float32)
SPEC_BLOCK = 16  # DCT block per patch (patch is area-averaged to 16x16 per channel)
HOG_BLOCK = 32  # gradient resolution per patch
HOG_BINS = 8
_ZZ_Y = _zigzag_mask(SPEC_BLOCK, 6)  # 21 low/mid luminance coefficients
_ZZ_C = _zigzag_mask(SPEC_BLOCK, 3)  # 6 low chroma coefficients per opponent channel
FEATURE_GROUPS = {"spectral_y": int(_ZZ_Y.sum()), "spectral_chroma": 2 * int(_ZZ_C.sum()),
                  "hog": HOG_BINS, "edges": 2, "colour_names": _NC}
FEATURE_DIM = sum(FEATURE_GROUPS.values())


@dataclass
class VisualAnalysis:
    features: np.ndarray  # (P, FEATURE_DIM) generic Spectral-HOG-colour descriptor
    concept_weights: np.ndarray  # (P, len(VISUAL_CONCEPTS)) perceptual colour/theme concepts
    color_frac: np.ndarray  # (P, NC)
    ui_weights: Optional[np.ndarray] = None  # optional UI-object layer
    objects: list[dict] = field(default_factory=list)


def analyze_visual(pixels: np.ndarray, rows: int, cols: int, buf=None, ui_concepts: bool = False) -> VisualAnalysis:
    """Generic multi-scale patch descriptors for ANY RGB(A) uint8 array (simulator grid or real capture).

    Per patch p of the rows x cols grid:
      * spectral: orthonormal 2D-DCT-II of the 16x16 area-averaged patch in the opponent space
        Y / (R-G) / (B-Y); zig-zag low+mid coefficients (21 Y, 6+6 chroma), signed-sqrt compressed
      * HOG: 8-bin unsigned orientation histogram of Scharr gradients at 32x32 per patch,
        magnitude-weighted, Hellinger-normalized; plus edge density and mean gradient magnitude
      * colour: fraction of pixels per basic colour term (perceptual LUT)
    """
    if pixels.ndim == 3 and pixels.shape[2] == 4:
        pixels = pixels[..., :3]
    h, w = pixels.shape[:2]
    P = rows * cols
    # ---- colour names (full resolution, 5-bit LUT) -----------------------------
    p16 = pixels.astype(np.uint16)
    idx = ((p16[..., 0] >> 3) << 10) | ((p16[..., 1] >> 3) << 5) | (p16[..., 2] >> 3)
    colors = _color_lut()[idx]
    if buf is not None:
        buf.track(p16)
        buf.track(idx)
        buf.track(colors)
    pmap = _patch_index_map(h, w, rows, cols)
    hist = np.bincount(pmap * _NC + colors.ravel(), minlength=P * _NC).reshape(P, _NC).astype(np.float32)
    frac = hist / np.maximum(np.bincount(pmap, minlength=P).astype(np.float32)[:, None], 1)

    # ---- spectral: area-average each channel to (rows*16, cols*16), opponent transform, DCT ----
    x = pixels.reshape(h, w * 3).astype(np.float32)
    if buf is not None:
        buf.track(x)
    H, Wd = rows * SPEC_BLOCK, cols * SPEC_BLOCK
    small = (_avg(h, H) @ x).reshape(H, w, 3)
    small = np.tensordot(small, _avg(w, Wd), axes=([1], [1])).transpose(0, 2, 1)  # (H, Wd, 3)
    opp = (small / 255.0) @ _OPP.T  # (H, Wd, 3)
    blocks = opp.reshape(rows, SPEC_BLOCK, cols, SPEC_BLOCK, 3).transpose(0, 2, 4, 1, 3).reshape(P, 3, SPEC_BLOCK, SPEC_BLOCK)
    c = _dct(SPEC_BLOCK)
    coef = (c @ blocks @ c.T).reshape(P, 3, SPEC_BLOCK * SPEC_BLOCK)
    spec_y = coef[:, 0, _ZZ_Y]
    spec_c = np.concatenate([coef[:, 1, _ZZ_C], coef[:, 2, _ZZ_C]], axis=1)
    spec_y = np.sign(spec_y) * np.sqrt(np.abs(spec_y))
    spec_c = np.sign(spec_c) * np.sqrt(np.abs(spec_c))

    # ---- Scharr HOG at 32x32 per patch ------------------------------------------
    Hg, Wg = rows * HOG_BLOCK, cols * HOG_BLOCK
    lum = (_avg(h, Hg) @ x) @ np.kron(_avg(w, Wg).T, np.array([[0.299], [0.587], [0.114]], np.float32)) / 255.0
    x.fill(0)
    gx = (3 * (lum[:-2, 2:] - lum[:-2, :-2]) + 10 * (lum[1:-1, 2:] - lum[1:-1, :-2]) + 3 * (lum[2:, 2:] - lum[2:, :-2])) / 32.0
    gy = (3 * (lum[2:, :-2] - lum[:-2, :-2]) + 10 * (lum[2:, 1:-1] - lum[:-2, 1:-1]) + 3 * (lum[2:, 2:] - lum[:-2, 2:])) / 32.0
    mag = np.hypot(gx, gy)
    ang = np.mod(np.arctan2(gy, gx), np.pi)
    b = np.minimum((ang * (HOG_BINS / np.pi)).astype(np.int32), HOG_BINS - 1)
    pm = _patch_index_map(Hg, Wg, rows, cols).reshape(Hg, Wg)[1:-1, 1:-1].ravel()
    hog = np.bincount(pm * HOG_BINS + b.ravel(), weights=mag.ravel(), minlength=P * HOG_BINS).reshape(P, HOG_BINS)
    hog = np.sqrt(hog / np.maximum(hog.sum(1, keepdims=True), 1e-9))  # Hellinger
    cnt = np.maximum(np.bincount(pm, minlength=P), 1)
    density = np.bincount(pm, weights=(mag > 0.04).ravel().astype(np.float32), minlength=P) / cnt
    mmag = np.bincount(pm, weights=mag.ravel(), minlength=P) / cnt
    edges = np.stack([np.sqrt(density), np.sqrt(np.minimum(mmag * 4, 1.0))], axis=1)
    # wipe every pixel-derived intermediate that could be inverted back to an image:
    # resampled channels, opponent image, its per-block copy and full DCT, gradient maps
    for arr in (lum, small, opp, blocks, coef, gx, gy, mag, ang):
        arr.fill(0)

    def gn(a):
        return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)

    feats = np.concatenate([gn(spec_y), gn(spec_c), gn(hog), edges * 0.7, gn(np.sqrt(frac))], axis=1).astype(np.float32)

    # ---- perceptual colour concepts: share of the salient (chromatic) content -----
    cw = np.zeros((P, len(VISUAL_CONCEPTS)), np.float32)
    chrom = frac[:, CHROMATIC].sum(1)
    share = frac[:, CHROMATIC] / np.maximum(chrom[:, None], 1e-6)
    sal = np.minimum(1.0, np.sqrt(chrom / 0.02))[:, None]
    col = share * sal
    col[col < 0.15] = 0
    cw[:, 3:_NC] = col
    cw[:, 0:3] = 0.15 * np.sqrt(frac[:, 0:3])
    win = hist.sum(0)
    cw[:, _VC_INDEX["dark" if win[0] >= win[2] else "light"]] = 0.3

    ui_w, objects = None, []
    if ui_concepts:
        from screengraph.embeddings.ui_concepts import detect_ui_concepts

        ui_w, objects = detect_ui_concepts(colors, rows, cols, pmap, buf)
    return VisualAnalysis(feats, cw, frac, ui_w, objects)


# ----------------------------------------------------------------------------
# Optional pluggable vision adapter (open_clip / SigLIP / mlx-vlm)
# ----------------------------------------------------------------------------
class VisionAdapter:
    """Interface for a learned image/text tower. Embeddings are mapped into the engine's d-dim space with
    a fixed random orthogonal map, so patch and query sides stay comparable and BQ/MaxSim/storage are unchanged."""

    name = "adapter"
    dim = 0

    def encode_patches(self, pixels: np.ndarray, rows: int, cols: int) -> np.ndarray:  # (P, dim)
        raise NotImplementedError

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:  # (len(texts), dim)
        raise NotImplementedError


class OpenClipAdapter(VisionAdapter):  # pragma: no cover - requires optional heavy deps + model weights
    """Uses ``open_clip`` if installed (``pip install open_clip_torch torch``). Patches are cropped from the
    pixel array the engine receives; capture at higher resolution for useful results."""

    def __init__(self, model: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k"):
        import open_clip  # noqa: F401
        import torch

        self._torch = torch
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model, pretrained=pretrained)
        self.tokenizer = open_clip.get_tokenizer(model)
        self.model.eval()
        self.name = f"open_clip:{model}"
        self.dim = int(self.model.text_projection.shape[1])

    def encode_patches(self, pixels, rows, cols):
        from PIL import Image

        h, w = pixels.shape[:2]
        crops = []
        for r in range(rows):
            for c in range(cols):
                tile = pixels[r * h // rows:(r + 1) * h // rows, c * w // cols:(c + 1) * w // cols, :3]
                crops.append(self.preprocess(Image.fromarray(np.ascontiguousarray(tile))))
        with self._torch.no_grad():
            e = self.model.encode_image(self._torch.stack(crops)).float().numpy()
        return l2n(e)

    def encode_text(self, texts):
        with self._torch.no_grad():
            e = self.model.encode_text(self.tokenizer(list(texts))).float().numpy()
        return l2n(e)


def load_adapter(name: Optional[str]) -> Optional[VisionAdapter]:
    if not name:
        return None
    if name.startswith("open_clip"):
        return OpenClipAdapter(*(name.split(":", 1)[1].split("/") if ":" in name else []))
    raise ValueError(f"unknown vision adapter {name!r}")


# ----------------------------------------------------------------------------
# Encoder
# ----------------------------------------------------------------------------
@dataclass
class EncodedWindow:
    patch_vecs: np.ndarray  # (P, d) float32, unit rows
    global_vec: np.ndarray  # (d,) float32, unit
    patch_concepts: list[dict[str, float]]
    objects: list[dict]

    def global_bq(self) -> np.ndarray:
        return binary_quantize(self.global_vec)

    def patch_bq(self) -> np.ndarray:
        return binary_quantize(self.patch_vecs)


class PatchEncoder:
    """Spectral-HOG-JL patch encoder fused with perceptual colour concepts and positioned AX text tokens.

        w_p = normalize( a_vis * normalize(Pi_perp R^T f_p)      generic visual descriptor (JL, colour-word-orthogonal)
                       + a_col * normalize(sum_c v_pc B[c])        perceptual colour / theme concepts
                       + a_txt * normalize(sum_t tf_pt B[t])       scrubbed AX tokens placed in the patch
                       [+ a_ui * normalize(sum_u u_pu B[u])]       optional UI-object layer (ablation)
                       [+ a_ad * normalize(O e_p)] )               optional learned adapter (open_clip ...)
    """

    def __init__(self, dim: int = 256, rows: int = 4, cols: int = 4, seed: int = 0x5C2E,
                 visual_weight: float = 1.0, text_weight: float = 0.8, descriptor_weight: float = 0.6,
                 ui_concepts: bool = False, adapter: Optional[VisionAdapter] = None):
        self.dim, self.rows, self.cols = dim, rows, cols
        self.basis = ConceptBasis(dim, seed)
        self.B_vis = self.basis.matrix(VISUAL_CONCEPTS)
        self.a, self.b, self.a_desc = visual_weight, text_weight, descriptor_weight
        self.ui_concepts = ui_concepts
        self.adapter = adapter
        rng = np.random.default_rng(seed + 17)
        # Achlioptas signed random projection R in {+1, -1}/sqrt(d): FEATURE_DIM -> d
        R = rng.choice(np.array([-1.0, 1.0], np.float32), size=(FEATURE_DIM, dim)) / np.sqrt(dim)
        # cross-modal orthogonality: remove the colour/theme concept subspace from the descriptor projection so
        # generic texture/spectrum energy can never masquerade as a colour word match
        Qc, _ = np.linalg.qr(self.B_vis.T.astype(np.float64))
        self.R = (R - (R @ Qc) @ Qc.T).astype(np.float32)
        if ui_concepts:
            from screengraph.embeddings.ui_concepts import UI_CONCEPTS

            self.ui_vocab = UI_CONCEPTS
            self.B_ui = self.basis.matrix(UI_CONCEPTS)
        if adapter is not None:
            O, _ = np.linalg.qr(np.random.default_rng(seed + 29).standard_normal((max(dim, adapter.dim), dim)))
            self.O = O[: adapter.dim].astype(np.float32)  # adapter space -> d (orthonormal columns when dim>=adapter.dim)

    # -- text -----------------------------------------------------------------
    def text_vector(self, tokens: Sequence[str], weights: Optional[Sequence[float]] = None) -> np.ndarray:
        if not tokens:
            return np.zeros(self.dim, np.float32)
        cnt: Counter = Counter()
        if weights is None:
            cnt.update(tokens)
        else:
            for t, wt in zip(tokens, weights):
                cnt[t] += wt
        keys = list(cnt)
        wts = np.array([1.0 + np.log(cnt[k]) if cnt[k] >= 1 else cnt[k] for k in keys], np.float32)
        return l2n(wts @ self.basis.matrix(keys))

    def encode_query_tokens(self, query: str) -> tuple[list[str], np.ndarray]:
        toks = list(dict.fromkeys(tokenize(query, expand_identifiers=False)))
        if self.ui_concepts:
            from screengraph.embeddings.ui_concepts import SHAPES

            colors = [t for t in toks if t in _COLOR_SET]
            toks += [f"{c}_{s}" for c in colors for s in toks if s in SHAPES]
        if not toks:
            return [], np.zeros((0, self.dim), np.float32)
        Q = self.basis.matrix(toks)
        if self.adapter is not None:
            qa = l2n(self.adapter.encode_text([query]) @ self.O)
            Q = np.vstack([Q, qa])
            toks = toks + [f"<{self.adapter.name}>"]
        return toks, Q

    def encode_text_query(self, query: str) -> np.ndarray:
        return self.text_vector(tokenize(query))

    # -- windows ----------------------------------------------------------------
    def encode(self, pixels: Optional[np.ndarray], nodes: Sequence[tuple[str, tuple[int, int, int, int]]],
               title: str = "", extra_text: str = "", buf=None, frame_size: Optional[tuple[int, int]] = None) -> EncodedWindow:
        P = self.rows * self.cols
        objects: list[dict] = []
        concepts = [dict() for _ in range(P)]
        if pixels is not None:
            h, w = pixels.shape[:2]
            va = analyze_visual(pixels, self.rows, self.cols, buf, self.ui_concepts)
            D = l2n(va.features @ self.R)
            C = va.concept_weights @ self.B_vis
            col_present = np.linalg.norm(C, axis=1) > 0
            V = self.a_desc * D + self.a * l2n(C)
            if va.ui_weights is not None:
                U = va.ui_weights @ self.B_ui
                V = V + self.a * l2n(U)
            if self.adapter is not None:
                V = V + self.a * l2n(self.adapter.encode_patches(pixels, self.rows, self.cols) @ self.O)
            vis_present = np.ones(P, bool)
            objects = va.objects
            for p in range(P):
                concepts[p] = {VISUAL_CONCEPTS[j]: round(float(va.concept_weights[p, j]), 3)
                               for j in np.flatnonzero(va.concept_weights[p] >= 0.5)}
                if va.ui_weights is not None:
                    concepts[p].update({self.ui_vocab[j]: round(float(va.ui_weights[p, j]), 3)
                                        for j in np.flatnonzero(va.ui_weights[p] >= 0.5) if "_" not in self.ui_vocab[j]})
            Vn = l2n(V)
        else:
            h, w = frame_size or (self.rows * 100, self.cols * 100)
            Vn = np.zeros((P, self.dim), np.float32)
            vis_present = col_present = np.zeros(P, bool)

        patch_tokens: list[list[tuple[str, float]]] = [[] for _ in range(P)]
        all_tokens: list[str] = []
        for text, (bx, by, bw, bh) in nodes:
            if not text:
                continue
            cy = by + bh / 2
            r = min(max(int(cy * self.rows / max(h, 1)), 0), self.rows - 1)
            pos = _token_positions(text)
            # salience: tokens of short standalone labels (badges, buttons, headings, panel titles)
            # weigh more than tokens buried in long body lines
            wt = 1.0 / np.sqrt(max(1, len(pos)) / 2.0) if len(pos) > 2 else 1.0
            for tok, rel in pos:
                x_ = bx + rel * bw
                c = min(max(int(x_ * self.cols / max(w, 1)), 0), self.cols - 1)
                patch_tokens[r * self.cols + c].append((tok, wt))
                all_tokens.append(tok)
        vocab = list(dict.fromkeys(all_tokens))
        T = np.zeros((P, self.dim), np.float32)
        if vocab:
            vi = {t: i for i, t in enumerate(vocab)}
            A = np.zeros((P, len(vocab)), np.float32)
            for p, toks in enumerate(patch_tokens):
                for t, wt in toks:
                    A[p, vi[t]] += wt
            A = np.where(A > 0, np.log1p(A) / np.log(2.0), 0.0).astype(np.float32)
            T = l2n(A @ self.basis.matrix(vocab))
        text_present = np.linalg.norm(T, axis=1) > 0
        patches = Vn + self.b * T
        empty = ~(vis_present | text_present)
        if empty.any():
            patches[empty] = self.basis.vec("__empty_patch")
        patches = l2n(patches).astype(np.float32)

        gtoks = tokenize(title) * 2 + all_tokens + tokenize(extra_text)
        tv = self.text_vector(gtoks)
        vm = Vn.mean(0) if vis_present.any() else np.zeros(self.dim, np.float32)
        g = 0.85 * tv + 0.15 * l2n(vm) if np.linalg.norm(tv) > 0 else vm
        if np.linalg.norm(g) == 0:
            g = self.basis.vec("__empty_window")
        return EncodedWindow(patches, l2n(g).astype(np.float32), concepts, objects)
