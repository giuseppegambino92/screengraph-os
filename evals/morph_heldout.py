"""Held-out generalization check for the morphological concept bank.

The workspace simulator and the detectors were developed side by side, so 6/6 on the simulator says
little about other renderings. This module draws scenes with an INDEPENDENT renderer: random canvas
sizes and aspect ratios, random light/dark backgrounds, random hues (not the simulator palette),
1-3 px strokes, optional anti-aliasing blur and sensor noise, a 2x up-scaled variant of every scene,
and hard negatives (volatile series without a spike, unconnected card grids, docked side panels,
unlabelled colour squares, coloured text). Every scene contains distractor text lines.

Reported per concept: window-level recall on positives and false-positive rate on negatives.
Run:  python -m evals.morph_heldout [--n 60] [--seed 1234]
"""

from __future__ import annotations

import argparse
import colorsys
import json
from collections import defaultdict

import numpy as np
from scipy import ndimage

from screengraph.embeddings.patch_encoder import analyze_visual


def _hue_rgb(rng, s=(0.55, 0.95), v=(0.65, 0.95)) -> np.ndarray:
    r, g, b = colorsys.hsv_to_rgb(rng.uniform(0, 1), rng.uniform(*s), rng.uniform(*v))
    return np.array([r, g, b]) * 255


class Scene:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.h = int(rng.integers(120, 320))
        self.w = int(rng.integers(150, 420))
        dark = rng.random() < 0.5
        base = rng.uniform(12, 40) if dark else rng.uniform(225, 252)
        self.bg = np.clip(base + rng.normal(0, 4, 3), 0, 255)
        self.ink = np.array([200.0] * 3) if dark else np.array([45.0] * 3)
        self.px = np.empty((self.h, self.w, 3), np.float32)
        self.px[:] = self.bg

    def rect(self, x, y, w, h, col):
        self.px[max(0, y):max(0, y + h), max(0, x):max(0, x + w)] = col

    def text(self, x, y, n, col, hgt=3):
        """A run of glyph-like strokes (variable height, word gaps)."""
        r = self.rng
        for i in range(n):
            if x + i >= self.w or r.random() < 0.15:
                continue
            g = int(r.integers(1, hgt + 1))
            self.px[y + hgt - g:y + hgt, x + i] = col

    def distractor_text(self, k=None, color_words=0.0, avoid=None):
        r = self.rng
        for _ in range(k if k is not None else int(r.integers(2, 6))):
            y = int(r.integers(2, self.h - 6))
            x = int(r.integers(2, max(3, self.w // 3)))
            if avoid and avoid[1] - 6 <= y <= avoid[1] + avoid[3] + 2:
                continue
            col = _hue_rgb(r) if r.random() < color_words else self.ink
            self.text(x, y, int(r.integers(10, max(11, self.w // 2))), col)

    def vline(self, x, y0, y1, col):
        a, b = sorted((int(y0), int(y1)))
        if 0 <= x < self.w:
            self.px[max(0, a):min(self.h, b + 1), x] = col

    def stroke(self, xs, ys, col, thick):
        prev = ys[0]
        for x, y in zip(xs, ys):
            a, b = sorted((int(prev), int(y)))
            for t in range(thick):
                if 0 <= x + t < self.w:
                    self.px[max(0, a):min(self.h, b + 1), x + t] = col
            prev = y

    def finish(self, blur: bool, noise: float, scale: int, noise_rng=None) -> np.ndarray:
        px = self.px
        if blur:
            px = ndimage.uniform_filter(px, size=(2, 2, 1))
        if scale > 1:
            px = np.repeat(np.repeat(px, scale, 0), scale, 1)
        if noise > 0:
            px = px + (noise_rng or self.rng).normal(0, noise, px.shape)
        return np.clip(px, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------- scene builders
def series(sc: Scene, spike: bool, volatile: bool):
    r = sc.rng
    aw, ah = int(sc.w * r.uniform(0.35, 0.85)), int(sc.h * r.uniform(0.2, 0.45))
    x0, y0 = int(r.integers(2, sc.w - aw - 2)), int(r.integers(8, sc.h - ah - 4))
    n = aw
    walk = np.cumsum(r.normal(0, 0.03 if volatile else 0.012, n))
    vals = np.clip(0.4 + walk - walk.mean(), 0.1, 0.65)
    if spike:
        sx = int(r.integers(int(0.1 * n), int(0.9 * n)))
        wid = max(1, int(n * r.uniform(0.005, 0.025)))
        peak = r.uniform(0.85, 0.98)
        lo, hi = max(0, sx - wid), min(n, sx + wid + 1)
        vals[lo:hi] = np.maximum(vals[lo:hi], np.interp(np.arange(lo, hi), [lo, sx, hi - 1], [vals[lo], peak, vals[hi - 1]]))
    ys = y0 + ah - 1 - (vals * (ah - 2)).astype(int)
    sc.stroke(np.arange(x0, x0 + n), ys, _hue_rgb(r), int(r.integers(1, 3)))
    sc.text(x0, max(0, y0 - 6), int(r.integers(10, 30)), sc.ink)  # panel caption
    return (x0, y0, aw, ah)


def bars(sc: Scene):
    r = sc.rng
    nb = int(r.integers(4, 12))
    bw = int(r.integers(2, 6))
    gap = bw + int(r.integers(1, 5))
    ah = int(sc.h * r.uniform(0.2, 0.45))
    x0, base = int(r.integers(2, max(3, sc.w - nb * gap - 2))), int(r.integers(ah + 8, sc.h - 3))
    col = _hue_rgb(r)
    for k in range(nb):
        bh = max(3, int(r.uniform(0.2, 1.0) * ah))
        sc.rect(x0 + k * gap, base - bh, bw, bh, col)


def tree(sc: Scene, connected: bool = True):
    r = sc.rng
    bw, bh = int(sc.w * r.uniform(0.07, 0.12)), max(5, int(sc.h * r.uniform(0.035, 0.06)))
    levels = [1, 2, 4] if r.random() < 0.6 else [1, 3]
    vgap = int(bh * r.uniform(1.4, 2.4))
    span = int(sc.w * r.uniform(0.55, 0.9))
    left = (sc.w - span) // 2
    y = int(r.integers(4, max(5, sc.h - len(levels) * (bh + vgap) - 4)))
    col, conn = _hue_rgb(r), np.clip(sc.ink * 0.5 + sc.bg * 0.5, 0, 255)
    centers = [[(left + int((k + 0.5) * span / n), y + li * (bh + vgap)) for k in range(n)] for li, n in enumerate(levels)]
    if connected:
        for li in range(1, len(levels)):
            for k, (cx, ly) in enumerate(centers[li]):
                px_, py_ = centers[li - 1][k * len(centers[li - 1]) // len(centers[li])]
                if r.random() < 0.5:  # elbow connector
                    mid = py_ + bh + (vgap // 2)
                    sc.vline(px_, py_ + bh, mid, conn)
                    a, b = sorted((px_, cx))
                    sc.px[mid, a:b + 1] = conn
                    sc.vline(cx, mid, ly - 1, conn)
                else:  # straight diagonal connector
                    n = max(abs(cx - px_), 1)
                    xs = np.linspace(px_, cx, n + 1).astype(int)
                    ys = np.linspace(py_ + bh, ly - 1, n + 1).astype(int)
                    for xx, yy in zip(xs, ys):
                        sc.px[yy, xx] = conn
                    sc.vline(cx, ys[-1], ly - 1, conn)
    for row in centers:
        for cx, ly in row:
            sc.rect(cx - bw // 2, ly, bw, bh, col)
            sc.text(cx - bw // 2 + 2, ly + max(1, bh // 2 - 1), max(2, bw - 4), sc.bg, hgt=min(3, bh - 2))


def card_grid(sc: Scene):
    """Hard negative for tree: same-colour boxes on a grid, no connectors."""
    r = sc.rng
    bw, bh = int(sc.w * 0.12), max(5, int(sc.h * 0.05))
    col = _hue_rgb(r)
    for i in range(2):
        for j in range(3):
            sc.rect(10 + j * (bw + 12), 20 + i * (bh + 14), bw, bh, col)
            sc.text(12 + j * (bw + 12), 21 + i * (bh + 14), max(2, bw - 4), sc.bg, hgt=min(3, bh - 2))


def pill(sc: Scene):
    r = sc.rng
    bh = max(4, int(sc.h * r.uniform(0.03, 0.05)))
    bw = int(bh * r.uniform(2.0, 5.0))
    x, y = int(r.integers(4, sc.w - bw - 4)), int(r.integers(4, sc.h - bh - 4))
    sc.rect(x, y, bw, bh, _hue_rgb(r))
    sc.text(x + 2, y + 1, bw - 4, sc.bg if sc.bg.mean() > 128 else np.array([250.0] * 3), hgt=max(1, bh - 2))


def square(sc: Scene):
    """Hard negative for badge: unlabelled colour squares (avatars, status dots)."""
    r = sc.rng
    s = max(4, int(sc.h * 0.04))
    for k in range(3):
        sc.rect(int(r.integers(2, sc.w - s - 2)), int(r.integers(2, sc.h - s - 2)), s, s, _hue_rgb(r))


def modal(sc: Scene, dim: bool):
    r = sc.rng
    mw, mh = int(sc.w * r.uniform(0.3, 0.55)), int(sc.h * r.uniform(0.22, 0.45))
    x, y = (sc.w - mw) // 2 + int(r.integers(-5, 6)), (sc.h - mh) // 2 + int(r.integers(-5, 6))
    if dim:
        sc.px *= 0.5
    sc.rect(x, y, mw, mh, _hue_rgb(r))
    for i in range(3):
        sc.text(x + 4, y + 4 + 6 * i, mw - 8, np.array([250.0] * 3))


def side_panel(sc: Scene):
    """Hard negative for modal: a large docked coloured panel touching the window edge."""
    r = sc.rng
    pw = int(sc.w * r.uniform(0.15, 0.3))
    sc.rect(0, 0, pw, sc.h, _hue_rgb(r, s=(0.4, 0.8), v=(0.3, 0.6)))


CASES = {
    "spike": (lambda sc: series(sc, True, False), {"spike"}),
    "spike_volatile": (lambda sc: series(sc, True, True), {"spike"}),
    "line_no_spike": (lambda sc: series(sc, False, False), set()),
    "volatile_no_spike": (lambda sc: series(sc, False, True), set()),
    "bars": (bars, {"bar_chart"}),
    "tree": (tree, {"tree"}),
    "boxes_no_connectors": (lambda sc: tree(sc, connected=False), set()),
    "card_grid": (card_grid, set()),
    "badge": (pill, {"badge"}),
    "unlabelled_squares": (square, set()),
    "modal": (lambda sc: modal(sc, False), {"modal"}),
    "modal_dimmed": (lambda sc: modal(sc, True), {"modal"}),
    "side_panel": (side_panel, set()),
    "coloured_text_only": (lambda sc: sc.distractor_text(8, color_words=0.8), set()),
}
CONCEPTS = ("spike", "bar_chart", "tree", "badge", "modal")


def run(n: int = 60, seed: int = 1234) -> dict:
    rng = np.random.default_rng(seed)
    tally = defaultdict(lambda: defaultdict(int))
    for case, (build, gold) in CASES.items():
        for i in range(n):
            sc = Scene(rng)
            sc.distractor_text()
            build(sc)
            scale = 2 if i % 2 else 1
            px = sc.finish(blur=bool(rng.random() < 0.5), noise=float(rng.choice([0.0, 2.0, 4.0])), scale=scale)
            det = {o["type"] for o in analyze_visual(px, 4, 4).objects}
            for c in CONCEPTS:
                key = ("pos" if c in gold else "neg")
                tally[c][f"{key}_n"] += 1
                tally[c][f"{key}_hit"] += int(c in det)
                tally[c][f"{key}_hit_scale{scale}"] += int(c in det)
                tally[c][f"{key}_n_scale{scale}"] += 1
            tally["_cases"][case + ":" + ",".join(sorted(det & set(CONCEPTS)) or ["-"])] += 1
    out = {}
    for c in CONCEPTS:
        t = tally[c]
        out[c] = {"recall": round(t["pos_hit"] / max(1, t["pos_n"]), 3), "positives": t["pos_n"],
                  "false_positive_rate": round(t["neg_hit"] / max(1, t["neg_n"]), 3), "negatives": t["neg_n"],
                  "recall_1x": round(t["pos_hit_scale1"] / max(1, t["pos_n_scale1"]), 3),
                  "recall_2x": round(t["pos_hit_scale2"] / max(1, t["pos_n_scale2"]), 3),
                  "fpr_1x": round(t["neg_hit_scale1"] / max(1, t["neg_n_scale1"]), 3),
                  "fpr_2x": round(t["neg_hit_scale2"] / max(1, t["neg_n_scale2"]), 3)}
    return {"n_per_case": n, "seed": seed, "per_concept": out, "detections_by_case": dict(tally["_cases"])}


def _series_pair(sc: Scene) -> tuple[np.ndarray, np.ndarray]:
    """The same calm or volatile series drawn twice: without and with a narrow spike."""
    r = sc.rng
    aw, ah = int(sc.w * r.uniform(0.35, 0.85)), int(sc.h * r.uniform(0.2, 0.45))
    x0, y0 = int(r.integers(2, sc.w - aw - 2)), int(r.integers(8, sc.h - ah - 4))
    walk = np.cumsum(r.normal(0, 0.03 if r.random() < 0.5 else 0.012, aw))
    vals = np.clip(0.4 + walk - walk.mean(), 0.1, 0.65)
    col, thick = _hue_rgb(r), int(r.integers(1, 3))
    base = sc.px.copy()
    out = []
    for spike in (False, True):
        sc.px[:] = base
        v = vals.copy()
        if spike:
            sx = int(r.integers(int(0.1 * aw), int(0.9 * aw)))
            wid = max(1, int(aw * r.uniform(0.005, 0.025)))
            lo, hi = max(0, sx - wid), min(aw, sx + wid + 1)
            v[lo:hi] = np.maximum(v[lo:hi], np.interp(np.arange(lo, hi), [lo, sx, hi - 1],
                                                      [v[lo], r.uniform(0.85, 0.98), v[hi - 1]]))
        sc.stroke(np.arange(x0, x0 + aw), y0 + ah - 1 - (v * (ah - 2)).astype(int), col, thick)
        out.append(sc.px.copy())
    return out[0], out[1]


def _cursor(px: np.ndarray, y: int, x: int, s: int) -> None:
    px[y:y + 3 * s, x:x + 2 * s] = 255
    px[y + 3 * s:y + 4 * s, x:x + s] = 0


def gate_probe(n: int = 120, seed: int = 4321) -> dict:
    """Perceptual gate on the independent renderer: (a) spike-only redraws must PASS, (b) cursor moves and
    caret blinks on an otherwise identical frame must be DROPPED. Noise is a static texture of the scene
    (the same field in every frame of a pair): screen captures are digital, an unchanged window redraws
    byte-identically. (With independent per-frame sensor noise the gate's dHash passes nearly every frame.)"""
    from screengraph.capture.phash import PerceptualGate, ax_tree_hash

    rng = np.random.default_rng(seed)
    t = defaultdict(int)
    for i in range(n):
        scale = 2 if i % 2 else 1
        blur, noise = bool(rng.random() < 0.5), float(rng.choice([0.0, 2.0, 4.0]))
        sc = Scene(rng)
        sc.distractor_text()
        a, b = _series_pair(sc)

        nseed = int(rng.integers(1 << 31))

        def fin(px):
            sc.px = px.astype(np.float32)
            return sc.finish(blur, noise, scale, noise_rng=np.random.default_rng(nseed))

        g = PerceptualGate(4, 6, quadrant_threshold=2)
        g.check("w", "ax", fin(a))
        d = g.check("w", "ax", fin(b))
        t["spike_n"] += 1
        t["spike_pass"] += d.passed
        t["spike_pass_by_excursion_stage"] += d.reason == "visual_excursion"
        # cursor / caret noise on the calm frame
        base = fin(a)
        g = PerceptualGate(4, 6, quadrant_threshold=2)
        g.check("w", "ax", base)
        H, W = base.shape[:2]
        for _ in range(3):
            f = fin(a)
            if rng.random() < 0.7:
                _cursor(f, int(rng.integers(0, H - 5 * scale)), int(rng.integers(0, W - 3 * scale)), scale)
            else:  # caret blink: 1 px wide, one text line tall
                y, x = int(rng.integers(0, H - 4 * scale)), int(rng.integers(0, W - 2))
                f[y:y + 3 * scale, x:x + scale] = 230
            d = g.check("w", "ax", f)
            t["cursor_n"] += 1
            t["cursor_false_pass"] += d.passed
            t["cursor_false_pass_by_excursion_stage"] += d.reason == "visual_excursion"
    return {"pairs": n, "seed": seed,
            "spike_only_pass_rate": round(t["spike_pass"] / t["spike_n"], 3),
            "spike_passes_via_excursion_stage": t["spike_pass_by_excursion_stage"],
            "cursor_caret_frames": t["cursor_n"],
            "cursor_caret_false_pass_rate": round(t["cursor_false_pass"] / t["cursor_n"], 3),
            "cursor_caret_false_passes_via_excursion_stage": t["cursor_false_pass_by_excursion_stage"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--gate", action="store_true", help="run the perceptual-gate probe instead")
    a = ap.parse_args()
    print(json.dumps(gate_probe(2 * a.n, a.seed) if a.gate else run(a.n, a.seed), indent=1))
