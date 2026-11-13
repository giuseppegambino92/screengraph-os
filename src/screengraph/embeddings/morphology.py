"""Cross-modal morphological concept bank (default layer of the pixel encoder).

Four generic, resolution-independent image primitives are measured on ANY RGB(A) window buffer with
numpy/scipy only, and each primitive is tied to a small, fixed set of words (the *concept bank*).
Words own seeded Gaussian vectors B[w] in the same space as query tokens, so a patch's morphological
evidence sum_k s_pk * sum_{w in bank(k)} B[w] has cosine ~ s_pk with the query word w (zero-shot: no
training, no labelled data; the word<->primitive table below is the whole "alignment").

  primitive (measured on connected components / row profiles)          -> bank words
  ------------------------------------------------------------------------------------------------
  thin curve: wide, low-fill chromatic component covering >=85% of        chart, line
    its columns (a plotted series)
  vertical excursion on that curve: column-top profile whose peak         spike, anomaly (+ chart, line)
    stands >= Z robust-MADs above the median, is narrow (<=8% of the
    width above half prominence) and steep (the local rise accounts for
    >=60% of the prominence within +-4% of the width)
  aligned solid bars: >=3 same-colour solid components on one baseline,   chart, bar
    similar widths, regular spacing, varying heights
  node-edge topology: >=3 similar compact boxes on >=2 levels joined by   tree, diagram
    thin connector components that touch >=2 boxes each
  compact labelled pill: small saturated component, aspect 1.5-7, with    badge
    enclosed contrasting holes (a label)
  floating overlay: large saturated component, not touching the window    modal
    border, centred
  row periodicity: autocorrelation peak of the horizontal-gradient row    table (weak)
    profile (text lines / table rows / log lines)

Every threshold is a fraction of the window or object size (plus small pixel floors against noise),
never a simulator coordinate, colour or glyph size. Detected objects carry their own colour name, and
colour x shape compounds ("red_spike") bind attributes to the object that has them, so "red badge" does
not match a red log line next to a green badge.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import ndimage

COLOR_NAMES = ("black", "gray", "white", "red", "orange", "yellow", "green", "cyan", "blue", "purple", "magenta")
CHROMATIC = np.array([False, False, False, True, True, True, True, True, True, True, True])
SHAPES = ("chart", "line", "bar", "spike", "modal", "badge", "tree", "diagram")
MORPH_WORDS = SHAPES + ("anomaly", "table")
COMPOUNDS = tuple(f"{c}_{s}" for c in COLOR_NAMES[3:] for s in SHAPES)
MORPH_CONCEPTS = MORPH_WORDS + COMPOUNDS
_IDX = {c: i for i, c in enumerate(MORPH_CONCEPTS)}
_COLOR_SET = frozenset(COLOR_NAMES[3:])
_NC = len(COLOR_NAMES)

# the concept bank: primitive -> {word: weight}
BANK = {
    "curve": {"chart": 1.0, "line": 1.0},
    "excursion": {"spike": 1.2, "anomaly": 1.0, "chart": 1.0, "line": 1.0},
    "bars": {"chart": 1.0, "bar": 1.0},
    "node_edge": {"tree": 1.0, "diagram": 1.0},
    "pill": {"badge": 1.2},
    "overlay": {"modal": 1.2},
    "periodic_rows": {"table": 0.4},
}


def _bind(words: dict[str, float], color: Optional[str]) -> dict[str, float]:
    out = dict(words)
    if color in _COLOR_SET:
        for w, v in words.items():
            if w in SHAPES:
                out[f"{color}_{w}"] = v
    return out


class _Writer:
    def __init__(self, P: int, h: int, w: int, rows: int, cols: int):
        self.W = np.zeros((P, len(MORPH_CONCEPTS)), np.float32)
        self.h, self.w, self.rows, self.cols = h, w, rows, cols

    def _put(self, p: int, words: dict[str, float], scale: float = 1.0) -> None:
        for name, v in words.items():
            j = _IDX[name]
            self.W[p, j] = max(self.W[p, j], v * scale)

    def point(self, y: float, x: float, words: dict[str, float]) -> None:
        r = min(max(int(y * self.rows / self.h), 0), self.rows - 1)
        c = min(max(int(x * self.cols / self.w), 0), self.cols - 1)
        self._put(r * self.cols + c, words)

    def mask(self, m: np.ndarray, sl, pmap: np.ndarray, words: dict[str, float], min_share: float = 0.08) -> None:
        ys, xs = np.nonzero(m)
        if len(ys) == 0:
            return
        pids = pmap.reshape(self.h, self.w)[ys + sl[0].start, xs + sl[1].start]
        share = np.bincount(pids, minlength=self.rows * self.cols) / len(ys)
        for p in np.flatnonzero(share >= min_share):
            self._put(int(p), words)

    def region(self, sl, words: dict[str, float], min_cover: float = 0.25) -> None:
        ph, pw = self.h / self.rows, self.w / self.cols
        for r in range(self.rows):
            for c in range(self.cols):
                oy = max(0.0, min(sl[0].stop, (r + 1) * ph) - max(sl[0].start, r * ph))
                ox = max(0.0, min(sl[1].stop, (c + 1) * pw) - max(sl[1].start, c * pw))
                cover = oy * ox / (ph * pw)
                if cover >= min_cover:
                    self._put(r * self.cols + c, words, min(1.0, cover * 1.5))


def _excursion(m: np.ndarray) -> Optional[dict]:
    """Vertical-excursion test on a thin curve mask (rows x cols of its bbox)."""
    bh, bw = m.shape
    cov = m.any(0)
    top = np.argmax(m, axis=0)
    heights = np.where(cov, bh - top, 0).astype(np.float32)
    hv = heights[cov]
    med = float(np.median(hv))
    mad = float(np.median(np.abs(hv - med)))
    pk = int(np.argmax(heights))
    peak = float(heights[pk])
    prom = peak - med
    if prom < max(3.0, 0.2 * bh):
        return None
    z = prom / (1.4826 * mad + max(1.0, 0.02 * bh))
    # full width at half prominence of THIS peak: the contiguous run around pk above med + prom/2
    half = heights > med + 0.5 * prom
    a_ = pk
    while a_ > 0 and half[a_ - 1]:
        a_ -= 1
    b_ = pk
    while b_ < bw - 1 and half[b_ + 1]:
        b_ += 1
    narrow = (b_ - a_ + 1) / bw
    k = max(3, int(round(0.04 * bw)))
    lo, hi = max(0, pk - k), min(bw, pk + k + 1)
    local_rise = peak - float(heights[lo:hi][cov[lo:hi]].min())
    steep = local_rise / prom
    # volatile series: the peak must also tower over everything outside its own neighbourhood
    rest = np.concatenate([heights[:lo], heights[hi:]])
    towers = rest.size > 0 and (peak - float(rest.max())) >= 0.25 * bh
    if narrow <= 0.08 and steep >= 0.6 and (z >= 6.0 or (z >= 3.0 and towers)):
        return {"x": pk, "y_peak": int(top[pk]), "y_base": int(bh - med), "z": round(z, 2),
                "prominence": round(prom / bh, 3), "energy": round(min(1.0, z / 12.0), 3)}
    return None


def periodic_rows(lum: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Per-patch row periodicity in [0, 1]: normalized autocorrelation peak (lag 2..10) of the
    horizontal-gradient-energy row profile; text lines, table rows and log lines are periodic."""
    Hg, Wg = lum.shape
    hg = np.abs(np.diff(lum, axis=1))
    hg = np.pad(hg, ((0, 0), (0, 1)))
    ph, pw = Hg // rows, Wg // cols
    prof = hg[: ph * rows, : pw * cols].reshape(rows, ph, cols, pw).sum(3).transpose(0, 2, 1).reshape(rows * cols, ph)
    prof = prof / pw
    energy = prof.mean(1)
    z = prof - prof.mean(1, keepdims=True)
    f = np.fft.rfft(z, n=2 * ph, axis=1)
    acf = np.fft.irfft(f * np.conj(f), axis=1)[:, :ph]
    acf = acf / np.maximum(acf[:, :1], 1e-9)
    lags = acf[:, 2:11]
    is_peak = (lags[:, 1:-1] >= lags[:, :-2]) & (lags[:, 1:-1] >= lags[:, 2:])
    best = np.where(is_peak, lags[:, 1:-1], -1.0).max(1)
    s = np.clip((best - 0.3) / 0.4, 0.0, 1.0) * (energy > 0.02)
    hg.fill(0)
    prof.fill(0)
    z.fill(0)
    return s.astype(np.float32)


def detect_morphology(pixels: np.ndarray, colors: np.ndarray, rows: int, cols: int, pmap: np.ndarray,
                      lum: Optional[np.ndarray] = None, buf=None) -> tuple[np.ndarray, list[dict]]:
    """pixels: (h, w, 3) uint8, colors: (h, w) basic-colour index map.
    Returns (P x len(MORPH_CONCEPTS)) weights and the detected objects (bbox in window pixels)."""
    h, w = colors.shape
    P = rows * cols
    wr = _Writer(P, h, w, rows, cols)
    objects: list[dict] = []
    win_area = float(h * w)

    # salient-colour mask by channel spread (chroma >= 12% of full scale, not near-black) rather than the
    # colour-name LUT cut: anti-aliased thin strokes keep their chroma while their saturation halves
    mx = pixels.max(-1)
    chrom = ((mx - pixels.min(-1)) >= 31) & (mx >= 40)
    mx.fill(0)
    labels, n = ndimage.label(chrom, structure=np.ones((3, 3), bool))
    if buf is not None:
        buf.track(chrom)
        buf.track(labels)
    comps = []
    if n:
        # vectorized per-component statistics: area, bbox, dominant colour, enclosed-hole area
        areas = np.bincount(labels.ravel(), minlength=n + 1)
        ccount = np.bincount((labels.astype(np.int64) * _NC + colors).ravel(), minlength=(n + 1) * _NC).reshape(n + 1, _NC)
        dom = ccount.argmax(1)
        for lab, sl in enumerate(ndimage.find_objects(labels), start=1):
            if sl is None or areas[lab] < 4:
                continue
            bh, bw = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
            if bh < 3 or bw < 2:
                continue
            a = float(areas[lab])
            comps.append({"lab": lab, "sl": sl, "bh": bh, "bw": bw, "area": a, "fill": a / (bh * bw),
                          "holes": 0.0,
                          "color": COLOR_NAMES[int(dom[lab])],
                          "touches": bool(sl[0].start == 0 or sl[1].start == 0 or sl[0].stop >= h or sl[1].stop >= w)})

        # enclosed holes (a label inside a pill) only matter for pill-shaped components that can resolve one:
        # 4-connected background regions not reaching the window border, attributed (per row span) to the
        # component bounding them on the left. One label pass, no iterative hole filling.
        if any(_pill_shape(c, h, w, win_area) for c in comps):
            bgl, nb = ndimage.label(~chrom)
            ext = np.zeros(nb + 1, bool)
            ext[np.concatenate([bgl[0], bgl[-1], bgl[:, 0], bgl[:, -1]])] = True
            ext[0] = True  # label 0 = the chromatic pixels themselves
            hole_px = ~ext[bgl]
            bgl.fill(0)
            if hole_px.any():
                col = np.where(chrom, np.arange(w)[None, :], 0)
                np.maximum.accumulate(col, axis=1, out=col)
                left = np.take_along_axis(labels, col, axis=1)
                hole_area = np.bincount(left[hole_px], minlength=n + 1)
                left.fill(0)
                col.fill(0)
                for c in comps:
                    ha = float(hole_area[c["lab"]])
                    c["holes"] = ha / (c["area"] + ha)
            hole_px.fill(False)

    def mask_of(c):
        if "mask" not in c:
            c["mask"] = labels[c["sl"]] == c["lab"]
        return c["mask"]

    used: set[int] = set()

    # ---- thin curves (+ vertical excursions) ----------------------------------------
    for c in comps:
        bh, bw = c["bh"], c["bw"]
        if bw < max(12, 0.10 * w) or bh < max(5, 0.04 * h) or c["fill"] > 0.3:
            continue
        m = mask_of(c)
        cov = m.any(0)
        if cov.mean() < 0.85 or float(np.median(m.sum(0)[cov])) > max(3.0, 0.08 * bh):
            continue
        used.add(c["lab"])
        sl = c["sl"]
        objects.append({"type": "line_chart", "color": c["color"], "bbox": _bb(sl)})
        wr.mask(m, sl, pmap, _bind(BANK["curve"], c["color"]))
        ex = _excursion(m)
        if ex is not None:
            x = sl[1].start + ex["x"]
            yp, yb = sl[0].start + ex["y_peak"], sl[0].start + ex["y_base"]
            objects.append({"type": "spike", "color": c["color"], "bbox": _bb(sl), "x": x, "y": yp,
                            "z": ex["z"], "prominence": ex["prominence"], "energy": ex["energy"]})
            for yy in (yp, (yp + yb) // 2, yb):
                wr.point(yy, x, _bind(BANK["excursion"], c["color"]))

    # ---- aligned solid bars ------------------------------------------------------
    solid = [c for c in comps if c["lab"] not in used and c["fill"] >= 0.85 and not c["touches"]]
    tol = max(2, int(0.02 * h))
    by_color: dict[str, list] = {}
    for c in solid:
        by_color.setdefault(c["color"], []).append(c)
    for color, cs in by_color.items():
        cs = sorted(cs, key=lambda c: c["sl"][0].stop)
        clusters: list[list] = []
        for c in cs:  # greedy 1-D clustering of baselines
            if clusters and c["sl"][0].stop - clusters[-1][0]["sl"][0].stop <= tol:
                clusters[-1].append(c)
            else:
                clusters.append([c])
        for grp in clusters:
            if len(grp) < 3:
                continue
            hs = np.array([c["bh"] for c in grp], np.float32)
            ws = np.array([c["bw"] for c in grp], np.float32)
            xs = np.sort([c["sl"][1].start for c in grp])
            gaps = np.diff(xs).astype(np.float32)
            if (hs.max() < max(4, 0.03 * h) or hs.std() / hs.max() < 0.15 or ws.max() / ws.min() > 2.0
                    or (hs / ws).max() < 1.0 or (gaps.size and gaps.std() / max(gaps.mean(), 1) > 0.5)):
                continue
            objects.append({"type": "bar_chart", "color": color, "bars": len(grp)})
            for c in grp:
                used.add(c["lab"])
                wr.mask(mask_of(c), c["sl"], pmap, _bind(BANK["bars"], color))

    # ---- compact components: floating overlays, node-edge diagrams, labelled pills -------
    pills, boxes = [], []
    for c in comps:
        if c["lab"] in used or c["touches"]:
            continue
        sl, bh, bw, a, fill = c["sl"], c["bh"], c["bw"], c["area"], c["fill"]
        aspect = bw / bh
        cy, cx = (sl[0].start + sl[0].stop) / 2, (sl[1].start + sl[1].stop) / 2
        if (0.03 * win_area <= a <= 0.6 * win_area and fill >= 0.6 and bw >= 0.15 * w and 0.5 <= aspect <= 4.5
                and abs(cx / w - 0.5) <= 0.25 and abs(cy / h - 0.5) <= 0.3):
            objects.append({"type": "modal", "color": c["color"], "bbox": _bb(sl)})
            wr.region(sl, _bind(BANK["overlay"], c["color"]))
            continue
        if a > 0.03 * win_area or fill < 0.6:
            continue
        if fill >= 0.7 and 0.6 <= aspect <= 5.0 and a >= 0.0005 * win_area and bh >= 4:
            boxes.append(c)  # node candidate (a labelled box may still turn out to be a pill)
        if _pill_shape(c, h, w, win_area) and 0.03 <= c["holes"] <= 0.5:
            pills.append(c)
    in_diagram = _node_edge(pixels, boxes, wr, objects, h, w, buf)
    for c in pills:  # a labelled box wired into a diagram is a node, not a badge
        if c["lab"] in in_diagram:
            continue
        sl = c["sl"]
        objects.append({"type": "badge", "color": c["color"], "bbox": _bb(sl)})
        wr.point((sl[0].start + sl[0].stop) / 2, (sl[1].start + sl[1].stop) / 2, _bind(BANK["pill"], c["color"]))

    # ---- row periodicity ------------------------------------------------------------
    if lum is not None:
        per = periodic_rows(lum, rows, cols)
        j = _IDX["table"]
        wr.W[:, j] = np.maximum(wr.W[:, j], per * BANK["periodic_rows"]["table"])
    # component silhouettes are pixel-derived binary images: wipe them (chrom/labels are also tracked)
    for c in comps:
        if "mask" in c:
            c["mask"].fill(False)
    chrom.fill(False)
    labels.fill(0)
    return wr.W, objects


def _pill_shape(c: dict, h: int, w: int, win_area: float) -> bool:
    """Small, wide, solid-ish component tall enough to enclose a resolvable label (>= 4 px)."""
    return (not c["touches"] and 4 <= c["bh"] <= max(5, 0.06 * h) and 1.5 <= c["bw"] / c["bh"] <= 7.0
            and c["area"] <= 0.01 * win_area and c["fill"] >= 0.6)


def _bb(sl) -> tuple[int, int, int, int]:
    """(x, y, w, h) like AXNode.bbox."""
    return (int(sl[1].start), int(sl[0].start), int(sl[1].stop - sl[1].start), int(sl[0].stop - sl[0].start))


def _node_edge(pixels, boxes, wr: _Writer, objects, h, w, buf) -> set:
    """Returns the labels of boxes wired into a detected diagram."""
    members: set = set()
    if len(boxes) < 3:
        return members
    by_color: dict[str, list] = {}
    for b in boxes:
        by_color.setdefault(b["color"], []).append(b)
    groups = []
    for color, bs in by_color.items():
        if len(bs) < 3:
            continue
        med = float(np.median([b["area"] for b in bs]))
        bs = [b for b in bs if med / 2.5 <= b["area"] <= med * 2.5]
        if len(bs) >= 3:
            groups.append((color, bs))
    if not groups:
        return members
    # a hierarchy has >= 2 levels and siblings (>= 2 boxes on one level); cheap geometric screen first
    cand = []
    for color, bs in groups:
        mh = float(np.median([b["bh"] for b in bs]))
        tops = sorted(b["sl"][0].start for b in bs)
        lv = np.cumsum([0] + [1 if b2 - a_ > 0.5 * mh else 0 for a_, b2 in zip(tops, tops[1:])])
        if lv[-1] + 1 >= 2 and np.bincount(lv).max() >= 2:
            cand.append((color, bs, int(lv[-1]) + 1))
    if not cand:
        return members
    # dominant background colour (4-bit/channel mode); foreground minus boxes = connector candidates
    q = (pixels[..., 0].astype(np.int32) >> 4) << 8 | (pixels[..., 1].astype(np.int32) >> 4) << 4 | (pixels[..., 2] >> 4)
    bgq = int(np.bincount(q.ravel(), minlength=4096).argmax())
    q.fill(0)
    bg = np.array([(bgq >> 8) * 16 + 8, ((bgq >> 4) & 15) * 16 + 8, (bgq & 15) * 16 + 8], np.int32)
    for color, bs, levels in cand:
        y0 = max(0, min(b["sl"][0].start for b in bs) - 3)
        y1 = min(h, max(b["sl"][0].stop for b in bs) + 3)
        x0 = max(0, min(b["sl"][1].start for b in bs) - 3)
        x1 = min(w, max(b["sl"][1].stop for b in bs) + 3)
        sub = pixels[y0:y1, x0:x1].astype(np.int32)
        fg = np.abs(sub - bg).max(-1) > 40
        sub.fill(0)
        boxmask = np.zeros(fg.shape, np.int32)
        for k, b in enumerate(bs, start=1):
            sl = b["sl"]
            boxmask[max(0, sl[0].start - y0 - 1):sl[0].stop - y0 + 1, max(0, sl[1].start - x0 - 1):sl[1].stop - x0 + 1] = k
        conn = fg & (boxmask == 0)
        lab, n = ndimage.label(conn, structure=np.ones((3, 3), bool))
        pairs = set()
        if n:
            grown = ndimage.grey_dilation(boxmask, size=(3, 3))
            # thin = low fill of its bbox or <= 3 px across
            areas = np.bincount(lab.ravel(), minlength=n + 1)
            thin = np.zeros(n + 1, bool)
            for i, sl in enumerate(ndimage.find_objects(lab), start=1):
                bh, bw = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
                thin[i] = areas[i] / (bh * bw) <= 0.4 or min(bh, bw) <= 3
            sel = (lab > 0) & (grown > 0)
            keys = np.unique(lab[sel].astype(np.int64) * 4096 + grown[sel])
            touched: dict[int, list] = {}
            for key in keys.tolist():
                ci, bi = divmod(key, 4096)
                if thin[ci]:
                    touched.setdefault(ci, []).append(bi)
            for t in touched.values():
                pairs.update((t[i], t[j]) for i in range(len(t)) for j in range(i + 1, len(t)))
            grown.fill(0)
        lab.fill(0)
        fg.fill(False)
        if len(pairs) >= 2:
            ys = [b["sl"][0].start for b in bs] + [b["sl"][0].stop for b in bs]
            xs = [b["sl"][1].start for b in bs] + [b["sl"][1].stop for b in bs]
            objects.append({"type": "tree", "color": color, "boxes": len(bs), "edges": len(pairs), "levels": levels,
                            "bbox": (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))})
            for b in bs:
                sl = b["sl"]
                members.add(b["lab"])
                wr.point((sl[0].start + sl[0].stop) / 2, (sl[1].start + sl[1].stop) / 2, _bind(BANK["node_edge"], color))
    return members
