"""OPTIONAL UI-object concept layer (off by default).

These are hand-written connected-component detectors for line charts (+ narrow
spikes), bar charts, solid modal panels, badges and box-and-connector trees,
with colour x shape compound concepts ("red_badge"). They were developed
together with the workspace simulator's renderers, so their accuracy on the
synthetic benchmark says little about real screens. They are kept only as an
explicit, measurable ablation (``Config.ui_concepts = True``); the default
encoder is the generic Spectral-HOG-JL representation in ``patch_encoder.py``.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

COLOR_NAMES = ("black", "gray", "white", "red", "orange", "yellow", "green", "cyan", "blue", "purple", "magenta")
CHROMATIC = np.array([False, False, False, True, True, True, True, True, True, True, True])
SHAPES = ("chart", "line", "bar", "spike", "modal", "badge", "tree", "diagram")
COMPOUNDS = tuple(f"{c}_{s}" for c in COLOR_NAMES[3:] for s in SHAPES)
UI_CONCEPTS = SHAPES + COMPOUNDS
_VC_INDEX = {c: i for i, c in enumerate(UI_CONCEPTS)}
_COLOR_SET = frozenset(COLOR_NAMES[3:])
_SHAPE_SET = frozenset(SHAPES)
_NC = len(COLOR_NAMES)


def detect_ui_concepts(colors: np.ndarray, rows: int, cols: int, pmap: np.ndarray, buf=None) -> tuple[np.ndarray, list[dict]]:
    """colors: (h, w) colour-name index map. Returns (P x len(UI_CONCEPTS)) weights and detected objects."""
    h, w = colors.shape
    W = np.zeros((rows * cols, len(UI_CONCEPTS)), np.float32)
    # ---- connected components of chromatic pixels --------------------------
    # Charts, spikes, modals, badges and box-trees are detected at *object* level
    # (so a spike crossing a patch boundary is still seen whole) and their concepts
    # are then assigned to the patches the object occupies.
    objects: list[dict] = []
    chrom_mask = CHROMATIC[colors]
    if buf is not None:
        buf.track(chrom_mask)
    labels, n = ndimage.label(chrom_mask, structure=np.ones((3, 3), bool))
    if buf is not None:
        buf.track(labels)
    if n:
        areas = np.bincount(labels.ravel(), minlength=n + 1)
        slices = ndimage.find_objects(labels)
        win_area = float(h * w)
        comps = []
        for lab, sl in enumerate(slices, start=1):
            if sl is None:
                continue
            bh, bw = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
            if bh < 4 or bw < 2:
                continue
            m = labels[sl] == lab
            col_name = int(np.bincount(colors[sl][m], minlength=_NC).argmax())
            comps.append({"lab": lab, "sl": sl, "bh": bh, "bw": bw, "area": float(areas[lab]),
                          "fill": float(areas[lab]) / (bh * bw), "color": col_name, "mask": m,
                          "touches": (sl[0].start == 0) + (sl[1].start == 0) + (sl[0].stop >= h) + (sl[1].stop >= w)})

        used: set[int] = set()
        # line charts (+ spikes): thin polyline spanning many columns
        for cpt in comps:
            m, bh, bw = cpt["mask"], cpt["bh"], cpt["bw"]
            if bw < max(16, 0.12 * w) or cpt["fill"] > 0.25:
                continue
            cols_cov = m.any(0)
            if cols_cov.mean() < 0.85 or m.sum(0)[cols_cov].mean() > 4.0:
                continue
            used.add(cpt["lab"])
            sl = cpt["sl"]
            objects.append({"type": "line_chart", "color": COLOR_NAMES[cpt["color"]]})
            _assign_mask(W, m, sl, pmap, h, w, rows, cols, {"chart": 1.0, "line": 1.0, COLOR_NAMES[cpt["color"]]: 1.0})
            top = np.argmax(m, axis=0)
            heights = (bh - top).astype(np.float32)
            med = float(np.median(heights))
            mad = float(np.median(np.abs(heights - med))) + 1.0
            pk = int(np.argmax(heights))
            peak = float(heights[pk])
            # spike = narrow peak that towers over the line's typical level (robust MAD test) or
            # over everything outside its own +-4 column neighbourhood (volatile series)
            rest = np.concatenate([heights[: max(0, pk - 4)], heights[pk + 5:]])
            towers = rest.size > 0 and peak - float(rest.max()) >= max(0.25 * bh, 8.0)
            if peak - med >= max(6.0 * mad, 0.35 * bh) or (towers and peak - med >= 0.35 * bh):
                high = heights > med + 0.5 * (peak - med)
                if high.sum() <= max(4, 0.08 * bw):
                    x = sl[1].start + pk
                    y_peak, y_base = sl[0].start + int(top[pk]), sl[0].start + int(bh - med)
                    objects.append({"type": "spike", "x": x, "y": y_peak})
                    for yy in (y_peak, (y_peak + y_base) // 2, y_base):
                        _assign_point(W, yy, x, h, w, rows, cols, {"spike": 1.2, "chart": 1.0, "line": 1.0,
                                                                   COLOR_NAMES[cpt["color"]]: 1.0})

        # bar charts: >= 3 solid same-colour bars sharing a baseline
        solid = [c for c in comps if c["lab"] not in used and c["fill"] >= 0.85 and c["touches"] == 0]
        groups: dict[tuple[int, int], list] = {}
        for c in solid:
            groups.setdefault((c["color"], c["sl"][0].stop // 2), []).append(c)
        for (col_name, _), bars in groups.items():
            if len(bars) >= 3 and np.std([b["bh"] for b in bars]) >= 2.0 and max(b["bh"] / b["bw"] for b in bars) >= 1.0:
                objects.append({"type": "bar_chart", "color": COLOR_NAMES[col_name], "bars": len(bars)})
                for b in bars:
                    used.add(b["lab"])
                    _assign_mask(W, b["mask"], b["sl"], pmap, h, w, rows, cols,
                                 {"chart": 1.0, "bar": 1.0, COLOR_NAMES[col_name]: 1.0})

        boxes: list[tuple] = []
        for c in comps:
            if c["lab"] in used or c["touches"]:
                continue
            sl, bh, bw, a, fill = c["sl"], c["bh"], c["bw"], c["area"], c["fill"]
            cy, cx = (sl[0].start + sl[0].stop) / 2, (sl[1].start + sl[1].stop) / 2
            aspect = bw / bh
            name = COLOR_NAMES[c["color"]]
            if a >= 0.04 * win_area and fill >= 0.6 and bw >= 0.15 * w and 0.5 <= aspect <= 4.5:
                objects.append({"type": "modal", "color": name})
                _assign_region(W, sl, h, w, rows, cols, {"modal": 1.2, name: 1.0})
            elif 4 <= bh <= 6 and 0.0005 * win_area <= a <= 0.01 * win_area and fill >= 0.7 and 1.8 <= aspect <= 6:
                objects.append({"type": "badge", "color": name})
                _assign_point(W, cy, cx, h, w, rows, cols, {"badge": 1.2, name: 1.0})
            elif bh >= 7 and 0.0008 * win_area <= a <= 0.03 * win_area and fill >= 0.75 and 0.8 <= aspect <= 4.5:
                boxes.append((c["color"], cy, cx, bh, bw))
        by_color: dict[int, list] = {}
        for b in boxes:
            by_color.setdefault(b[0], []).append(b)
        for col_name, bs in by_color.items():
            if len(bs) < 3:
                continue
            level_x: dict[int, set] = {}
            for b in bs:
                level_x.setdefault(round(b[1] / max(b[3], 1)), set()).add(round(b[2] / max(b[4], 1)))
            if len(level_x) >= 2 and max(len(v) for v in level_x.values()) >= 2:
                objects.append({"type": "tree", "color": COLOR_NAMES[col_name], "boxes": len(bs)})
                for b in bs:
                    _assign_point(W, b[1], b[2], h, w, rows, cols,
                                  {"tree": 1.0, "diagram": 1.0, COLOR_NAMES[col_name]: 1.0})
    return W, objects


def _bind(concepts: dict[str, float]) -> dict[str, float]:
    """Add colour x shape compound concepts for an object's own colour and shapes."""
    colors = [c for c in concepts if c in _COLOR_SET]
    out = dict(concepts)
    for c in colors:
        for sh, v in concepts.items():
            if sh in _SHAPE_SET:
                out[f"{c}_{sh}"] = v
    return out


def _assign_mask(W, mask, sl, pmap, h, w, rows, cols, concepts: dict[str, float], min_share: float = 0.08) -> None:
    """Assign concepts to every patch holding >= min_share of the object's pixels."""
    concepts = {k: v for k, v in _bind(concepts).items() if k in _VC_INDEX}
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return
    pids = pmap.reshape(h, w)[ys + sl[0].start, xs + sl[1].start]
    share = np.bincount(pids, minlength=rows * cols) / len(ys)
    for p in np.flatnonzero(share >= min_share):
        for name, val in concepts.items():
            j = _VC_INDEX[name]
            W[p, j] = max(W[p, j], val)


def _assign_point(W, cy, cx, h, w, rows, cols, concepts: dict[str, float]) -> None:
    concepts = {k: v for k, v in _bind(concepts).items() if k in _VC_INDEX}
    p = min(int(cy * rows / h), rows - 1) * cols + min(int(cx * cols / w), cols - 1)
    for name, val in concepts.items():
        j = _VC_INDEX[name]
        W[p, j] = max(W[p, j], val)


def _assign_region(W, sl, h, w, rows, cols, concepts: dict[str, float]) -> None:
    concepts = {k: v for k, v in _bind(concepts).items() if k in _VC_INDEX}
    ph, pw = h / rows, w / cols
    for r in range(rows):
        for c in range(cols):
            oy = max(0.0, min(sl[0].stop, (r + 1) * ph) - max(sl[0].start, r * ph))
            ox = max(0.0, min(sl[1].stop, (c + 1) * pw) - max(sl[1].start, c * pw))
            cover = (oy * ox) / (ph * pw)
            if cover >= 0.25:
                p = r * cols + c
                for name, val in concepts.items():
                    j = _VC_INDEX[name]
                    W[p, j] = max(W[p, j], val * min(1.0, cover * 1.5))


