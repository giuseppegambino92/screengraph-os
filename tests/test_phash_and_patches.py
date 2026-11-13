"""2D-DCT pHash idle gating, 1-bit BQ, Matryoshka slicing and Patch-MaxSim spatial localization."""

from __future__ import annotations

import time

import numpy as np
import pytest
from scipy.fft import dct

from screengraph.capture.phash import (PerceptualGate, ax_tree_hash, dct_matrix, dhash64, hamming64, masked_hamming,
                                       perceptual_signature, phash64)
from screengraph.capture.simulator import WorkspaceSimulator
from screengraph.embeddings.maxsim import maxsim, parse_spatial, PatchIndex
from screengraph.embeddings.patch_encoder import (PatchEncoder, analyze_visual, binary_quantize, hamming,
                                                  hamming_to_cosine, l2n, matryoshka)


def test_dct_matrix_is_orthonormal_and_matches_scipy():
    c = dct_matrix(32).astype(np.float64)
    assert np.allclose(c @ c.T, np.eye(32), atol=1e-5)
    x = np.random.default_rng(0).random((32, 32))
    ref = dct(dct(x, axis=0, norm="ortho"), axis=1, norm="ortho")
    assert np.allclose(c @ x @ c.T, ref, atol=1e-4)


def test_phash_is_stable_and_discriminative():
    rng = np.random.default_rng(1)
    a = (rng.random((180, 160, 3)) * 255).astype(np.uint8)
    b = a.copy()
    b[50:52, 60:62] = 255  # a few pixels
    c = (rng.random((180, 160, 3)) * 255).astype(np.uint8)
    assert hamming64(phash64(a), phash64(a)) == 0
    assert hamming64(phash64(a), phash64(b)) <= 4
    assert hamming64(phash64(a), phash64(c)) > 16
    assert hamming64(dhash64(a), dhash64(c)) > 10


def _sim_frame(wid="dash"):
    sim = WorkspaceSimulator(seed=3)
    for tk in sim.session(changes_only=True):
        if tk.index > 6:
            break
    return sim, sim.windows[wid]


def test_gate_drops_cursor_and_caret_ticks_but_passes_real_changes():
    sim, term = _sim_frame("term")
    gate = PerceptualGate(4, 6, quadrant_threshold=2)
    px, nodes, _ = term.render()
    ax = ax_tree_hash(term.title, term.url, term.doc_path, nodes)
    assert gate.check("t", ax, px.copy()).passed  # first frame
    cur = px.copy()
    cur[40:43, 70:72] = 255  # mouse cursor
    cur[-6:-3, 12:13] = 230  # caret
    d = gate.check("t", ax, cur)
    assert not d.passed and d.reason == "idle_perceptual_match"
    term.run("pytest -x", [("FAILED tests/test_pool.py", (230, 80, 80))] * 6)
    px2, nodes2, _ = term.render()
    assert gate.check("t", ax_tree_hash(term.title, "", "", nodes2), px2).reason == "ax_tree_changed"


def test_gate_catches_visual_only_spike_via_quadrant_hash():
    sim, dash = _sim_frame("dash")
    gate = PerceptualGate(4, 6, quadrant_threshold=2)
    px, nodes, _ = dash.render()
    ax = ax_tree_hash(dash.title, dash.url, "", nodes)
    gate.check("d", ax, px)
    dash.spike_at = 0.72  # canvas-only change: AX tree identical
    px2, nodes2, _ = dash.render()
    assert ax_tree_hash(dash.title, dash.url, "", nodes2) == ax
    d = gate.check("d", ax, px2)
    assert d.passed and d.reason == "visual_change"
    a, b = perceptual_signature(px), perceptual_signature(px2)
    assert masked_hamming(a[0], b[0]) <= 4  # global pHash alone would have missed it


def test_gate_drop_latency_budget():
    sim, term = _sim_frame("term")
    gate = PerceptualGate()
    px, nodes, _ = term.render()
    ax = ax_tree_hash(term.title, "", "", nodes)
    gate.check("t", ax, px)
    times = []
    for _ in range(300):
        t0 = time.perf_counter()
        d = gate.check("t", ax, px)
        times.append((time.perf_counter() - t0) * 1e3)
        assert not d.passed
    assert np.percentile(times, 50) < 0.5  # ~0.15 ms on a laptop; loose bound for CI


def test_session_gate_drops_most_idle_and_misses_no_semantic_change(session_run):
    ticks = session_run["ticks"]
    idle = [t for t in ticks if t["tick"].kind == "idle"]
    change = [t for t in ticks if t["tick"].kind == "change"]
    dropped = sum(not t["result"].stored for t in ticks)
    assert dropped / len(ticks) > 0.80
    assert all(t["result"].stored for t in change)
    assert sum(not t["result"].stored for t in idle) / len(idle) > 0.95


# ------------------------------------------------------------ BQ / Matryoshka
def test_binary_quantization_packs_signs_and_hamming_tracks_cosine():
    rng = np.random.default_rng(0)
    v = rng.standard_normal((2000, 256)).astype(np.float32)
    bits = binary_quantize(v)
    assert bits.shape == (2000, 4) and bits.dtype == np.uint64
    first = int(bits[0, 0])
    assert all(((first >> i) & 1) == (v[0, i] > 0) for i in range(64))
    q = l2n(rng.standard_normal(256).astype(np.float32))
    t = rng.uniform(-1, 1, (2000, 1)).astype(np.float32)  # docs spanning cosines in [-1, 1]
    docs = l2n(t * q + np.sqrt(1 - t ** 2) * l2n(rng.standard_normal((2000, 256)).astype(np.float32)))
    cos = docs @ q
    est = hamming_to_cosine(hamming(binary_quantize(docs), binary_quantize(q)[None, :]), 256)
    assert np.corrcoef(cos, est)[0, 1] > 0.95
    assert np.abs(cos - est).mean() < 0.1


def test_matryoshka_prefix_is_unit_norm_and_preserves_ranking():
    rng = np.random.default_rng(1)
    enc = PatchEncoder()
    docs = l2n(rng.standard_normal((200, 256)).astype(np.float32))
    q = l2n(docs[17] + 0.03 * rng.standard_normal(256).astype(np.float32))
    for d in (64, 128, 256):
        m = matryoshka(docs, d)
        assert m.shape[1] == d and np.allclose(np.linalg.norm(m, axis=1), 1, atol=1e-5)
        assert int(np.argmax(m @ matryoshka(q, d))) == 17
    assert enc.dim == 256


# ------------------------------------------------------------ MaxSim
def test_maxsim_localizes_the_patch_holding_the_concept():
    enc = PatchEncoder()
    rng = np.random.default_rng(2)
    P = 16
    patches = l2n(rng.standard_normal((5, P, 256)).astype(np.float32) * 0.2)
    target = enc.basis.vec("spike") + enc.basis.vec("red") + enc.basis.vec("chart")
    patches[3, 7] = l2n(target)
    _, Q = enc.encode_query_tokens("red spike chart")
    scores, best = maxsim(Q, patches)
    assert int(np.argmax(scores)) == 3 and int(best[3]) == 7


def test_parse_spatial_quadrant():
    q, w, label = parse_spatial("red memory spike chart in top-right quadrant", 4, 4)
    assert label == "top-right" and "quadrant" not in q and "top" not in q
    w = w.reshape(4, 4)
    assert w[0, 3] > 1 and w[1, 2] > 1 and w[3, 0] == 1 and w[0, 0] == 1


def test_visual_detectors_on_simulated_windows():
    sim = WorkspaceSimulator(seed=7)
    seen = {}
    for tk in sim.session(changes_only=True):
        if tk.artifacts:
            for a in tk.artifacts:
                seen.setdefault(a["name"], (tk.frame.pixels.copy(), a))
    for name, kind in [("red_spike", "spike"), ("blue_tree", "tree"), ("orange_modal", "modal"),
                       ("green_bars", "bar_chart"), ("red_badge", "badge"), ("cyan_line", "line_chart")]:
        px, art = seen[name]
        va = analyze_visual(px, 4, 4, ui_concepts=True)  # optional ablation layer
        kinds = {o["type"] for o in va.objects}
        assert kind in kinds, (name, kinds)


def test_patch_index_search_matches_exact_maxsim_and_returns_quadrant(session_run):
    sg = session_run["sg"]
    q, sp, _ = parse_spatial("red memory spike chart in top-right quadrant", 4, 4)
    _, Q = sg.encoder.encode_query_tokens(q)
    hits = sg.patches.search(Q, 3, sp)
    ev = sg.store.get_event(hits[0].event_id)
    assert ev["app_category"] == "dashboard"
    assert hits[0].quadrant == "top-right" and hits[0].col >= 2


def test_bq_prefilter_funnel_agrees_with_exact_search(session_run):
    sg = session_run["sg"]
    idx: PatchIndex = sg.patches
    _, Q = sg.encoder.encode_query_tokens("blue architecture tree diagram")
    exact = idx.search(Q, 1, candidates=10 ** 9)
    funnel = idx.search(Q, 1, candidates=16)  # forces the 64-bit Hamming prefilter to do real work
    assert exact[0].event_id == funnel[0].event_id


# ------------------------------------------------------------------ phase 2
def test_fused_signature_equals_reference_dcts():
    from screengraph.capture.phash import _robust_bits, area_downsample, _avg_matrix

    rng = np.random.default_rng(4)
    for shape in [(180, 160, 3), (130, 225, 3), (40, 40, 3)]:
        px = (rng.random(shape) * 255).astype(np.uint8)
        g2 = area_downsample(px, 64, 64)
        if g2.shape != (64, 64):
            pad = np.zeros((64, 64), np.float32)
            pad[: g2.shape[0], : g2.shape[1]] = g2
            g2 = pad
        c8 = dct_matrix(32)[:8]
        blocks = [_avg_matrix(64, 32) @ g2 @ _avg_matrix(64, 32).T] + \
                 [g2[r * 32:(r + 1) * 32, k * 32:(k + 1) * 32] for r in (0, 1) for k in (0, 1)]
        ref = [_robust_bits((c8 @ b @ c8.T).ravel()) for b in blocks]
        got = perceptual_signature(px)
        assert masked_hamming(got[0], ref[0]) == 0
        assert all(masked_hamming(a, b) == 0 for a, b in zip(got[2], ref[1:]))


def test_generic_encoder_any_shape_rgba_deterministic():
    enc = PatchEncoder()
    rng = np.random.default_rng(5)
    for shape in [(1, 1, 3), (15, 15, 3), (40, 300, 3), (130, 225, 4), (400, 700, 3)]:
        px = (rng.random(shape) * 255).astype(np.uint8)
        a = enc.encode(px.copy(), [("hello world", (0, 0, 10, 3))])
        b = enc.encode(px.copy(), [("hello world", (0, 0, 10, 3))])
        assert a.patch_vecs.shape == (16, 256) and np.isfinite(a.patch_vecs).all()
        assert np.allclose(np.linalg.norm(a.patch_vecs, axis=1), 1, atol=1e-5)
        assert np.array_equal(a.patch_vecs, b.patch_vecs)


def test_descriptor_projection_is_orthogonal_to_colour_words():
    from screengraph.embeddings.patch_encoder import FEATURE_DIM, VISUAL_CONCEPTS

    enc = PatchEncoder()
    f = np.random.default_rng(6).random((50, FEATURE_DIM)).astype(np.float32)
    d = l2n(f @ enc.R)
    colours = enc.basis.matrix(list(VISUAL_CONCEPTS))
    assert np.abs(d @ colours.T).max() < 1e-4  # texture/spectrum energy never reads as "red"


def test_int8_quantization_preserves_maxsim_ranking_and_localization():
    from screengraph.embeddings.patch_encoder import dequantize_int8, quantize_int8

    rng = np.random.default_rng(7)
    docs = l2n(rng.standard_normal((300, 16, 128)).astype(np.float32))
    q8 = [quantize_int8(d) for d in docs]
    deq = np.stack([dequantize_int8(q, s) for q, s in q8])
    assert np.abs(deq - docs).max() < 0.01
    agree = 0
    for _ in range(100):
        tgt = int(rng.integers(300)), int(rng.integers(16))
        Q = l2n(docs[tgt[0], tgt[1]][None] + 0.05 * rng.standard_normal((3, 128)).astype(np.float32))
        sf, bf = maxsim(Q, docs)
        si, bi = maxsim(Q, deq)
        agree += int(np.argmax(sf) == np.argmax(si) and bf[np.argmax(sf)] == bi[np.argmax(si)])
    assert agree == 100


def test_patch_index_stores_int8(session_run):
    idx = session_run["sg"].patches
    assert idx._q.dtype == np.int8 and idx._q.shape[1:] == (16, 128)
    blob = session_run["sg"].store.conn.execute("SELECT length(patch_i8) FROM event_vectors LIMIT 1").fetchone()[0]
    assert blob == 16 * 128


def test_query_by_example_finds_same_app(session_run):
    sg = session_run["sg"]
    dash = [e for e in sg.store.timeline(0, 1e13, 10000) if e["app_category"] == "dashboard"]
    res = sg.visual_search_like(dash[0]["id"], k=3)
    assert res and all(sg.store.get_event(r["event_id"])["app_category"] == "dashboard" for r in res)


def test_vision_adapter_plumbing():
    from screengraph.embeddings.patch_encoder import VisionAdapter

    class Fake(VisionAdapter):
        name, dim = "fake", 32

        def encode_patches(self, pixels, rows, cols):
            return l2n(np.ones((rows * cols, 32), np.float32) + np.arange(32, dtype=np.float32))

        def encode_text(self, texts):
            return l2n(np.ones((len(texts), 32), np.float32) + np.arange(32, dtype=np.float32))

    enc = PatchEncoder(adapter=Fake())
    toks, Q = enc.encode_query_tokens("red chart")
    assert toks[-1] == "<fake>" and Q.shape == (4, 256)  # red, chart, red_chart (binding), adapter
    e = enc.encode(np.zeros((40, 40, 3), np.uint8), [])
    s_with, _ = maxsim(Q, e.patch_vecs[None])
    s_without, _ = maxsim(Q[:-1], e.patch_vecs[None])
    assert s_with[0] > s_without[0]  # the adapter's text embedding matches its own image embedding
