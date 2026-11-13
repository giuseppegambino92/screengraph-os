"""Cross-modal morphological concept bank: detection, resolution independence, hard negatives,
held-out generalization floor, privacy wiping, and the pure-visual anomaly causal bridge."""

from __future__ import annotations

import numpy as np
import pytest

from screengraph.capture.ephemeral import EphemeralFrameBuffer, ZeroingLedger
from screengraph.capture.models import BoundingBox, ViewportFrame
from screengraph.capture.simulator import DashboardWindow, WorkspaceSimulator
from screengraph.embeddings.morphology import MORPH_CONCEPTS
from screengraph.embeddings.patch_encoder import PatchEncoder, analyze_visual, l2n

KINDS = [("red_spike", "spike"), ("blue_tree", "tree"), ("orange_modal", "modal"),
         ("green_bars", "bar_chart"), ("red_badge", "badge"), ("cyan_line", "line_chart")]


@pytest.fixture(scope="module")
def artifact_frames():
    sim = WorkspaceSimulator(seed=7)
    seen = {}
    for tk in sim.session(changes_only=True):
        for a in tk.artifacts:
            seen.setdefault(a["name"], (tk.frame.pixels.copy(), a))
    return seen


def test_default_layer_detects_every_artifact_kind(artifact_frames):
    for name, kind in KINDS:
        px, art = artifact_frames[name]
        objs = analyze_visual(px, 4, 4).objects  # default: morphology ON, ui_concepts OFF
        assert kind in {o["type"] for o in objs}, name
        assert not any(o.get("source") == "ui_concepts" for o in objs)


def test_spike_lands_in_a_gold_patch(artifact_frames):
    px, art = artifact_frames["red_spike"]
    va = analyze_visual(px, 4, 4)
    j = MORPH_CONCEPTS.index("spike")
    hot = {divmod(int(p), 4) for p in np.flatnonzero(va.morph_weights[:, j] >= 0.5)}
    assert hot & {tuple(p) for p in art["patches"]}


@pytest.mark.parametrize("scale", [2, 3])
def test_resolution_independence(artifact_frames, scale):
    for name, kind in KINDS:
        px, _ = artifact_frames[name]
        up = np.repeat(np.repeat(px, scale, 0), scale, 1)
        assert kind in {o["type"] for o in analyze_visual(up, 4, 4).objects}, (name, scale)


def test_hard_negatives_from_independent_renderer():
    from evals.morph_heldout import CASES, Scene

    rng = np.random.default_rng(2024)
    for case, forbidden in [("volatile_no_spike", "spike"), ("line_no_spike", "spike"),
                            ("side_panel", "modal"), ("unlabelled_squares", "badge"),
                            ("boxes_no_connectors", "tree"), ("card_grid", "tree")]:
        for _ in range(6):
            sc = Scene(rng)
            sc.distractor_text()
            CASES[case][0](sc)
            assert forbidden not in {o["type"] for o in analyze_visual(sc.finish(False, 2.0, 1), 4, 4).objects}, case


def test_heldout_generalization_floor():
    """Independent renderer (random sizes/hues/strokes/blur/noise/2x). Floors sit below the measured
    test-split numbers in the README; the spike false-positive rate must stay exactly zero because a
    spike sets the causal anomaly flag."""
    from evals.morph_heldout import run

    r = run(n=12, seed=98765)["per_concept"]
    assert r["spike"]["false_positive_rate"] == 0.0
    assert r["modal"]["recall"] >= 0.9 and r["tree"]["recall"] >= 0.7 and r["bar_chart"]["recall"] >= 0.6
    assert r["spike"]["recall"] >= 0.4 and r["badge"]["recall"] >= 0.6


def test_text_queries_bind_colour_to_the_detected_object(artifact_frames):
    enc = PatchEncoder()
    px, _ = artifact_frames["red_badge"]
    toks, Q = enc.encode_query_tokens("red failing badge")
    assert "red_badge" in toks
    w = enc.encode(px.copy(), [])
    j = toks.index("red_badge")
    assert float((w.patch_vecs @ Q[j]).max()) > 0.25  # the bound concept is present in some patch
    _, Qg = enc.encode_query_tokens("green badge")
    assert float((w.patch_vecs @ Qg[-1]).max()) > 0.25  # the passing (green) badges bind too


def test_morph_word_subspace_is_orthogonal_to_the_descriptor_projection():
    from screengraph.embeddings.patch_encoder import FEATURE_DIM

    enc = PatchEncoder()
    f = np.random.default_rng(3).random((40, FEATURE_DIM)).astype(np.float32)
    words = enc.basis.matrix(["spike", "tree", "badge", "modal", "red_spike", "table"])
    assert np.abs(l2n(f @ enc.R) @ words.T).max() < 1e-4


def test_morphology_wipes_its_pixel_derived_intermediates(artifact_frames):
    px, _ = artifact_frames["blue_tree"]
    ledger = ZeroingLedger()
    frame = px.copy()
    with EphemeralFrameBuffer(frame, ledger) as buf:
        va = analyze_visual(frame, 4, 4, buf=buf)
    assert not frame.any() and "tree" in {o["type"] for o in va.objects}
    snap = ledger.snapshot()
    assert snap["verification_failures"] == 0 and snap["buffers_zeroed"] >= 5  # frame + tracked intermediates


# ------------------------------------------------------------------ causal: pure-visual anomaly bridge
T0 = 1_731_600_000.0


class _Dash(DashboardWindow):
    def __init__(self, captions, seed=5):
        self.PANELS = [(captions[i], col, kind) for i, (_, col, kind) in enumerate(DashboardWindow.PANELS)]
        super().__init__("dash", BoundingBox(0, 0, 1280, 1440, 1), pid=1, seed=seed)


def _dash_frame(d: _Dash, ts: float) -> ViewportFrame:
    px, nodes, _ = d.render()
    return ViewportFrame(window_id="dash", app="Grafana", title=d.title, bbox=d.bbox, ts=ts, app_category="dashboard",
                         url=d.url, ax_nodes=nodes, pixels=px, source="sim")


def _run(engine, caption: str, terminal_text: str):
    d = _Dash(["requests/s · gateway", caption, "CI build duration (min)", "p95 latency (ms) · checkout"])
    calm = engine.ingest_frame(_dash_frame(d, T0))
    # a dashboard refresh redraws the series and the spike appears in it. (A spike-only redraw with no
    # other pixel change passes the perceptual gate in only ~46% of positions: see README limitations.)
    d.refresh()
    d.spike_at = 0.6
    spike = engine.ingest_frame(_dash_frame(d, T0 + 30))
    term = engine.ingest_text_event("iTerm2", "terminal", terminal_text, window_id="term", ts=T0 + 70)
    fix = engine.ingest_text_event("Code", "ide", "limits: { memory: 2Gi }", doc_path="k8s/orders.yaml",
                                   window_id="ide", ts=T0 + 120)
    return calm, spike, term, fix


def test_visual_anomaly_bridge_links_a_pixel_only_spike_to_the_log(engine):
    calm, spike, term, fix = _run(engine, "orders-svc memory (GiB)",
                                  "kubectl get pods\norders-svc-6f7c-zz1   0/1   OOMKilled   2   9m\nsee k8s/orders.yaml")
    assert spike.stored and calm.stored
    ev = engine.store.get_event(spike.event_id)["entities"]
    assert ev["visual_anomalies"][0]["caption"] == "orders-svc memory (GiB)"  # caption association
    e = [x for x in term.edges if x.src == spike.event_id]
    assert e and e[0].relation == "VISUAL_ANOMALY_CORRELATION"
    assert any("xmodal_cos=" in s for s in e[0].evidence) and "visual_anomaly:orders-svc" in e[0].evidence
    assert not any(x.src == calm.event_id and x.relation == "VISUAL_ANOMALY_CORRELATION" for x in term.edges)
    res = engine.why("k8s/orders.yaml")
    assert res["root_causes"][0]["event_id"] == spike.event_id


def test_visual_anomaly_bridge_requires_the_same_subsystem(engine):
    _, spike, term, _ = _run(engine, "billing-api memory (GiB)",
                             "kubectl get pods\norders-svc-6f7c-zz1   0/1   OOMKilled   2   9m")
    assert not any(x.src == spike.event_id and x.relation == "VISUAL_ANOMALY_CORRELATION" for x in term.edges)


def test_persisting_spike_is_not_new_evidence(engine):
    d = _Dash(["requests/s · gateway", "orders-svc memory (GiB)", "CI build duration (min)", "p95 latency"])
    d.spike_at = 0.6
    first = engine.ingest_frame(_dash_frame(d, T0))
    d.refresh()  # new data, spike still visible (scrolled left)
    again = engine.ingest_frame(_dash_frame(d, T0 + 30))
    term = engine.ingest_text_event("iTerm2", "terminal", "orders-svc-6f7c-zz1 0/1 OOMKilled", window_id="t",
                                    ts=T0 + 60)
    assert again.stored
    srcs = {x.src for x in term.edges if x.relation == "VISUAL_ANOMALY_CORRELATION"}
    assert first.event_id in srcs and again.event_id not in srcs
