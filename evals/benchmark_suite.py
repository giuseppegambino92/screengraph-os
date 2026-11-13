"""Deterministic benchmark: screengraph-os vs. flat timestamp/vector and single-vector baselines.

Run:  python -m evals.benchmark_suite [--scale 10000] [--seed 7] [--out evals/latest_report.json]

Everything runs offline against the deterministic workspace simulator. Numbers
for "legacy" continuous-capture storage are a *parametric model* with the
assumptions printed alongside (we do not run third-party tools here); all
screengraph-os numbers are measured.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import sqlite3
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from screengraph import __version__
from screengraph.capture.simulator import WorkspaceSimulator
from screengraph.config import Config
from screengraph.embeddings.maxsim import global_average_scores, hamming_maxsim, maxsim, parse_spatial
from screengraph.embeddings.patch_encoder import binary_quantize, l2n
from screengraph.db.store import Store
from screengraph.engine import ScreenGraph
from screengraph.graph.retriever import rrf

ROOT_ROLES = {"C1": {"root", "copy"}, "C2": {"root", "copy"}, "C3": {"root", "copy"}, "C4": {"root"}}

# Legacy continuous-capture storage model (assumptions, not measurements of any specific product)
LEGACY_MODEL = {
    "capture_fps": 0.5,
    "video_bytes_per_frame": 100_000,  # H.264 across 5120x1440 + 2560x1440, mostly-static screens
    "embedding_bytes_per_frame": 512 * 4,  # one float32 CLIP-style 512-d vector per frame
    "ocr_text_bytes_per_frame": 2_000,
    "hours_per_day": 8,
    "days": 30,
}


def pct(a, q):
    return float(np.percentile(np.asarray(a, dtype=np.float64), q)) if len(a) else None


def lat(a) -> dict:
    return {"n": len(a), "p50_ms": round(pct(a, 50), 4), "p95_ms": round(pct(a, 95), 4),
            "p99_ms": round(pct(a, 99), 4), "max_ms": round(float(np.max(a)), 4)} if len(a) else {"n": 0}


def f1(pred: set, gold: set) -> tuple[float, float, float]:
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


# ---------------------------------------------------------------------------
def run_session(sg: ScreenGraph, sim: WorkspaceSimulator) -> dict:
    """Ingest the session. Ground truth: an idle tick that passes the gate (pixel noise) re-observes
    the window's current semantic state, so it inherits that state's chain/role label."""
    ticks = []
    state: dict[str, object] = {}
    for tk in sim.session():
        r = sg.ingest_frame(tk.frame)
        if tk.action is not None:
            state[tk.frame.window_id] = tk.action
        lab = tk.action or state.get(tk.frame.window_id)
        ticks.append({
            "index": tk.index, "kind": tk.kind, "window": tk.frame.window_id, "stored": r.stored,
            "reason": r.reason, "event_id": r.event_id, "timings": r.timings_ms,
            "chain": lab.chain if lab else None, "role": lab.role if lab else None,
            "label": tk.action.label if tk.action else "", "artifacts": tk.artifacts,
            "pixels_zero_after": (not tk.frame.pixels.any()) if tk.frame.pixels is not None else True,
            "copy": tk.action.copy if tk.action else None,
        })
    return {"ticks": ticks}


def empty_db_bytes() -> int:
    """Size of a freshly created database (schema pages only), checkpointed."""
    d = Path(tempfile.mkdtemp(prefix="sg-empty-"))
    try:
        st = Store(d / "e.db")
        n = st.size_bytes(checkpoint=True)
        st.close()
        return n
    finally:
        shutil.rmtree(d, ignore_errors=True)


def gate_and_storage(sg: ScreenGraph, ticks: list[dict], session_hours: float, data_dir: Path) -> dict:
    kinds = Counter((t["kind"], t["stored"]) for t in ticks)
    n = len(ticks)
    dropped = sum(1 for t in ticks if not t["stored"])
    idle = sum(1 for t in ticks if t["kind"] == "idle")
    change = n - idle
    drop_ms = [t["timings"]["gate_ms"] for t in ticks if not t["stored"]]
    sg.store.set_audit({"ledger": sg.ledger.snapshot(), "counters": sg.counters})
    db_bytes = sg.store.size_bytes(checkpoint=True)
    hours_month = LEGACY_MODEL["hours_per_day"] * LEGACY_MODEL["days"]
    # Projection: the schema's fixed pages exist once; only the per-hour growth scales with time.
    # (Phase 1 multiplied the whole file, fixed pages included, by 240 h; reported as *_naive for comparison.)
    base = empty_db_bytes()
    sg_month_mb = (base + max(0, db_bytes - base) / session_hours * hours_month) / 1e6
    naive_month_mb = db_bytes / session_hours * hours_month / 1e6
    frames_month = LEGACY_MODEL["capture_fps"] * 3600 * hours_month
    per_frame = (LEGACY_MODEL["video_bytes_per_frame"] + LEGACY_MODEL["embedding_bytes_per_frame"]
                 + LEGACY_MODEL["ocr_text_bytes_per_frame"])
    legacy_month_mb = frames_month * per_frame / 1e6
    audit = sg.privacy_audit()
    return {
        "ticks": n, "change_ticks": change, "idle_ticks": idle,
        "dropped": dropped, "drop_rate": round(dropped / n, 4),
        "idle_drop_rate": round(kinds[("idle", False)] / max(idle, 1), 4),
        "semantic_changes_missed": kinds[("change", False)],
        "idle_false_passes": kinds[("idle", True)],
        "gate_latency_dropped_frames": lat(drop_ms),
        "stored_events": sum(1 for t in ticks if t["stored"]),
        "db_bytes_session": db_bytes,
        "db_bytes_per_stored_event": round(db_bytes / max(1, sum(1 for t in ticks if t["stored"])), 1),
        "session_hours": round(session_hours, 4),
        "projected_30d_8h_mb": round(sg_month_mb, 2),
        "projected_30d_8h_mb_naive": round(naive_month_mb, 2),
        "empty_schema_bytes": base,
        "growth_bytes_per_stored_event": round((db_bytes - base) / max(1, sum(1 for t in ticks if t["stored"])), 1),
        "legacy_model_30d_8h_mb": round(legacy_month_mb, 1),
        "legacy_model_assumptions": LEGACY_MODEL,
        "storage_reduction_pct": round(100 * (1 - sg_month_mb / legacy_month_mb), 3),
        "zero_pixel_audit": {
            "image_files_on_disk": audit["disk"]["image_magic_hits"] + audit["disk"]["image_extension_hits"],
            "files_scanned": audit["disk"]["files_scanned"],
            "pixel_blob_suspects": audit["sqlite"]["pixel_blob_suspects"],
            "max_blob_bytes": audit["sqlite"]["max_blob_bytes"],
            "ram_buffers_zeroed": audit["ram"]["buffers_zeroed"],
            "ram_bytes_zeroed": audit["ram"]["bytes_zeroed"],
            "wipe_verification_failures": audit["ram"]["verification_failures"],
            "frames_with_nonzero_pixels_after_ingest": sum(1 for t in ticks if not t["pixels_zero_after"]),
            "ledger_chain_head": audit["ram"]["chain_head"],
            "guarantee_holds": audit["zero_pixel_guarantee_holds"],
        },
    }


# ---------------------------------------------------------------------------
def causal_eval(sg: ScreenGraph, ticks: list[dict]) -> dict:
    chains: dict[str, list[dict]] = defaultdict(list)
    for t in ticks:
        if t["chain"] and t["stored"]:
            chains[t["chain"]].append(t)
    all_events = sg.store.timeline(0, 1e13, limit=100000)
    per = {}
    for cid, path in WorkspaceSimulator.CHAIN_TARGETS.items():
        nodes = chains[cid]
        fix = [t["event_id"] for t in nodes if t["role"] == "fix"]
        gold_roots = {t["event_id"] for t in nodes if t["role"] in ROOT_ROLES[cid]}
        gold_chain = {t["event_id"] for t in nodes if t["role"] != "fix"}
        target = fix[-1] if fix else None

        # --- screengraph-os
        t0 = time.perf_counter()
        why = sg.why(path)
        why_ms = (time.perf_counter() - t0) * 1e3
        pred_root = why["root_causes"][0]["event_id"] if why.get("root_causes") else None
        pred_chain = {c["event_id"] for c in why.get("chain", [])} - {target}
        p, r, f = f1(pred_chain, gold_chain)

        # --- baseline A: flat timestamp ("what happened right before the edit?")
        tev = sg.store.get_event(target)
        before = [e for e in all_events if e["ts"] < tev["ts"] and e["window_id"] != tev["window_id"]]
        k = len(gold_chain)
        ta_root = before[-1]["id"] if before else None
        ta_chain = {e["id"] for e in before[-k:]}
        tp_, tr_, tf_ = f1(ta_chain, gold_chain)

        # --- baseline B: flat hybrid retrieval (BM25 + dense, no graph) on the file path + edited code
        # (unpruned BM25 + dense cosine, fused by RRF; same-window and future events excluded)
        query = f"{path} " + " ".join((tev.get("body") or "").split()[:40])
        lists = {"bm25": sg.store.fts_search(query, 100, max_df=1.0),
                 "dense": sg.dense.search(sg.encoder.encode_text_query(query), 100)}
        fused, _ = rrf(lists)
        evs = sg.store.get_events(fused)
        ranked = [e for e, _ in sorted(fused.items(), key=lambda x: -x[1])
                  if e in evs and evs[e]["window_id"] != tev["window_id"] and evs[e]["ts"] <= tev["ts"]][:k]
        vb_root = ranked[0] if ranked else None
        vb_chain = set(ranked)
        vp_, vr_, vf_ = f1(vb_chain, gold_chain)

        per[cid] = {
            "target": path, "target_event": target, "gold_roots": sorted(gold_roots), "gold_chain": sorted(gold_chain),
            "screengraph": {"root": pred_root, "root_hit": pred_root in gold_roots, "chain": sorted(pred_chain),
                            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
                            "latency_ms": round(why_ms, 3)},
            "baseline_timestamp": {"root": ta_root, "root_hit": ta_root in gold_roots, "f1": round(tf_, 4)},
            "baseline_flat_retrieval": {"root": vb_root, "root_hit": vb_root in gold_roots, "f1": round(vf_, 4)},
        }

    def mean(key, field):
        return round(float(np.mean([float(v[key][field]) for v in per.values()])), 4)

    # secret linkability: same keyed tag in terminal + chat, connected by a causal edge
    tags = {}
    for e in all_events:
        for tag in e["entities"].get("secrets", []):
            tags.setdefault(tag, set()).add((e["id"], e["app_category"]))
    linked = []
    for tag, evs in tags.items():
        cats = {c for _, c in evs}
        if {"terminal", "chat"} <= cats:
            ids = {i for i, _ in evs}
            edges = [ed for ed in sg.store.edges() if ed["src"] in ids and ed["dst"] in ids and ed["c_bridge"] > 0]
            linked.append({"tag": tag, "events": sorted(ids), "edges": [(ed["src"], ed["dst"], ed["relation"]) for ed in edges]})
    rel = Counter(ed["relation"] for ed in sg.store.edges())
    return {
        "chains": per,
        "summary": {
            "screengraph_root_recall_at_1": mean("screengraph", "root_hit"),
            "screengraph_chain_f1": mean("screengraph", "f1"),
            "screengraph_chain_precision": mean("screengraph", "precision"),
            "screengraph_chain_recall": mean("screengraph", "recall"),
            "timestamp_baseline_root_recall_at_1": mean("baseline_timestamp", "root_hit"),
            "timestamp_baseline_chain_f1": mean("baseline_timestamp", "f1"),
            "flat_retrieval_root_recall_at_1": mean("baseline_flat_retrieval", "root_hit"),
            "flat_retrieval_chain_f1": mean("baseline_flat_retrieval", "f1"),
        },
        "secret_linkability": {"linked_tags": linked,
                               "cross_app_secret_linked": any(l["edges"] for l in linked)},
        "edge_relations": dict(rel),
    }


# ---------------------------------------------------------------------------
def visual_eval(sg: ScreenGraph, ticks: list[dict]) -> dict:
    gold: dict[str, dict[int, set]] = defaultdict(dict)
    for t in ticks:
        if not t["stored"]:
            continue
        for a in t["artifacts"]:
            gold[a["name"]][t["event_id"]] = {tuple(p) for p in a["patches"]}
    idx = sg.patches
    allpos = np.arange(idx.n)
    P_all = idx.patches(allpos)  # (N, P, 128)
    ids = idx._ids[: idx.n]
    rows = {}
    agg = defaultdict(list)
    for name, query in WorkspaceSimulator.VISUAL_QUERIES.items():
        g = gold[name]
        q_clean, spatial, _ = parse_spatial(query, idx.rows, idx.cols)
        toks, Q = sg.encoder.encode_query_tokens(q_clean)
        res = {"query": query, "gold_events": len(g)}
        # patch maxsim (float16 128-d rescoring, the production path)
        hits = idx.search(Q, 3, spatial)
        top = hits[0]
        loc_ok = top.event_id in g and (top.row, top.col) in g[top.event_id]
        res["patch_maxsim"] = {"top_event": top.event_id, "recall_at_1": top.event_id in g,
                               "recall_at_3": any(h.event_id in g for h in hits),
                               "patch": [top.row, top.col], "quadrant_ok": bool(loc_ok)}
        # Matryoshka 64-d and pure 1-bit Hamming MaxSim
        for dim in (64,):
            s, _ = maxsim(l2n(Q[:, :dim]), l2n(P_all[:, :, :dim]), spatial)
            res[f"patch_maxsim_{dim}d"] = {"recall_at_1": int(ids[np.argmax(s)]) in g}
        bq = binary_quantize(l2n(P_all[:, :, :64]))
        hs = hamming_maxsim(binary_quantize(l2n(Q[:, :64])), bq, 64, spatial)
        res["patch_hamming_1bit_64d"] = {"recall_at_1": int(ids[np.argmax(hs)]) in g}
        # baseline: global window average vector (same encoder, same IDF weighting)
        gs = global_average_scores(l2n(Q[:, :128]), P_all)
        order = np.argsort(-gs)
        res["global_average"] = {"top_event": int(ids[order[0]]), "recall_at_1": int(ids[order[0]]) in g,
                                 "recall_at_3": any(int(ids[o]) in g for o in order[:3]), "quadrant_ok": False}
        rows[name] = res
        agg["patch_r1"].append(res["patch_maxsim"]["recall_at_1"])
        agg["patch_r3"].append(res["patch_maxsim"]["recall_at_3"])
        agg["patch_loc"].append(res["patch_maxsim"]["quadrant_ok"])
        agg["patch64_r1"].append(res["patch_maxsim_64d"]["recall_at_1"])
        agg["bq_r1"].append(res["patch_hamming_1bit_64d"]["recall_at_1"])
        agg["global_r1"].append(res["global_average"]["recall_at_1"])
        agg["global_r3"].append(res["global_average"]["recall_at_3"])
        # int8 vs unquantized float32 patches (same encoder): does SQ change the answer?
        if sg.float_shadow:
            F = np.stack([sg.float_shadow[int(e)] for e in ids])
            fs, fb = maxsim(l2n(Q[:, :128]), F, spatial)
            o = int(np.argmax(fs))
            fr, fc = divmod(int(fb[o]), idx.cols)
            same = int(ids[o]) == top.event_id and (fr, fc) == (top.row, top.col)
            f_r1 = int(ids[o]) in g
            f_loc = f_r1 and (fr, fc) in g[int(ids[o])]
            res["float32_reference"] = {"top_event": int(ids[o]), "patch": [fr, fc], "recall_at_1": f_r1,
                                        "quadrant_ok": bool(f_loc), "int8_identical_top1_and_patch": bool(same)}
            agg["int8_agrees"].append(same)
            agg["float_r1"].append(f_r1)
            agg["float_loc"].append(f_loc)
    m = {k: round(float(np.mean(v)), 4) for k, v in agg.items()}
    return {"queries": rows, "summary": {
        "patch_maxsim_recall_at_1": m["patch_r1"], "patch_maxsim_recall_at_3": m["patch_r3"],
        "patch_quadrant_localization_acc": m["patch_loc"],
        "patch_maxsim_64d_recall_at_1": m["patch64_r1"], "patch_hamming_1bit_recall_at_1": m["bq_r1"],
        "global_average_recall_at_1": m["global_r1"], "global_average_recall_at_3": m["global_r3"],
        "global_average_quadrant_localization_acc": 0.0,
        "int8_vs_float32_top1_and_patch_agreement": m.get("int8_agrees"),
        "float32_reference_recall_at_1": m.get("float_r1"),
        "float32_reference_localization_acc": m.get("float_loc")}}


# ---------------------------------------------------------------------------
def privacy_eval(sg: ScreenGraph, sim: WorkspaceSimulator, data_dir: Path) -> dict:
    sg.store.size_bytes(checkpoint=True)
    raw = b""
    for suffix in ("", "-wal"):
        p = Path(sg.store.path + suffix)
        if p.exists():
            raw += p.read_bytes()
    conn = sqlite3.connect(sg.store.path)
    conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS temp.v USING fts5vocab(main, 'events_fts', 'row')")
    vocab = [r[0] for r in conn.execute("SELECT term FROM temp.v")]
    text_cols = []
    for (t,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'events_fts%'"):
        for c in [r[1] for r in conn.execute(f"PRAGMA table_info({t})")]:
            text_cols += [str(v) for (v,) in conn.execute(f"SELECT {c} FROM {t} WHERE typeof({c})='text'")]
    conn.close()
    blob_text = "\n".join(text_cols)
    results = {}
    for name, secret in sim.secrets.items():
        fragments = [secret] if "\n" not in secret else [l for l in secret.split("\n") if "PRIVATE KEY" not in l]
        if name == "credit_card":
            fragments += [f"{secret[:4]} {secret[4:8]} {secret[8:12]} {secret[12:]}"]
        leaked_bytes = any(f.encode() in raw for f in fragments)
        leaked_text = any(f in blob_text for f in fragments)
        leaked_fts = any(len(v) >= 12 and any(v in f.lower() for f in fragments) for v in vocab)
        results[name] = {"redacted": not (leaked_bytes or leaked_text or leaked_fts),
                         "leak_db_bytes": leaked_bytes, "leak_text_columns": leaked_text, "leak_fts_vocab": leaked_fts}
    n_ok = sum(r["redacted"] for r in results.values())
    return {"secrets_injected": len(results), "secrets_redacted": n_ok,
            "redaction_recall": round(n_ok / len(results), 4), "per_secret": results,
            "redactions_by_kind": sg.store.redaction_counts(),
            "pii_redactions": {k: v for k, v in sg.store.redaction_counts().items() if k in ("email", "ipv4", "ipv6", "credit_card")}}


# ---------------------------------------------------------------------------
RETRIEVAL_QUERIES = [
    "SQLiteBusyException database is locked", "db_pool.py busy_timeout", "jwt expired clockTolerance",
    "red memory spike chart in top-right quadrant", "blue architecture tree diagram", "orange warning modal",
    "OOMKilled payments-api", "E0502 cannot borrow", "green bar chart of build durations",
    "what was marco saying about middleware", "kubernetes memory limits", "red failing badge",
]


def retrieval_latency(sg: ScreenGraph, repeats: int = 20) -> dict:
    for q in RETRIEVAL_QUERIES:  # warm-up
        sg.search(q, 10)
    times = []
    for _ in range(repeats):
        for q in RETRIEVAL_QUERIES:
            t0 = time.perf_counter()
            sg.search(q, 10)
            times.append((time.perf_counter() - t0) * 1e3)
    return lat(times)


def scale_eval(n_target: int, base: Path, seed: int) -> dict:
    d = base / "scale"
    sg = ScreenGraph(Config.ephemeral(d))
    ing, k = [], 0
    while sg.dense.n < n_target:
        sim = WorkspaceSimulator(seed=seed + 101 + k, t0=1_731_500_000.0 + k * 4000.0)
        for tk in sim.session(changes_only=True):
            r = sg.ingest_frame(tk.frame)
            if r.stored:
                ing.append(r.timings_ms["total_ms"])
            if sg.dense.n >= n_target:
                break
        k += 1
    db = sg.store.size_bytes(checkpoint=True)
    out = {"events": sg.dense.n, "db_mb": round(db / 1e6, 2), "ingest_total_per_stored_frame": lat(ing),
           "hybrid_retrieval": retrieval_latency(sg, repeats=10)}
    t = []
    for path in WorkspaceSimulator.CHAIN_TARGETS.values():
        t0 = time.perf_counter()
        sg.why(path)
        t.append((time.perf_counter() - t0) * 1e3)
    out["causal_why"] = lat(t)
    sg.close()
    return out


HELDOUT_TEST_SEED = 20261113


def ablation(base: Path, seed: int, name: str, note: str, **overrides) -> dict:
    """Same session with one encoder switch flipped."""
    sg = ScreenGraph(Config.ephemeral(base / f"ablation_{name}", **overrides))
    sim = WorkspaceSimulator(seed=seed)
    ticks = run_session(sg, sim)["ticks"]
    out = {"note": note, "config": overrides,
           "visual": visual_eval(sg, ticks)["summary"], "causal": causal_eval(sg, ticks)["summary"]}
    sg.close()
    return out


def ablation_ui_concepts(base: Path, seed: int) -> dict:
    return ablation(base, seed, "ui", "ui_concepts=True adds the hand-written chart/spike/bar/modal/badge/tree "
                    "detectors that were developed together with the simulator; never the default.", ui_concepts=True)


def ablation_morphology_off(base: Path, seed: int) -> dict:
    return ablation(base, seed, "nomorph", "morphology=False: the Phase-2 encoder (Spectral-HOG-JL descriptor + colour "
                    "names + AX tokens) without the morphological concept bank.", morphology=False)


def gate_spike_only_probe(seeds: int = 10, positions: int = 17) -> dict:
    """How often does the perceptual gate pass a frame whose ONLY change is a narrow chart spike
    (no data refresh, identical AX tree)? Frames the gate drops never reach the detectors."""
    from screengraph.capture.models import BoundingBox
    from screengraph.capture.phash import PerceptualGate, ax_tree_hash
    from screengraph.capture.simulator import DashboardWindow

    passed = total = 0
    for sd in range(seeds):
        for pos in np.linspace(0.12, 0.92, positions):
            d = DashboardWindow("dash", BoundingBox(0, 0, 1280, 1440, 1), pid=1, seed=sd)
            g = PerceptualGate(4, 6, quadrant_threshold=2)
            px, nodes, _ = d.render()
            ax = ax_tree_hash(d.title, d.url, "", nodes)
            g.check("d", ax, px)
            d.spike_at = float(pos)
            total += 1
            passed += g.check("d", ax, d.render()[0]).passed
    return {"spike_only_frames": total, "passed_gate": passed, "pass_rate": round(passed / total, 4),
            "note": "spike appearing with no other pixel or AX change; with a data refresh (the simulator's C4 "
                    "chain, typical dashboards) the frame passes as a visual change"}


def seed_sweep(seeds: list[int], base: Path) -> dict:
    """Same script, different seeds: cursor/caret noise, chart data, jitter and secrets all change."""
    rows = []
    for sd in seeds:
        sg = ScreenGraph(Config.ephemeral(base / f"seed{sd}"))
        sim = WorkspaceSimulator(seed=sd)
        ticks = run_session(sg, sim)["ticks"]
        g = gate_and_storage(sg, ticks, sim.duration / 3600, base)
        c = causal_eval(sg, ticks)["summary"]
        v = visual_eval(sg, ticks)["summary"]
        p = privacy_eval(sg, sim, base)
        rows.append({"seed": sd, "drop_rate": g["drop_rate"], "missed_changes": g["semantic_changes_missed"],
                     "root_recall": c["screengraph_root_recall_at_1"], "chain_f1": c["screengraph_chain_f1"],
                     "visual_r1": v["patch_maxsim_recall_at_1"], "global_r1": v["global_average_recall_at_1"],
                     "localization": v["patch_quadrant_localization_acc"], "redaction_recall": p["redaction_recall"],
                     "chain_precision": c["screengraph_chain_precision"],
                     "projected_30d_mb": g["projected_30d_8h_mb"]})
        sg.close()
    keys = [k for k in rows[0] if k != "seed"]
    return {"seeds": seeds, "runs": rows,
            "mean": {k: round(float(np.mean([r[k] for r in rows])), 4) for k in keys},
            "min": {k: round(float(np.min([r[k] for r in rows])), 4) for k in keys},
            "max": {k: round(float(np.max([r[k] for r in rows])), 4) for k in keys}}


# ---------------------------------------------------------------------------
def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--scale", type=int, default=10000, help="events for the at-scale latency run (0 = skip)")
    ap.add_argument("--out", default=str(Path(__file__).with_name("latest_report.json")))
    ap.add_argument("--keep", action="store_true", help="keep the temporary data directory")
    ap.add_argument("--sweep", type=int, default=10, help="extra simulator seeds for the robustness sweep (0 = skip)")
    ap.add_argument("--no-ablation", action="store_true", help="skip the ui_concepts=True ablation run")
    ap.add_argument("--heldout", type=int, default=60, help="scenes per case for the held-out morphology test (0 = skip)")
    ap.add_argument("--live", type=int, default=0, help="also capture N real desktop ticks (macOS) into a temp DB")
    args = ap.parse_args(argv)

    base = Path(tempfile.mkdtemp(prefix="screengraph-bench-"))
    t_start = time.perf_counter()
    try:
        sg = ScreenGraph(Config.ephemeral(base / "session"))
        sg.float_shadow = {}
        sim = WorkspaceSimulator(seed=args.seed)
        sess = run_session(sg, sim)
        ticks = sess["ticks"]
        stored = [t for t in ticks if t["stored"]]
        session_hours = sim.duration / 3600.0
        report = {
            "meta": {
                "screengraph_version": __version__, "seed": args.seed, "python": sys.version.split()[0],
                "numpy": np.__version__, "sqlite": sqlite3.sqlite_version, "platform": platform.platform(),
                "machine": platform.machine(), "processor": platform.processor() or platform.machine(),
                "cpu_count": os.cpu_count(), "generated_unix": int(time.time()),
                "load_average_1_5_15_at_start": [round(x, 2) for x in os.getloadavg()] if hasattr(os, "getloadavg") else None,
                "workload": {"ticks": len(ticks), "duration_min": sim.duration / 60, "windows": len(sim.windows),
                             "monitors": 2, "causal_chains": len(WorkspaceSimulator.CHAIN_TARGETS),
                             "visual_artifacts": len(WorkspaceSimulator.VISUAL_QUERIES),
                             "injected_secrets": len(sim.secrets)},
            },
            "gate_and_storage": gate_and_storage(sg, ticks, session_hours, base),
            "causal": causal_eval(sg, ticks),
            "visual": visual_eval(sg, ticks),
            "privacy": privacy_eval(sg, sim, base),
            "latency": {
                "phash_gate_check_dropped": lat([t["timings"]["gate_ms"] for t in ticks if not t["stored"]]),
                "phash_gate_check_all": lat([t["timings"]["gate_ms"] for t in ticks]),
                "patch_encode_bq_ram_wipe": lat([t["timings"]["encode_bq_wipe_ms"] for t in stored]),
                "pii_scrub": lat([t["timings"]["scrub_ms"] for t in stored]),
                "causal_edge_inference": lat([t["timings"]["causal_ms"] for t in stored]),
                "sqlite_txn_store": lat([t["timings"]["store_ms"] for t in stored]),
                "ingest_total_per_stored_frame": lat([t["timings"]["total_ms"] for t in stored]),
                "hybrid_rrf_graph_retrieval": retrieval_latency(sg),
            },
        }
        sg.close()
        if not args.no_ablation:
            report["ablation_ui_concepts"] = ablation_ui_concepts(base, args.seed)
            report["ablation_morphology_off"] = ablation_morphology_off(base, args.seed)
        if args.heldout:
            from evals.morph_heldout import gate_probe, run as heldout_run

            report["morph_heldout_test_split"] = heldout_run(args.heldout, seed=HELDOUT_TEST_SEED)
            report["morph_heldout_test_split"]["note"] = (
                f"independent renderer; seed {HELDOUT_TEST_SEED} was first evaluated once after the Phase-3b "
                "detectors were frozen (seed 1234 = development split)")
            report["morph_heldout_previous_split"] = heldout_run(args.heldout, seed=98765)
            report["morph_heldout_previous_split"]["note"] = (
                "Phase-3 test split; its aggregate results were SEEN before the Phase-3b changes, so it is "
                "no longer a clean test")
            report["gate_heldout_probe"] = gate_probe(2 * args.heldout, seed=HELDOUT_TEST_SEED)
        report["gate_spike_only_probe"] = gate_spike_only_probe()
        if args.live:
            from screengraph.capture.live import run_live_capture

            report["live_capture"] = run_live_capture(args.live, interval=1.0, data_dir=base / "live")
        if args.sweep:
            report["robustness_seed_sweep"] = seed_sweep([args.seed + 1 + i for i in range(args.sweep)], base)
        if args.scale:
            report["latency"]["at_scale"] = scale_eval(args.scale, base, args.seed)
        report["meta"]["benchmark_wall_s"] = round(time.perf_counter() - t_start, 2)
        if hasattr(os, "getloadavg"):
            report["meta"]["load_average_1_5_15_at_end"] = [round(x, 2) for x in os.getloadavg()]
    finally:
        if not args.keep:
            shutil.rmtree(base, ignore_errors=True)
    Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    print_report(report)
    print(f"\nreport written to {args.out}")
    return report


def print_report(r: dict) -> None:
    try:
        from rich.console import Console
        from rich.table import Table
    except ImportError:  # pragma: no cover
        print(json.dumps(r, indent=2, default=str))
        return
    con = Console()
    g, c, v, p, l = r["gate_and_storage"], r["causal"]["summary"], r["visual"]["summary"], r["privacy"], r["latency"]
    t = Table(title="screengraph-os benchmark", show_lines=False)
    t.add_column("metric")
    t.add_column("screengraph-os", justify="right")
    t.add_column("baseline", justify="right")
    t.add_row("frames dropped by idle gate", f"{g['drop_rate']:.1%}", "0% (continuous capture)")
    t.add_row("semantic changes missed by gate", str(g["semantic_changes_missed"]), "-")
    t.add_row("projected storage, 30d x 8h", f"{g['projected_30d_8h_mb']:.1f} MB", f"{g['legacy_model_30d_8h_mb']:.0f} MB (model)")
    t.add_row("image files / pixel blobs on disk", str(g["zero_pixel_audit"]["image_files_on_disk"] + g["zero_pixel_audit"]["pixel_blob_suspects"]), "-")
    t.add_row("root-cause recall@1", f"{c['screengraph_root_recall_at_1']:.0%}",
              f"{c['timestamp_baseline_root_recall_at_1']:.0%} ts / {c['flat_retrieval_root_recall_at_1']:.0%} flat")
    t.add_row("causal chain F1", f"{c['screengraph_chain_f1']:.3f}",
              f"{c['timestamp_baseline_chain_f1']:.3f} ts / {c['flat_retrieval_chain_f1']:.3f} flat")
    t.add_row("visual recall@1", f"{v['patch_maxsim_recall_at_1']:.0%}", f"{v['global_average_recall_at_1']:.0%} global-avg")
    t.add_row("quadrant localization", f"{v['patch_quadrant_localization_acc']:.0%}", "n/a")
    t.add_row("secret redaction recall", f"{p['secrets_redacted']}/{p['secrets_injected']}", "-")
    for k in ("phash_gate_check_dropped", "patch_encode_bq_ram_wipe", "causal_edge_inference", "hybrid_rrf_graph_retrieval"):
        t.add_row(f"{k} p50/p95", f"{l[k]['p50_ms']:.3f} / {l[k]['p95_ms']:.3f} ms", "")
    if "at_scale" in l:
        s = l["at_scale"]
        t.add_row(f"retrieval @ {s['events']} events p50/p95",
                  f"{s['hybrid_retrieval']['p50_ms']:.3f} / {s['hybrid_retrieval']['p95_ms']:.3f} ms", "")
    con.print(t)
    if "ablation_ui_concepts" in r:
        ab = r["ablation_ui_concepts"]
        con.print(f"ablation ui_concepts=True: visual R@1 {ab['visual']['patch_maxsim_recall_at_1']:.0%}, "
                  f"localization {ab['visual']['patch_quadrant_localization_acc']:.0%}, "
                  f"root R@1 {ab['causal']['screengraph_root_recall_at_1']:.0%}, chain F1 {ab['causal']['screengraph_chain_f1']:.3f}")
    if "ablation_morphology_off" in r:
        ab = r["ablation_morphology_off"]
        con.print(f"ablation morphology=False: visual R@1 {ab['visual']['patch_maxsim_recall_at_1']:.0%}, "
                  f"localization {ab['visual']['patch_quadrant_localization_acc']:.0%}, "
                  f"root R@1 {ab['causal']['screengraph_root_recall_at_1']:.0%}, chain F1 {ab['causal']['screengraph_chain_f1']:.3f}")
    if "morph_heldout_test_split" in r:
        pc = r["morph_heldout_test_split"]["per_concept"]
        con.print("held-out morphology (independent renderer): " + ", ".join(
            f"{k} R {v['recall']:.0%}/FPR {v['false_positive_rate']:.1%}" for k, v in pc.items()))
    if "gate_spike_only_probe" in r:
        gp = r["gate_spike_only_probe"]
        con.print(f"gate pass rate for spike-only redraws (simulator): {gp['passed_gate']}/{gp['spike_only_frames']}")
    if "gate_heldout_probe" in r:
        gh = r["gate_heldout_probe"]
        con.print(f"gate on held-out renderer: spike-only pass {gh['spike_only_pass_rate']:.1%}, cursor/caret false "
                  f"pass {gh['cursor_caret_false_pass_rate']:.1%} ({gh['cursor_caret_false_passes_via_excursion_stage']} "
                  f"via the excursion stage)")
    if "live_capture" in r:
        lv = r["live_capture"]
        con.print(f"live capture: {lv.get('status')} ticks={lv.get('ticks')} frames={lv.get('frames_seen')} "
                  f"stored={lv.get('events_stored')} ingest p95={lv.get('ingest_ms', {}).get('p95_ms')} ms "
                  f"image files on disk={lv.get('image_files_on_disk')}")
    if "robustness_seed_sweep" in r:
        sw = r["robustness_seed_sweep"]
        t2 = Table(title=f"robustness across {len(sw['seeds'])} other seeds (min / mean)")
        for k in ("drop_rate", "missed_changes", "root_recall", "chain_f1", "visual_r1", "global_r1",
                  "localization", "redaction_recall"):
            t2.add_column(k, justify="right")
        t2.add_row(*[f"{sw['min'][k]:.3g} / {sw['mean'][k]:.3g}" for k in ("drop_rate", "missed_changes", "root_recall",
                     "chain_f1", "visual_r1", "global_r1", "localization", "redaction_recall")])
        con.print(t2)


if __name__ == "__main__":
    main()
