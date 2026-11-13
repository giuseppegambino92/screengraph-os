"""ScreenGraph: the ingestion pipeline and query facade.

    frame -> [AX hash] -> PerceptualGate --drop--> wipe buffer, done (no I/O)
                              | pass
                              v
            EphemeralFrameBuffer{ scrub AX text -> patch encode -> Matryoshka + 1-bit BQ } -> wipe
                              v
            entities -> SQLite txn { event, FTS5, vectors, redaction log, causal edges, audit }
"""

from __future__ import annotations

import contextlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from screengraph.capture.ephemeral import EphemeralFrameBuffer, ZeroingLedger
from screengraph.capture.models import AXNode, BoundingBox, ViewportFrame
from screengraph.capture.phash import PerceptualGate, ax_tree_hash
from screengraph.config import Config
from screengraph.db.store import Store
from screengraph.embeddings.maxsim import PatchIndex
from screengraph.embeddings.patch_encoder import (PatchEncoder, binary_quantize, dequantize_int8, matryoshka,
                                                  quantize_int8, tokenize)
from screengraph.graph.causal_engine import (CHROME_ROLES, ROLE_SEP, CausalEngine, CausalGraph, CausalNode, Edge,
                                             extract_entities, norm_text)
from screengraph.graph.retriever import DenseIndex, HybridRetriever
from screengraph.privacy.audit import scan_directory, scan_sqlite_blobs
from screengraph.privacy.scrubber import Scrubber


@dataclass
class IngestResult:
    stored: bool
    reason: str
    event_id: Optional[int] = None
    edges: list[Edge] = field(default_factory=list)
    redactions: int = 0
    timings_ms: dict[str, float] = field(default_factory=dict)


class ScreenGraph:
    def __init__(self, cfg: Optional[Config] = None, db_path: Optional[str | Path] = None):
        self.cfg = cfg or Config()
        self.cfg.ensure_dirs()
        self.store = Store(db_path or self.cfg.db_path)
        self.scrubber = Scrubber(self.cfg.resolve_hmac_key(), self.cfg.entropy_threshold,
                                 self.cfg.redact_emails, self.cfg.redact_ips)
        from screengraph.embeddings.patch_encoder import load_adapter

        self.encoder = PatchEncoder(self.cfg.dim, self.cfg.grid_rows, self.cfg.grid_cols, self.cfg.embed_seed,
                                    self.cfg.visual_weight, self.cfg.text_weight, self.cfg.descriptor_weight,
                                    ui_concepts=self.cfg.ui_concepts, adapter=load_adapter(self.cfg.vision_adapter),
                                    morphology=self.cfg.morphology)
        self.gate = PerceptualGate(self.cfg.phash_threshold, self.cfg.dhash_threshold, self.cfg.phash_size,
                                   self.cfg.quadrant_threshold)
        self.ledger = ZeroingLedger()
        self.causal = CausalEngine(self.cfg, transitions=self.store.transitions())
        self.graph = CausalGraph()
        self.dense = DenseIndex(self.cfg.dim)
        self.patches = PatchIndex(self.cfg.grid_rows, self.cfg.grid_cols, self.cfg.patch_store_dim, self.cfg.prefilter_dim)
        self.counters = {"ticks": 0, "dropped": 0, "stored": 0, "redactions": 0}
        self.float_shadow: Optional[dict] = None  # eval only: set to {} to keep unquantized patch vectors in RAM
        self._load()
        self.retriever = HybridRetriever(self.cfg, self.store, self.encoder, self.dense, self.patches, self.graph)

    # ------------------------------------------------------------------ boot
    def _load(self) -> None:
        audit = self.store.get_audit()
        if "ledger" in audit:
            self.ledger.restore(audit["ledger"])
        if "counters" in audit:
            self.counters.update(audit["counters"])
        self._max_loaded = 0
        self.refresh()
        lo, hi = self.store.time_range()
        if hi is not None:
            for ev in self.store.recent_events(hi - self.cfg.causal_horizon_s):
                g = self.dense.vector(ev["id"])
                if g is not None:
                    self.causal.add(self._node_from_row(ev, g))

    def refresh(self) -> int:
        """Load events/vectors/edges written since the last load (e.g. by a separate capture
        daemon while this process serves MCP queries). Returns the number of new events."""
        P = self.cfg.grid_rows * self.cfg.grid_cols
        new = 0
        for r in self.store.conn.execute(
                "SELECT event_id, global_i8, global_scale, patch_bq, patch_i8, patch_scale FROM event_vectors "
                "WHERE event_id > ? ORDER BY event_id", (self._max_loaded,)):
            self.dense.add(r["event_id"], dequantize_int8(np.frombuffer(r["global_i8"], np.int8), r["global_scale"]))
            self.patches.add(r["event_id"], np.frombuffer(r["patch_i8"], np.int8).reshape(P, -1), r["patch_scale"],
                             np.frombuffer(r["patch_bq"], np.uint64).reshape(P, -1))
            new += 1
        if new:
            for r in self.store.conn.execute("SELECT id, ts, entities FROM events WHERE id > ?", (self._max_loaded,)):
                flags = json.loads(r["entities"] or "{}").get("flags", [])
                self.graph.add_node(r["id"], r["ts"], "error" in flags or "anomaly" in flags)
            self.graph.add_edges(Edge.from_row(r) for r in self.store.edges("WHERE dst > ?", (self._max_loaded,)))
            self._max_loaded = int(self.store.conn.execute("SELECT MAX(id) FROM events").fetchone()[0] or 0)
        return new

    def _node_from_row(self, ev: dict, g: np.ndarray) -> CausalNode:
        ents = ev["entities"]
        node = CausalNode.build(ev["id"], ev["ts"], ev["window_id"], ev["app_category"], g, ents,
                                ev.get("body", ""), ev.get("title", ""), ev.get("url", ""), ev.get("doc_path", ""),
                                (ents.get("clipboard") or [""])[0])
        return self._attach_anomaly_vec(node)

    def _attach_anomaly_vec(self, node: CausalNode) -> CausalNode:
        """Cross-modal anomaly vector: caption tokens + the morphological bank words of an excursion, in the
        same concept space as text states, so cos(anomaly, destination) is a visual-to-code similarity."""
        if node.anomalies:
            node.anomaly_vec = self.encoder.text_vector(
                [t for a in node.anomalies for t in tokenize(a["caption"])] + ["spike", "anomaly"])
        return node

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.store.set_audit({"ledger": self.ledger.snapshot(), "counters": self.counters})
        self.store.close()

    def __enter__(self) -> "ScreenGraph":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---------------------------------------------------------------- ingest
    def ingest_frame(self, frame: ViewportFrame) -> IngestResult:
        t0 = time.perf_counter()
        self.counters["ticks"] += 1
        ax_hash = ax_tree_hash(frame.title, frame.url, frame.doc_path, frame.ax_nodes)
        pixels = frame.pixels
        buf = EphemeralFrameBuffer(pixels, self.ledger) if pixels is not None else None
        timings: dict[str, float] = {}
        with buf if buf is not None else contextlib.nullcontext():
            dec = self.gate.check(frame.window_id, ax_hash, pixels)
            if not dec.passed and frame.clipboard_copied:
                dec.passed, dec.reason = True, "clipboard_copy"  # a copy is a semantic event even if pixels are idle
            timings["gate_ms"] = dec.elapsed_ms
            if not dec.passed:
                self.counters["dropped"] += 1
                drop_result = IngestResult(False, dec.reason, timings_ms=timings)
            else:
                drop_result = None
                t1 = time.perf_counter()
                lines, reds = self.scrubber.scrub_lines([n.text for n in frame.ax_nodes])
                title = self.scrubber.scrub(frame.title)
                url = self.scrubber.scrub(frame.url)
                clip = self.scrubber.scrub(frame.clipboard_copied) if frame.clipboard_copied else None
                reds = reds + title.redactions + url.redactions + (clip.redactions if clip else [])
                t2 = time.perf_counter()
                timings["scrub_ms"] = (t2 - t1) * 1e3
                enc = self.encoder.encode(
                    pixels, [(s, n.bbox) for s, n in zip(lines, frame.ax_nodes)], title=title.text,
                    extra_text=f"{url.text} {frame.doc_path}", buf=buf,
                    frame_size=(frame.bbox.height, frame.bbox.width) if pixels is None else None,
                )
                # int8 symmetric SQ (one scale per window) of the 128-d Matryoshka prefix + 64-bit sign codes
                patch_i8, patch_scale = quantize_int8(matryoshka(enc.patch_vecs, self.cfg.patch_store_dim))
                patch_bq = binary_quantize(matryoshka(enc.patch_vecs, self.cfg.prefilter_dim))
                g = enc.global_vec
                g_i8, g_scale = quantize_int8(g)
                g_bq = binary_quantize(g)
        t3 = time.perf_counter()  # buffer wiped at this point
        if drop_result is not None:
            drop_result.timings_ms["total_ms"] = (t3 - t0) * 1e3
            return drop_result
        timings["encode_bq_wipe_ms"] = (t3 - t2) * 1e3

        body = "\n".join(lines)
        # entities come from *content* nodes only (not titles, explorer rows, tabs, toolbars)
        content = [(n.role, s) for s, n in zip(lines, frame.ax_nodes) if n.role not in CHROME_ROLES and s]
        ents = extract_entities("\n".join(t for _, t in content), url.text, frame.doc_path)
        visual = sorted({c for pc in enc.patch_concepts for c in pc if c not in ("dark", "light")})
        if visual:
            ents["visual"] = visual
            if "spike" in visual and "anomaly" not in ents["flags"]:
                ents["flags"].append("anomaly")
        anomalies = self._caption_anomalies(enc.objects, frame.ax_nodes, lines, frame)
        if anomalies:
            ents["visual_anomalies"] = anomalies
        if clip:
            ents["clipboard"] = [norm_text(clip.text)[:500]]
        rec = {
            "ts": frame.ts, "window_id": frame.window_id, "app": frame.app, "app_category": frame.app_category,
            "bundle_id": frame.bundle_id, "title": title.text, "pid": frame.pid, "monitor_id": frame.bbox.monitor_id,
            "x": frame.bbox.x, "y": frame.bbox.y, "w": frame.bbox.width, "h": frame.bbox.height,
            "z_index": frame.z_index, "url": url.text, "doc_path": frame.doc_path, "body": body, "entities": ents,
            "clipboard_hash": self.scrubber.hash_text(norm_text(clip.text)) if clip else None,
            "phash": dec.phash, "dhash": dec.dhash, "source": frame.source, "gate_reason": dec.reason,
        }
        with self.store.transaction():
            t4 = time.perf_counter()
            eid = self.store.insert_event(rec)
            self.store.insert_vectors(eid, self.cfg.dim, self.cfg.grid_rows, self.cfg.grid_cols, g_i8.tobytes(), g_scale,
                                      g_bq.tobytes(), patch_bq.tobytes(), patch_i8.tobytes(), patch_scale,
                                      enc.patch_concepts)
            if reds:
                self.store.log_redactions(eid, frame.ts, reds)
            if self.float_shadow is not None:
                self.float_shadow[eid] = matryoshka(enc.patch_vecs, self.cfg.patch_store_dim)
            node = CausalNode.build(eid, frame.ts, frame.window_id, frame.app_category, g, ents, body, title.text,
                                    url.text, frame.doc_path, clip.text if clip else "",
                                    content_lines=[f"{r}{ROLE_SEP}{t}" for r, t in content])
            self._attach_anomaly_vec(node)
            prev_cat = self.causal._last_cat
            t5 = time.perf_counter()
            edges = self.causal.infer(node)
            timings["causal_ms"] = (time.perf_counter() - t5) * 1e3
            if edges:
                self.store.insert_edges(e.row() for e in edges)
            if prev_cat is not None:
                self.store.bump_transition(prev_cat, frame.app_category)
            self.counters["stored"] += 1
            self.counters["redactions"] += len(reds)
            self.store.set_audit({"ledger": self.ledger.snapshot(), "counters": self.counters})
            timings["store_ms"] = (time.perf_counter() - t4) * 1e3 - timings["causal_ms"]
        self.dense.add(eid, dequantize_int8(g_i8, g_scale))  # RAM index == what a reload sees
        self.patches.add(eid, patch_i8, patch_scale, patch_bq)
        self._max_loaded = max(self._max_loaded, eid)
        self.graph.add_node(eid, frame.ts, node.has_error)
        self.graph.add_edges(edges)
        timings["total_ms"] = (time.perf_counter() - t0) * 1e3
        return IngestResult(True, dec.reason, eid, edges, len(reds), timings)

    def _caption_anomalies(self, objects: list[dict], ax_nodes, lines: list[str], frame: ViewportFrame) -> list[dict]:
        """Pure-pixel anomalies (vertical excursions) bound to the subsystem they plot.

        Caption association (document-layout convention): the scrubbed AX label nearest *above* the
        curve's bounding box that overlaps it horizontally, or a label inside it, within 25% of the
        window height. The caption text is what the anomaly is *about* ("payments-api memory")."""
        out = []
        if not objects or frame.pixels is None:
            return out
        h = max(1, frame.pixels.shape[0])
        for o in objects:
            if o.get("type") != "spike" or "bbox" not in o:
                continue
            ox, oy, ow, oh = o["bbox"]
            best, best_d = None, None
            for n, txt in zip(ax_nodes, lines):
                if not txt or n.role in CHROME_ROLES:
                    continue
                nx, ny, nw, nh = n.bbox
                if nx > ox + ow or nx + nw < ox:  # must overlap the curve horizontally
                    continue
                if ny + nh > oy + oh or ny < oy - 0.25 * h:  # above (or inside) the curve, not far above
                    continue
                d = max(0, oy - (ny + nh))
                if best_d is None or d < best_d:
                    best, best_d = txt, d
            if best:
                out.append({"caption": best[:160], "energy": float(o.get("energy", 1.0)), "x": int(o["x"]),
                            "z": float(o.get("z", 0.0))})
        return out

    def ingest_text_event(self, app: str, app_category: str, text: str, title: str = "", url: str = "",
                          doc_path: str = "", window_id: Optional[str] = None, ts: Optional[float] = None,
                          clipboard: Optional[str] = None, source: str = "hook") -> IngestResult:
        """Hook / MCP path: terminal commands, IDE saves, manual notes. No pixels involved."""
        lines = (text or "").splitlines() or [""]
        nodes = [AXNode("text", ln, (0, i * 10, 1000, 10)) for i, ln in enumerate(lines)]
        frame = ViewportFrame(
            window_id=window_id or f"{app_category}:{app}", app=app, title=title or app,
            bbox=BoundingBox(0, 0, 1000, max(10 * len(lines), 40)), ts=ts if ts is not None else time.time(),
            app_category=app_category, url=url, doc_path=doc_path, ax_nodes=nodes, clipboard_copied=clipboard,
            source=source,
        )
        return self.ingest_frame(frame)

    # ----------------------------------------------------------------- queries
    def search(self, query: str, k: int = 10, **kw):
        return self.retriever.search(query, k, **kw)

    def why(self, target: str) -> dict:
        return self.retriever.why(target)

    def visual_search(self, query: str, k: int = 5) -> list[dict]:
        return self.retriever.visual_search(query, k)

    def visual_search_like(self, event_id: int, k: int = 5) -> list[dict]:
        return self.retriever.visual_search_like(event_id, k)

    def timeline(self, t0: Optional[float] = None, t1: Optional[float] = None, causal_only: bool = False) -> dict:
        lo, hi = self.store.time_range()
        if lo is None:
            return {"events": 0, "edges": 0, "narrative": [], "mermaid": "graph LR"}
        return self.retriever.timeline(t0 if t0 is not None else lo, t1 if t1 is not None else hi, causal_only)

    def active_workspace(self, live_frames: Optional[list[ViewportFrame]] = None) -> dict:
        latest = self.store.latest_per_window()
        ids = [e["id"] for e in latest]
        edges = [Edge.from_row(r) for r in self.store.edges_touching(ids)]
        windows = []
        for e in latest:
            links = [{"from": ed.src, "to": ed.dst, "relation": ed.relation, "score": ed.score}
                     for ed in edges if (ed.src == e["id"] or ed.dst == e["id"]) and ed.causal][:6]
            body_lines = [l for l in (e.get("body") or "").splitlines() if l.strip()]
            windows.append({
                "window_id": e["window_id"], "event_id": e["id"], "app": e["app"], "app_category": e["app_category"],
                "title": e["title"], "url": e["url"] or None, "doc_path": e["doc_path"] or None,
                "bbox": {"x": e["x"], "y": e["y"], "width": e["w"], "height": e["h"], "monitor_id": e["monitor_id"]},
                "last_change_ts": e["ts"], "summary": body_lines[-6:],
                "entities": {k: v[:8] for k, v in e["entities"].items() if v and k not in ("clipboard",)},
                "causal_links": links,
            })
        out = {"windows": windows, "monitors": sorted({w["bbox"]["monitor_id"] for w in windows}),
               "screenshots_used": 0}
        if live_frames is not None:
            out["live"] = [{"window_id": f.window_id, "app": f.app, "title": f.title, "bbox": f.bbox.to_dict(),
                            "source": f.source} for f in live_frames]
        return out

    def privacy_audit(self) -> dict:
        self.store.set_audit({"ledger": self.ledger.snapshot(), "counters": self.counters})
        disk = scan_directory(Path(self.store.path).parent) if self.store.path != ":memory:" else {}
        blobs = scan_sqlite_blobs(self.store.conn)
        led = self.ledger.snapshot()
        reds = self.store.redaction_counts()
        ok = (disk.get("image_magic_hits", 0) == 0 and disk.get("image_extension_hits", 0) == 0
              and blobs["pixel_blob_suspects"] == 0 and led["verification_failures"] == 0)
        return {
            "zero_pixel_guarantee_holds": ok,
            "disk": disk, "sqlite": blobs,
            "ram": {**led, "all_wipes_verified_zero": led["verification_failures"] == 0},
            "gate": {"ticks": self.counters["ticks"], "dropped": self.counters["dropped"],
                     "stored": self.counters["stored"]},
            "redactions": {"total": sum(reds.values()), "by_kind": reds},
        }

    def stats(self, checkpoint: bool = False) -> dict:
        c = self.store.counts()
        lo, hi = self.store.time_range()
        return {**c, "db_bytes": self.store.size_bytes(checkpoint=checkpoint), "span_s": (hi - lo) if lo is not None else 0,
                **{f"counter_{k}": v for k, v in self.counters.items()}}
