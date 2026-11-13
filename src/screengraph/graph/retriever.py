"""4-way hybrid retrieval: FTS5 BM25 + dense Matryoshka cosine + visual Patch-MaxSim + causal graph walk.

Fusion is Reciprocal Rank Fusion,  RRF(d) = sum_r 1 / (k + rank_r(d)),  k = 60.
The graph channel takes the top seeds of the first three channels and
propagates their RRF mass along causal edges (both directions), so a query
that hits the IDE edit also surfaces the terminal error that caused it.
"""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from screengraph.config import Config
from screengraph.db.store import Store
from screengraph.embeddings.maxsim import PatchIndex, parse_spatial
from screengraph.embeddings.patch_encoder import PatchEncoder, binary_quantize, hamming, l2n, tokenize
from screengraph.graph.causal_engine import CausalGraph, Edge, TraceResult


class DenseIndex:
    """Global window vectors (float32 in RAM) with a 1-bit Hamming prefilter and Matryoshka slicing."""

    def __init__(self, dim: int):
        self.dim = dim
        self._ids = np.zeros(0, np.int64)
        self._vecs = np.zeros((0, dim), np.float32)
        self._bits = np.zeros((0, dim // 64), np.uint64)
        self.n = 0

    def add(self, eid: int, vec: np.ndarray) -> None:
        if self.n >= len(self._ids):
            new = max(64, 2 * len(self._ids))
            ids, vecs, bits = np.zeros(new, np.int64), np.zeros((new, self.dim), np.float32), np.zeros((new, self.dim // 64), np.uint64)
            ids[: self.n], vecs[: self.n], bits[: self.n] = self._ids[: self.n], self._vecs[: self.n], self._bits[: self.n]
            self._ids, self._vecs, self._bits = ids, vecs, bits
        v = np.asarray(vec, np.float32)
        self._ids[self.n] = eid
        self._vecs[self.n] = v
        self._bits[self.n] = binary_quantize(v)
        self.n += 1

    def vector(self, eid: int) -> Optional[np.ndarray]:
        idx = np.flatnonzero(self._ids[: self.n] == eid)
        return self._vecs[idx[0]] if len(idx) else None

    def search(self, q: np.ndarray, k: int = 50, dim: Optional[int] = None, candidates: int = 512) -> list[tuple[int, float]]:
        if self.n == 0 or not np.any(q):
            return []
        d = dim or self.dim
        if self.n > candidates and d == self.dim:
            ham = hamming(self._bits[: self.n], binary_quantize(q)[None, :])
            pool = np.argpartition(ham, candidates - 1)[:candidates]
        else:
            pool = np.arange(self.n)
        V = self._vecs[pool, :d]
        qs = l2n(q[:d])
        s = (V @ qs) / np.maximum(np.linalg.norm(V, axis=1), 1e-12)
        top = np.argsort(-s)[:k]
        return [(int(self._ids[pool[i]]), float(s[i])) for i in top]


def rrf(lists: dict[str, list[tuple[int, float]]], k: int = 60) -> tuple[dict[int, float], dict[int, dict[str, int]]]:
    fused: dict[int, float] = {}
    ranks: dict[int, dict[str, int]] = {}
    for name, lst in lists.items():
        for rank, (eid, _) in enumerate(lst, start=1):
            fused[eid] = fused.get(eid, 0.0) + 1.0 / (k + rank)
            ranks.setdefault(eid, {})[name] = rank
    return fused, ranks


@dataclass
class Hit:
    event_id: int
    score: float
    ranks: dict[str, int]
    event: dict
    snippet: str = ""
    patch: Optional[dict] = None
    links: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        e = self.event
        return {
            "event_id": self.event_id, "score": round(self.score, 6), "channel_ranks": self.ranks,
            "ts": e.get("ts"), "app": e.get("app"), "app_category": e.get("app_category"),
            "window_title": e.get("title"), "url": e.get("url") or None, "doc_path": e.get("doc_path") or None,
            "snippet": self.snippet, "patch": self.patch, "causal_links": self.links,
        }


def best_snippet(body: str, query: str, width: int = 160) -> str:
    terms = set(tokenize(query))
    best, best_s = "", -1
    for line in (body or "").splitlines():
        s = len(terms & set(tokenize(line)))
        if s > best_s and line.strip():
            best, best_s = line.strip(), s
    return best[:width]


class HybridRetriever:
    def __init__(self, cfg: Config, store: Store, encoder: PatchEncoder, dense: DenseIndex,
                 patches: PatchIndex, graph: CausalGraph):
        self.cfg, self.store, self.encoder = cfg, store, encoder
        self.dense, self.patches, self.graph = dense, patches, graph
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sg-bm25")

    # ------------------------------------------------------------------ search
    def channels(self, query: str, depth: Optional[int] = None) -> tuple[dict, Optional[dict]]:
        depth = depth or self.cfg.list_depth
        lists: dict[str, list[tuple[int, float]]] = {}
        # BM25 runs on a worker thread while the numpy channels run here: SQLite and BLAS both
        # release the GIL. The connection is never used by both threads at the same time.
        fut = self._pool.submit(self.store.fts_search, query, depth)
        lists["dense"] = self.dense.search(self.encoder.encode_text_query(query), depth)
        q_clean, spatial, _ = parse_spatial(query, self.patches.rows, self.patches.cols)
        _, Q = self.encoder.encode_query_tokens(q_clean)
        vhits = self.patches.search(Q, depth, spatial, candidates=self.cfg.prefilter_candidates)
        lists["visual"] = [(h.event_id, h.score) for h in vhits]
        loc = {h.event_id: {"row": h.row, "col": h.col, "quadrant": h.quadrant, "maxsim": round(h.score, 4)}
               for h in vhits}
        lists["bm25"] = fut.result()
        return lists, loc

    def graph_channel(self, fused: dict[int, float], seeds: int = 8, depth: int = 50) -> list[tuple[int, float]]:
        mass: dict[int, float] = {}
        top = sorted(fused.items(), key=lambda x: -x[1])[:seeds]
        for eid, s in top:
            mass[eid] = mass.get(eid, 0.0) + s
            for nb, e in self.graph.neighbors(eid):
                w = s * e.score * (1.0 if e.causal else 0.5)
                mass[nb] = mass.get(nb, 0.0) + w
                for nb2, e2 in self.graph.neighbors(nb):
                    if nb2 != eid and e2.causal:
                        mass[nb2] = mass.get(nb2, 0.0) + 0.5 * w * e2.score
        return sorted(mass.items(), key=lambda x: -x[1])[:depth]

    def search(self, query: str, k: int = 10, channels: tuple[str, ...] = ("bm25", "dense", "visual", "graph"),
               app_category: Optional[str] = None) -> list[Hit]:
        lists, loc = self.channels(query)
        lists = {n: l for n, l in lists.items() if n in channels}
        fused, _ = rrf(lists, self.cfg.rrf_k)
        if "graph" in channels:
            lists["graph"] = self.graph_channel(fused)
        fused, ranks = rrf(lists, self.cfg.rrf_k)
        order = sorted(fused.items(), key=lambda x: -x[1])
        events = self.store.get_events([e for e, _ in order[: k * 3]])
        hits: list[Hit] = []
        for eid, s in order:
            ev = events.get(eid)
            if ev is None or (app_category and ev["app_category"] != app_category):
                continue
            links = [{"event_id": nb, "relation": e.relation, "direction": "cause" if e.dst == eid else "effect",
                      "score": e.score} for nb, e in self.graph.neighbors(eid) if e.causal][:5]
            hits.append(Hit(eid, s, ranks.get(eid, {}), ev, best_snippet(ev.get("body", ""), query),
                            (loc or {}).get(eid), links))
            if len(hits) >= k:
                break
        return hits

    def visual_search(self, query: str, k: int = 5) -> list[dict]:
        q_clean, spatial, qlabel = parse_spatial(query, self.patches.rows, self.patches.cols)
        toks, Q = self.encoder.encode_query_tokens(q_clean)
        hits = self.patches.search(Q, k, spatial, candidates=self.cfg.prefilter_candidates)
        events = self.store.get_events([h.event_id for h in hits])
        out = []
        for h in hits:
            ev = events.get(h.event_id, {})
            pw, ph = (ev.get("w") or 0) / self.patches.cols, (ev.get("h") or 0) / self.patches.rows
            concepts = self.store.get_concepts(h.event_id)
            out.append({
                "event_id": h.event_id, "maxsim": round(h.score, 4), "ts": ev.get("ts"), "app": ev.get("app"),
                "window_title": ev.get("title"), "monitor_id": ev.get("monitor_id"),
                "patch": {"row": h.row, "col": h.col, "quadrant": h.quadrant,
                          "screen_bbox": {"x": int((ev.get("x") or 0) + h.col * pw), "y": int((ev.get("y") or 0) + h.row * ph),
                                          "width": int(pw), "height": int(ph)},
                          "visual_concepts": concepts[h.row * self.patches.cols + h.col] if concepts else {}},
                "query_tokens": toks, "spatial_constraint": qlabel,
            })
        return out

    def visual_search_like(self, event_id: int, k: int = 5) -> list[dict]:
        """Query-by-example: the reference window's own patch vectors are the query tokens (late interaction
        against every other window), so windows that *look and read* alike rank first."""
        pos = self.patches.position(np.array([event_id]))[0]
        if pos < 0:
            return []
        Q = self.patches.patches(np.array([pos]))[0]
        hits = [h for h in self.patches.search(Q, k + 1, candidates=self.cfg.prefilter_candidates)
                if h.event_id != event_id][:k]
        events = self.store.get_events([h.event_id for h in hits])
        return [{"event_id": h.event_id, "maxsim": round(h.score / len(Q), 4), "app": events.get(h.event_id, {}).get("app"),
                 "window_title": events.get(h.event_id, {}).get("title"), "best_patch": [h.row, h.col]} for h in hits]

    # ---------------------------------------------------------------- causal why
    def resolve_targets(self, target: str) -> list[int]:
        target = target.strip()
        evs = self.store.events_with_doc_path(target) if re.search(r"[\w-]\.\w{1,5}$", target) else []
        if evs:
            return [evs[-1]["id"]]
        hits = self.store.fts_search(target, 20)
        if not hits:
            hits = self.dense.search(self.encoder.encode_text_query(target), 5)
        events = self.store.get_events([h[0] for h in hits])
        ide = [h[0] for h in hits if events.get(h[0], {}).get("app_category") == "ide"]
        return [ide[0]] if ide else ([hits[0][0]] if hits else [])

    def why(self, target: str, max_depth: int = 6) -> dict:
        t0 = time.perf_counter()
        targets = self.resolve_targets(target)
        if not targets:
            return {"target": target, "found": False, "message": "no matching events"}
        tr: TraceResult = self.graph.trace_back(targets, max_depth=max_depth)
        ids = set(tr.chain) | set(targets) | {r for r, _ in tr.roots[:3]}
        events = self.store.get_events(ids)

        def brief(eid: int) -> dict:
            e = events.get(eid, {})
            return {"event_id": eid, "ts": e.get("ts"), "app": e.get("app"), "app_category": e.get("app_category"),
                    "window_title": e.get("title"), "url": e.get("url") or None, "doc_path": e.get("doc_path") or None,
                    "snippet": best_snippet(e.get("body", ""), target + " error exception failed")}

        steps = []
        for eid in tr.chain:
            incoming = [e for e in tr.edges if e.dst == eid]
            steps.append({**brief(eid), "via": [{"from": e.src, "relation": e.relation, "score": e.score,
                                                 "evidence": e.evidence[:4]} for e in incoming]})
        narrative = []
        for s in steps:
            via = f" <- {s['via'][0]['relation']} ({', '.join(s['via'][0]['evidence'][:2])})" if s["via"] else ""
            narrative.append(f"[{s['app']}] {s['window_title']}{via}")
        return {
            "target": target, "found": True, "target_events": [brief(t) for t in targets],
            "root_causes": [{**brief(r), "root_score": sc} for r, sc in tr.roots[:3]],
            "chain": steps, "narrative": narrative,
            "mermaid": self.mermaid(tr.chain, tr.edges, events),
            "elapsed_ms": round((time.perf_counter() - t0) * 1e3, 3),
        }

    # ----------------------------------------------------------------- timeline
    @staticmethod
    def _mlabel(e: dict) -> str:
        t = (e.get("title") or e.get("app") or "")[:40]
        return f"{e.get('app', '')}: {t}".replace('"', "'").replace("[", "(").replace("]", ")")

    def mermaid(self, nodes: list[int], edges: list[Edge], events: dict[int, dict]) -> str:
        lines = ["graph LR"]
        for n in nodes:
            if n in events:
                lines.append(f'  e{n}["{self._mlabel(events[n])}"]')
        for e in edges:
            if e.src in events and e.dst in events:
                arrow = "==>" if e.causal else "-.->"
                lines.append(f"  e{e.src} {arrow}|{e.relation} {e.score:.2f}| e{e.dst}")
        return "\n".join(lines)

    def timeline(self, t0: float, t1: float, causal_only: bool = False, limit: int = 200) -> dict:
        evs = self.store.timeline(t0, t1, limit)
        ids = [e["id"] for e in evs]
        idset = set(ids)
        edges = [Edge.from_row(r) for r in self.store.edges_touching(ids) if r["src"] in idset and r["dst"] in idset]
        if causal_only:
            edges = [e for e in edges if e.causal]
        byid = {e["id"]: e for e in evs}
        narrative = []
        parents: dict[int, list[Edge]] = {}
        for e in edges:
            parents.setdefault(e.dst, []).append(e)
        for e in evs:
            ps = sorted(parents.get(e["id"], []), key=lambda x: -x.score)
            cause = ""
            if ps and ps[0].causal:
                src = byid.get(ps[0].src, {})
                cause = f"  <= {ps[0].relation} from #{ps[0].src} ({src.get('app', '')})"
            narrative.append({"event_id": e["id"], "ts": e["ts"], "app": e["app"], "title": e["title"],
                              "line": f"#{e['id']} [{e['app_category']}] {e['title']}{cause}"})
        shown = ids if not causal_only else sorted({x for ed in edges for x in (ed.src, ed.dst)})
        return {"t_start": t0, "t_end": t1, "events": len(evs), "edges": len(edges),
                "narrative": narrative, "mermaid": self.mermaid(shown, edges, byid)}
