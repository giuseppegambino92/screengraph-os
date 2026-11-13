"""Transactional graph + vector store on SQLite.

The public write API accepts only scrubbed text, entity JSON, hashes, packed
bit-vectors and float16 vectors. There is deliberately no method that accepts
an ndarray of pixels or image bytes.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, Optional

from screengraph.db.schema import connect

_FTS_TOKEN = re.compile(r"[A-Za-z0-9_]+")
_FTS_STOP = frozenset("a an the of in on at to for from with by and or is are was why did i my me what".split())


def _to_signed64(x: int) -> int:
    return x - (1 << 64) if x >= (1 << 63) else x


def fts_terms(text: str) -> list[str]:
    toks = [t for t in _FTS_TOKEN.findall(text or "") if t.lower() not in _FTS_STOP]
    return list(dict.fromkeys(toks))[:24]


def fts_query(text: str) -> str:
    return " OR ".join(f'"{t}"' for t in fts_terms(text))


class Store:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn: sqlite3.Connection = connect(self.path)

    def close(self) -> None:
        try:
            self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.Error:
            pass
        self.conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            yield self.conn
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        else:
            self.conn.execute("COMMIT")

    # ---------------------------------------------------------------- writes
    def insert_event(self, rec: dict) -> int:
        ents = json.dumps(rec.get("entities") or {}, sort_keys=True, separators=(",", ":"))
        cur = self.conn.execute(
            """INSERT INTO events (ts, window_id, app, app_category, bundle_id, title, pid, monitor_id,
                                   x, y, w, h, z_index, url, doc_path, body, entities, clipboard_hash,
                                   phash, dhash, source, gate_reason)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                rec["ts"], rec["window_id"], rec["app"], rec["app_category"], rec.get("bundle_id", ""),
                rec.get("title", ""), rec.get("pid", 0), rec.get("monitor_id", 0),
                rec.get("x", 0), rec.get("y", 0), rec.get("w", 0), rec.get("h", 0), rec.get("z_index", 0),
                rec.get("url", ""), rec.get("doc_path", ""), rec.get("body", ""), ents,
                rec.get("clipboard_hash"), _to_signed64(rec.get("phash", 0)), _to_signed64(rec.get("dhash", 0)),
                rec.get("source", "sim"), rec.get("gate_reason", ""),
            ),
        )
        eid = int(cur.lastrowid)
        self.conn.execute(
            "INSERT INTO events_fts(rowid, title, body, entities, app) VALUES (?,?,?,?,?)",
            (eid, rec.get("title", ""), rec.get("body", ""), ents, rec["app"]),
        )
        return eid

    def insert_vectors(self, event_id: int, dim: int, rows: int, cols: int, global_i8: bytes, global_scale: float,
                       global_bq: bytes, patch_bq: bytes, patch_i8: bytes, patch_scale: float, concepts: list) -> None:
        compact = {str(i): _pack_concepts(c) for i, c in enumerate(concepts) if c}  # only patches with concepts
        self.conn.execute(
            "INSERT INTO event_vectors VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (event_id, dim, rows, cols, global_i8, float(global_scale), global_bq, patch_bq, patch_i8,
             float(patch_scale), json.dumps(compact, separators=(",", ":")) if compact else None),
        )

    def insert_edges(self, edges: Iterable[tuple]) -> None:
        self.conn.executemany(
            "INSERT OR REPLACE INTO causal_edges VALUES (?,?,?,?,?,?,?,?,?,?)", list(edges)
        )

    def log_redactions(self, event_id: Optional[int], ts: float, reds: Iterable) -> None:
        self.conn.executemany(
            "INSERT INTO redaction_log(ts, event_id, kind, category, tag) VALUES (?,?,?,?,?)",
            [(ts, event_id, r.kind, r.category, r.tag) for r in reds],
        )

    def bump_transition(self, src: str, dst: str, inc: float = 1.0) -> None:
        self.conn.execute(
            "INSERT INTO app_transitions VALUES (?,?,?) "
            "ON CONFLICT(src_cat, dst_cat) DO UPDATE SET count = count + excluded.count",
            (src, dst, inc),
        )

    def set_audit(self, values: dict) -> None:
        self.conn.executemany(
            "INSERT INTO privacy_audit VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            [(k, json.dumps(v)) for k, v in values.items()],
        )

    # ----------------------------------------------------------------- reads
    def get_audit(self) -> dict:
        return {r["key"]: json.loads(r["value"]) for r in self.conn.execute("SELECT key, value FROM privacy_audit")}

    def transitions(self) -> dict[tuple[str, str], float]:
        return {(r[0], r[1]): r[2] for r in self.conn.execute("SELECT src_cat, dst_cat, count FROM app_transitions")}

    @staticmethod
    def _row(r: sqlite3.Row) -> dict:
        d = dict(r)
        d["entities"] = json.loads(d.get("entities") or "{}")
        return d

    def get_event(self, event_id: int) -> Optional[dict]:
        r = self.conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
        return self._row(r) if r else None

    def get_events(self, ids: Iterable[int]) -> dict[int, dict]:
        ids = list({int(i) for i in ids})
        out: dict[int, dict] = {}
        for i in range(0, len(ids), 500):
            chunk = ids[i:i + 500]
            q = f"SELECT * FROM events WHERE id IN ({','.join('?' * len(chunk))})"
            for r in self.conn.execute(q, chunk):
                out[r["id"]] = self._row(r)
        return out

    def fts_search(self, query: str, k: int = 50, max_df: float = 0.2) -> list[tuple[int, float]]:
        """BM25 over FTS5. Query terms present in more than ``max_df`` of all events have
        near-zero IDF but dominate the cost of an OR query, so they are pruned (the two rarest
        terms are always kept)."""
        terms = fts_terms(query)
        if not terms:
            return []
        if len(terms) > 2:
            n = self.conn.execute("SELECT MAX(rowid) FROM events").fetchone()[0] or 0
            ph = ",".join("?" * len(terms))
            df = {r[0]: r[1] for r in self.conn.execute(
                f"SELECT term, doc FROM events_fts_vocab WHERE term IN ({ph})", [t.lower() for t in terms])}
            terms.sort(key=lambda t: df.get(t.lower(), 0))
            terms = [t for i, t in enumerate(terms) if i < 2 or df.get(t.lower(), 0) <= max_df * n]
        q = " OR ".join(f'"{t}"' for t in terms)
        try:
            rows = self.conn.execute(
                "SELECT rowid, bm25(events_fts, 2.0, 1.0, 1.5, 0.5) AS s FROM events_fts "
                "WHERE events_fts MATCH ? ORDER BY s LIMIT ?", (q, k)
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [(int(r[0]), float(-r[1])) for r in rows]

    def events_with_doc_path(self, path: str) -> list[dict]:
        base = os.path.basename(path)
        rows = self.conn.execute(
            "SELECT * FROM events WHERE doc_path = ? OR doc_path LIKE ? ORDER BY ts",
            (path, f"%/{base}" if base else path),
        ).fetchall()
        return [self._row(r) for r in rows]

    def timeline(self, t0: float, t1: float, limit: int = 500) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM events WHERE ts BETWEEN ? AND ? ORDER BY ts LIMIT ?", (t0, t1, limit)
        ).fetchall()
        return [self._row(r) for r in rows]

    def latest_per_window(self) -> list[dict]:
        rows = self.conn.execute(
            """SELECT e.* FROM events e JOIN (SELECT window_id, MAX(ts) AS mts FROM events GROUP BY window_id) m
               ON e.window_id = m.window_id AND e.ts = m.mts ORDER BY e.monitor_id, e.x"""
        ).fetchall()
        return [self._row(r) for r in rows]

    def recent_events(self, since_ts: float) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM events WHERE ts >= ? ORDER BY ts", (since_ts,)).fetchall()
        return [self._row(r) for r in rows]

    def edges(self, where: str = "", params: tuple = ()) -> list[dict]:
        rows = self.conn.execute(f"SELECT * FROM causal_edges {where}", params).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["evidence"] = json.loads(d["evidence"] or "[]")
            out.append(d)
        return out

    def edges_touching(self, ids: Iterable[int]) -> list[dict]:
        ids = list(ids)
        if not ids:
            return []
        ph = ",".join("?" * len(ids))
        return self.edges(f"WHERE src IN ({ph}) OR dst IN ({ph})", tuple(ids) * 2)

    def get_concepts(self, event_id: int, patches: int = 16) -> list:
        r = self.conn.execute("SELECT concepts FROM event_vectors WHERE event_id = ?", (event_id,)).fetchone()
        compact = json.loads(r[0]) if r and r[0] else {}
        return [_unpack_concepts(compact.get(str(i), {})) for i in range(patches)]

    def redaction_counts(self) -> dict[str, int]:
        return {r[0]: r[1] for r in self.conn.execute("SELECT kind, COUNT(*) FROM redaction_log GROUP BY kind")}

    def counts(self) -> dict[str, int]:
        c = self.conn
        return {
            "events": c.execute("SELECT COUNT(*) FROM events").fetchone()[0],
            "edges": c.execute("SELECT COUNT(*) FROM causal_edges").fetchone()[0],
            "windows": c.execute("SELECT COUNT(DISTINCT window_id) FROM events").fetchone()[0],
            "redactions": c.execute("SELECT COUNT(*) FROM redaction_log").fetchone()[0],
        }

    def max_blob_bytes(self) -> int:
        q = """SELECT MAX(m) FROM (
                 SELECT MAX(MAX(length(global_i8)), MAX(length(global_bq)), MAX(length(patch_bq)),
                            MAX(length(patch_i8))) AS m FROM event_vectors
                 UNION ALL SELECT MAX(length(body)) FROM events)"""
        return int(self.conn.execute(q).fetchone()[0] or 0)

    def time_range(self) -> tuple[Optional[float], Optional[float]]:
        r = self.conn.execute("SELECT MIN(ts), MAX(ts) FROM events").fetchone()
        return r[0], r[1]

    def size_bytes(self, checkpoint: bool = True) -> int:
        if self.path == ":memory:":
            pc = self.conn.execute("PRAGMA page_count").fetchone()[0]
            ps = self.conn.execute("PRAGMA page_size").fetchone()[0]
            return pc * ps
        if checkpoint:
            self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        total = 0
        for suffix in ("", "-wal", "-shm"):
            p = self.path + suffix
            if os.path.exists(p) and suffix != "-shm":
                total += os.path.getsize(p)
        return total


def _pack_concepts(c: dict) -> str:
    """{"red": 0.83, "chart": 1.0} -> "red=.83 chart" (weights >= 1 are implicit, 2 decimals)."""
    out = []
    for k, v in c.items():
        v = round(float(v), 2)
        out.append(k if v >= 1.0 else f"{k}={v:.2f}".replace("=0.", "=."))
    return " ".join(out)


def _unpack_concepts(v) -> dict:
    if isinstance(v, dict):  # rows written before the packed format
        return v
    out = {}
    for tok in v.split():
        k, _, w = tok.partition("=")
        out[k] = float(w) if w else 1.0
    return out
