"""Model Context Protocol server (JSON-RPC 2.0 over newline-delimited stdio, or direct dispatch).

Tools:
  screengraph_active_workspace   zero-screenshot semantic state of every window + causal links
  screengraph_causal_why         "what caused me to write/change this?" (backward causal walk)
  screengraph_visual_search      Patch-MaxSim visual concept search with quadrant localization
  screengraph_timeline_replay    chronological narrative + Mermaid causal graph for a time range
  screengraph_ingest_event       hook/manual ingestion of terminal commands, IDE saves, notes
  screengraph_privacy_audit      zero-pixel proof, RAM wipe ledger, redaction counters
  screengraph_search             4-way hybrid RRF search (BM25 + dense + visual + graph)
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from datetime import datetime
from typing import Any, Callable, Optional, TextIO

from pydantic import BaseModel, Field, ValidationError

from screengraph import __version__

SUPPORTED_PROTOCOLS = ("2025-06-18", "2025-03-26", "2024-11-05")
SERVER_INFO = {"name": "screengraph-os", "version": __version__}


# ---------------------------------------------------------------------------
# Tool argument schemas
# ---------------------------------------------------------------------------
class ActiveWorkspaceArgs(BaseModel):
    live: bool = Field(False, description="Also enumerate currently visible native windows (metadata only, no pixels).")


class CausalWhyArgs(BaseModel):
    target: str = Field(..., description="File path, symbol, error string or code snippet to explain.")
    max_depth: int = Field(6, ge=1, le=12, description="Maximum backward hops.")


class VisualSearchArgs(BaseModel):
    query: str = Field(..., description="Visual description, e.g. 'red memory spike chart in top-right quadrant'.")
    k: int = Field(5, ge=1, le=50)


class TimelineArgs(BaseModel):
    t_start: Optional[str] = Field(None, description="Unix seconds or ISO-8601. Default: start of history.")
    t_end: Optional[str] = Field(None, description="Unix seconds or ISO-8601. Default: now/end of history.")
    last_minutes: Optional[float] = Field(None, gt=0, description="Shortcut: window ending at the latest event.")
    causal_only: bool = Field(False, description="Only events connected by evidence-bearing causal edges.")


class IngestArgs(BaseModel):
    app: str = Field(..., description="Application name, e.g. 'zsh', 'Code', 'Chrome'.")
    app_category: str = Field("other", description="terminal | browser | ide | dashboard | chat | other")
    text: str = Field(..., description="Visible text / command output / saved file excerpt. Scrubbed before storage.")
    title: str = ""
    url: str = ""
    doc_path: str = ""
    window_id: Optional[str] = None
    ts: Optional[float] = None
    clipboard: Optional[str] = Field(None, description="Text copied from this window (enables clipboard bridges).")


class PrivacyAuditArgs(BaseModel):
    pass


class SearchArgs(BaseModel):
    query: str
    k: int = Field(10, ge=1, le=100)
    app_category: Optional[str] = None


def _ts(v: Optional[str]) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return datetime.fromisoformat(v.replace("Z", "+00:00")).timestamp()


TOOLS: dict[str, tuple[type[BaseModel], str]] = {
    "screengraph_active_workspace": (ActiveWorkspaceArgs,
        "Zero-screenshot context handoff: the current semantic state of every open window across monitors "
        "(app, title, URL/document, bounding box, latest visible text summary, entities) plus active causal links."),
    "screengraph_causal_why": (CausalWhyArgs,
        "Cognitive time-travel debugging: given a file path, symbol or snippet, walk the cross-app causal graph "
        "backwards and return the root cause (terminal error, chat message, dashboard anomaly, docs page) and the "
        "full evidence chain that led to the edit."),
    "screengraph_visual_search": (VisualSearchArgs,
        "Search past windows by visual concept, colour, chart shape or layout using late-interaction Patch-MaxSim; "
        "returns the window event plus the exact spatial patch (row, col, quadrant, screen bbox)."),
    "screengraph_timeline_replay": (TimelineArgs,
        "Reconstruct a chronological narrative and a Mermaid causal graph of a work session between two times."),
    "screengraph_ingest_event": (IngestArgs,
        "Ingest a terminal command, IDE save, note or simulated viewport state. Text is PII-scrubbed, embedded and "
        "linked into the causal graph."),
    "screengraph_privacy_audit": (PrivacyAuditArgs,
        "Proof of zero on-disk pixels (magic-byte scan, blob bounds), RAM-wipe hash-chain ledger and PII/secret "
        "redaction counters."),
    "screengraph_search": (SearchArgs,
        "Hybrid 4-way search (FTS5 BM25 + dense Matryoshka + visual Patch-MaxSim + causal graph walk, fused by RRF)."),
}


def tool_definitions() -> list[dict]:
    out = []
    for name, (model, desc) in TOOLS.items():
        schema = model.model_json_schema()
        schema.pop("title", None)
        out.append({"name": name, "description": desc, "inputSchema": schema})
    return out


# ---------------------------------------------------------------------------
class MCPServer:
    def __init__(self, engine_factory: Callable[[], Any]):
        self._factory = engine_factory
        self._engine = None
        self.initialized = False

    @property
    def engine(self):
        if self._engine is None:
            self._engine = self._factory()
        return self._engine

    # -- tool implementations -------------------------------------------------
    def call_tool(self, name: str, args: dict) -> dict:
        model, _ = TOOLS[name]
        a = model.model_validate(args or {})
        sg = self.engine
        if name != "screengraph_ingest_event":
            sg.refresh()
        if name == "screengraph_active_workspace":
            live = None
            if a.live:
                from screengraph.capture.os_hooks import NativeCapture

                cap = NativeCapture(with_pixels=False, with_ax=False)
                live = cap.capture_frames()
            return sg.active_workspace(live)
        if name == "screengraph_causal_why":
            return sg.retriever.why(a.target, max_depth=a.max_depth)
        if name == "screengraph_visual_search":
            return {"query": a.query, "results": sg.visual_search(a.query, a.k)}
        if name == "screengraph_timeline_replay":
            t0, t1 = _ts(a.t_start), _ts(a.t_end)
            if a.last_minutes:
                _, hi = sg.store.time_range()
                t1 = t1 or hi or time.time()
                t0 = t1 - a.last_minutes * 60
            return sg.timeline(t0, t1, a.causal_only)
        if name == "screengraph_ingest_event":
            r = sg.ingest_text_event(a.app, a.app_category, a.text, a.title, a.url, a.doc_path, a.window_id,
                                     a.ts, a.clipboard, source="mcp")
            return {"stored": r.stored, "reason": r.reason, "event_id": r.event_id, "redactions": r.redactions,
                    "edges": [{"from": e.src, "to": e.dst, "relation": e.relation, "score": e.score,
                               "evidence": e.evidence} for e in r.edges],
                    "timings_ms": {k: round(v, 3) for k, v in r.timings_ms.items()}}
        if name == "screengraph_privacy_audit":
            return sg.privacy_audit()
        if name == "screengraph_search":
            return {"query": a.query, "results": [h.to_dict() for h in sg.search(a.query, a.k, app_category=a.app_category)]}
        raise KeyError(name)

    # -- JSON-RPC ---------------------------------------------------------------
    @staticmethod
    def _error(mid, code: int, message: str, data=None) -> dict:
        err = {"code": code, "message": message}
        if data is not None:
            err["data"] = data
        return {"jsonrpc": "2.0", "id": mid, "error": err}

    def handle(self, msg: Any) -> Optional[dict]:
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0" or not isinstance(msg.get("method"), str):
            return self._error(msg.get("id") if isinstance(msg, dict) else None, -32600, "Invalid Request")
        method, mid, params = msg["method"], msg.get("id"), msg.get("params") or {}
        is_notification = "id" not in msg
        try:
            if method == "initialize":
                req = params.get("protocolVersion")
                ver = req if req in SUPPORTED_PROTOCOLS else SUPPORTED_PROTOCOLS[0]
                result = {"protocolVersion": ver, "capabilities": {"tools": {"listChanged": False}},
                          "serverInfo": SERVER_INFO,
                          "instructions": "Local desktop memory with a cross-app causal graph. Use "
                                          "screengraph_causal_why to explain why code changed, "
                                          "screengraph_active_workspace for current context."}
            elif method in ("notifications/initialized", "initialized"):
                self.initialized = True
                return None
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": tool_definitions()}
            elif method == "tools/call":
                name = params.get("name")
                if name not in TOOLS:
                    return self._error(mid, -32602, f"Unknown tool: {name}")
                try:
                    data = self.call_tool(name, params.get("arguments") or {})
                    result = {"content": [{"type": "text", "text": json.dumps(data, default=str, indent=1)}],
                              "structuredContent": data, "isError": False}
                except ValidationError as ve:
                    result = {"content": [{"type": "text", "text": f"Invalid arguments: {ve.errors(include_url=False)}"}],
                              "isError": True}
                except Exception as exc:  # tool errors are reported in-band per MCP
                    result = {"content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}], "isError": True}
            elif method in ("resources/list",):
                result = {"resources": []}
            elif method in ("prompts/list",):
                result = {"prompts": []}
            elif method.startswith("notifications/"):
                return None
            else:
                return None if is_notification else self._error(mid, -32601, f"Method not found: {method}")
        except Exception as exc:  # pragma: no cover - defensive
            return self._error(mid, -32603, "Internal error", {"detail": str(exc), "trace": traceback.format_exc(limit=3)})
        return None if is_notification else {"jsonrpc": "2.0", "id": mid, "result": result}

    def handle_raw(self, line: str) -> Optional[str]:
        line = line.strip()
        if not line:
            return None
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            return json.dumps(self._error(None, -32700, "Parse error"))
        if isinstance(msg, list):  # JSON-RPC batch
            out = [r for r in (self.handle(m) for m in msg) if r is not None]
            return json.dumps(out, default=str) if out else None
        r = self.handle(msg)
        return json.dumps(r, default=str) if r is not None else None

    def serve_stdio(self, stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> None:
        for line in stdin:
            resp = self.handle_raw(line)
            if resp is not None:
                stdout.write(resp + "\n")
                stdout.flush()
        if self._engine is not None:
            self._engine.close()


def serve(home: Optional[str] = None) -> None:
    from screengraph.config import Config
    from screengraph.engine import ScreenGraph

    def factory():
        cfg = Config() if home is None else Config(data_dir=__import__("pathlib").Path(home).expanduser())
        return ScreenGraph(cfg)

    MCPServer(factory).serve_stdio()


if __name__ == "__main__":
    serve()
