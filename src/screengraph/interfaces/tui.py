"""Rich terminal dashboard: multi-monitor viewport map, live causal chains, storage footprint, privacy counters."""

from __future__ import annotations

import time
from typing import Optional

from rich import box
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from screengraph import __version__
from screengraph.i18n import t

CAT_STYLE = {"terminal": "green", "browser": "blue", "ide": "magenta", "dashboard": "yellow", "chat": "cyan",
             "other": "white"}
REL_STYLE = {"CLIPBOARD_TRANSFER": "bold cyan", "ERROR_TO_RESEARCH": "red", "RESEARCH_TO_CODE_EDIT": "magenta",
             "ERROR_TO_CODE_EDIT": "bold red", "UI_TO_LOG_CORRELATION": "yellow", "LINK_FOLLOW": "blue",
             "SEMANTIC_CONTINUATION": "dim"}
LEGACY_MB_PER_HOUR = 44_949 / 240  # benchmark legacy model (see evals/benchmark_suite.py)


def _ago(ts: Optional[float], now: float) -> str:
    if ts is None:
        return "-"
    d = max(0.0, now - ts)
    return f"{d:.0f}s" if d < 90 else f"{d / 60:.0f}m" if d < 5400 else f"{d / 3600:.1f}h"


def monitor_map(windows: list[dict], width: int = 76) -> Text:
    """ASCII map of monitors and window viewports (scaled bounding boxes)."""
    if not windows:
        return Text(t("tui.no_viewports"), style="dim")
    xs = [w["bbox"]["x"] for w in windows] + [w["bbox"]["x"] + w["bbox"]["width"] for w in windows]
    ys = [w["bbox"]["y"] for w in windows] + [w["bbox"]["y"] + w["bbox"]["height"] for w in windows]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    sx = width / max(1, x1 - x0)
    rows = max(5, min(9, int((y1 - y0) * sx / 2.2)))
    sy = rows / max(1, y1 - y0)
    grid = [[" "] * width for _ in range(rows)]
    styles = [[""] * width for _ in range(rows)]
    for w in windows:
        b = w["bbox"]
        c0, c1 = int((b["x"] - x0) * sx), min(width - 1, int((b["x"] + b["width"] - x0) * sx) - 1)
        r0, r1 = int((b["y"] - y0) * sy), min(rows - 1, int((b["y"] + b["height"] - y0) * sy) - 1)
        st = CAT_STYLE.get(w.get("app_category", "other"), "white")
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                edge = r in (r0, r1) or c in (c0, c1)
                grid[r][c] = ("+" if r in (r0, r1) and c in (c0, c1) else "-" if r in (r0, r1) else "|") if edge else " "
                styles[r][c] = st
        label = f"{w['app'][:max(1, c1 - c0 - 1)]}"
        mid = (r0 + r1) // 2
        for i, ch in enumerate(label[: max(0, c1 - c0 - 1)]):
            grid[mid][c0 + 1 + i] = ch
            styles[mid][c0 + 1 + i] = "bold " + st
    out = Text()
    for r in range(rows):
        for c in range(width):
            out.append(grid[r][c], style=styles[r][c] or None)
        out.append("\n")
    mons = sorted({w["bbox"]["monitor_id"] for w in windows})
    out.append(t("tui.map_footer", monitors=len(mons), viewports=len(windows)), style="dim")
    return out


def viewport_table(windows: list[dict], now: float) -> Table:
    tbl = Table(box=box.SIMPLE_HEAD, expand=True, pad_edge=False)
    for col in ("tui.col.mon", "tui.col.app", "tui.col.doc", "tui.col.bbox", "tui.col.changed"):
        tbl.add_column(t(col), overflow="ellipsis", no_wrap=True)
    for w in windows:
        b = w["bbox"]
        st = CAT_STYLE.get(w["app_category"], "white")
        doc = w.get("doc_path") or w.get("url") or w["title"]
        tbl.add_row(str(b["monitor_id"]), Text(w["app"], style=st), doc,
                    f"{b['width']}x{b['height']}@{b['x']},{b['y']}", _ago(w["last_change_ts"], now))
    return tbl


def causal_panel(sg, max_chains: int = 3) -> Group:
    edits = sg.store.conn.execute(
        "SELECT id, doc_path FROM events WHERE app_category='ide' AND doc_path != '' ORDER BY ts DESC LIMIT 20"
    ).fetchall()
    trees, seen = [], set()
    for row in edits:
        if row["doc_path"] in seen:
            continue
        seen.add(row["doc_path"])
        tr = sg.graph.trace_back([row["id"]])
        if not tr.roots:
            continue
        evs = sg.store.get_events(set(tr.chain) | {row["id"]})
        root = tr.roots[0][0]
        tree = Tree(Text(t("tui.why", doc=row["doc_path"]), style="bold magenta"))
        for nid in tr.chain:
            e = evs.get(nid)
            if not e:
                continue
            inc = [ed for ed in tr.edges if ed.dst == nid]
            rel = max(inc, key=lambda x: x.score).relation if inc else t("tui.root_cause")
            style = "bold red" if nid == root else REL_STYLE.get(rel, "white")
            tree.add(Text.assemble((f"{rel:<22}", style), (f" #{nid} ", "dim"),
                                   (f"{e['app']}: ", CAT_STYLE.get(e['app_category'], "white")),
                                   ((e['title'] or '')[:46], "")))
        trees.append(tree)
        if len(trees) >= max_chains:
            break
    if not trees:
        return Group(Text(t("tui.no_chains"), style="dim"))
    return Group(*trees)


def storage_panel(sg) -> Table:
    st = sg.stats()
    hours = max(st["span_s"] / 3600, 1e-6)
    mb = st["db_bytes"] / 1e6
    mb_h = mb / hours
    tbl = Table.grid(padding=(0, 2))
    tbl.add_column(style="dim")
    tbl.add_column(justify="right")
    tbl.add_row(t("tui.events_edges"), f"{st['events']} / {st['edges']}")
    tbl.add_row(t("tui.database"), f"{mb:.2f} MB")
    tbl.add_row(t("tui.footprint"), t("tui.kb_hour", kb=mb_h * 1000))
    tbl.add_row(t("tui.projected"), f"{mb_h * 240:.0f} MB")
    tbl.add_row(t("tui.video_model"), t("tui.gb_hour", gb=LEGACY_MB_PER_HOUR / 1000))
    tbl.add_row(t("tui.reduction"), Text(f"{100 * (1 - mb_h / LEGACY_MB_PER_HOUR):.2f}%", style="bold green"))
    return tbl


def privacy_panel(sg) -> Table:
    led = sg.ledger.snapshot()
    reds = sg.store.redaction_counts()
    tbl = Table.grid(padding=(0, 2))
    tbl.add_column(style="dim")
    tbl.add_column(justify="right")
    c = sg.counters
    tbl.add_row(t("tui.frames"), f"{c['ticks']} / {c['dropped']}")
    tbl.add_row(t("tui.buffers_zeroed"), f"{led['buffers_zeroed']}")
    tbl.add_row(t("tui.bytes_zeroed"), f"{led['bytes_zeroed'] / 1e6:.1f} MB")
    tbl.add_row(t("tui.wipe_failures"), Text(str(led["verification_failures"]),
                                             style="bold green" if led["verification_failures"] == 0 else "bold red"))
    tbl.add_row(t("tui.pixels_on_disk"), Text(t("tui.zero_files"), style="bold green"))
    for k, v in sorted(reds.items(), key=lambda x: -x[1])[:7]:
        tbl.add_row(t("tui.redacted", kind=k), str(v))
    return tbl


def build_layout(sg, width: int = 120) -> Layout:
    now = time.time()
    _, hi = sg.store.time_range()
    ref = hi if hi is not None and now - hi > 3600 else now  # replayed/simulated data: relative to last event
    ws = sg.active_workspace()["windows"]
    lay = Layout()
    lay.split_column(Layout(name="head", size=3), Layout(name="top", size=13), Layout(name="mid"), Layout(name="bot", size=12))
    lay["head"].update(Panel(Text.assemble(("screengraph-os ", "bold"), (f"v{__version__}", "dim"),
                                           ("  ·  " + t("tui.tagline"), "italic")),
                             box=box.HEAVY))
    map_w = max(24, int(width * 3 / 5) - 4)
    lay["top"].split_row(Layout(Panel(monitor_map(ws, map_w), title=t("tui.panel.viewports"), border_style="blue"), ratio=3),
                         Layout(Panel(storage_panel(sg), title=t("tui.panel.storage"), border_style="green"), ratio=2))
    lay["mid"].update(Panel(causal_panel(sg), title=t("tui.panel.chains"), border_style="magenta"))
    lay["bot"].split_row(Layout(Panel(viewport_table(ws, ref), title=t("tui.panel.windows"), border_style="cyan"), ratio=3),
                         Layout(Panel(privacy_panel(sg), title=t("tui.panel.privacy"), border_style="red"), ratio=2))
    return lay


def render_status(sg, console: Optional[Console] = None) -> None:
    console = console or Console()
    console.print(build_layout(sg, console.width), height=max(40, console.height or 40))


def run_dashboard(sg, simulate: bool = False, refresh_s: float = 0.5, duration_s: Optional[float] = None,
                  seed: int = 7) -> None:
    console = Console()
    ticks = None
    if simulate:
        from screengraph.capture.simulator import WorkspaceSimulator

        ticks = iter(WorkspaceSimulator(seed=seed, t0=time.time() - 3600).session())
    t_end = time.time() + duration_s if duration_s else None
    with Live(build_layout(sg, console.width), console=console, refresh_per_second=4, screen=True) as live:
        try:
            while t_end is None or time.time() < t_end:
                if ticks is not None:
                    for _ in range(6):
                        tk = next(ticks, None)
                        if tk is None:
                            ticks = None
                            break
                        sg.ingest_frame(tk.frame)
                else:
                    sg.refresh()
                live.update(build_layout(sg, console.width))
                time.sleep(refresh_s)
        except KeyboardInterrupt:
            pass
