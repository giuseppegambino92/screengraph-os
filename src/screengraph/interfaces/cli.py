"""screengraph CLI (Typer)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from screengraph import __version__

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="screengraph-os: zero-screenshot desktop memory + cross-app causal graph.")
console = Console()
_STATE: dict = {"home": None}


def _engine():
    from screengraph.config import Config
    from screengraph.engine import ScreenGraph

    cfg = Config(data_dir=Path(_STATE["home"]).expanduser()) if _STATE["home"] else Config()
    return ScreenGraph(cfg)


def _print_json(obj) -> None:
    console.print_json(json.dumps(obj, default=str))


@app.callback()
def main_callback(
    home: Optional[str] = typer.Option(None, "--home", envvar="SCREENGRAPH_HOME",
                                       help="Data directory (default ~/.screengraph)."),
):
    _STATE["home"] = home


@app.command()
def version():
    """Print the version."""
    console.print(f"screengraph-os {__version__}")


@app.command()
def simulate(seed: int = typer.Option(7, help="Simulator seed."),
             ticks: int = typer.Option(450, help="Window ticks to replay."),
             now: bool = typer.Option(True, help="Anchor the session so it ends now.")):
    """Replay the deterministic 60-minute multi-monitor developer session into the database."""
    from screengraph.capture.simulator import WorkspaceSimulator

    with _engine() as sg:
        sim = WorkspaceSimulator(seed=seed, n_ticks=ticks, t0=(time.time() - 3600) if now else 1_731_500_000.0)
        stored = dropped = 0
        t0 = time.perf_counter()
        for tk in sim.session():
            r = sg.ingest_frame(tk.frame)
            stored += r.stored
            dropped += not r.stored
        el = time.perf_counter() - t0
        st = sg.stats(checkpoint=True)
    console.print(f"[green]replayed {stored + dropped} ticks[/] in {el:.2f}s: {stored} stored, {dropped} dropped by the "
                  f"idle gate ({dropped / max(1, stored + dropped):.1%}); db {st['db_bytes'] / 1e6:.2f} MB, "
                  f"{st['edges']} causal edges")


@app.command()
def capture(interval: float = typer.Option(2.0, help="Seconds between polls."),
            duration: float = typer.Option(60.0, help="Total seconds to run (0 = forever)."),
            pixels: bool = typer.Option(False, help="Also grab window pixels (macOS + pyobjc; wiped after encoding)."),
            git_repo: Optional[str] = typer.Option(None, help="Watch a git working tree for edits."),
            clipboard: bool = typer.Option(True, help="Watch the clipboard.")):
    """Capture real windows (best-effort native hooks) + clipboard + git working tree."""
    from screengraph.capture.os_hooks import ClipboardWatcher, GitWatcher, NativeCapture

    cap = NativeCapture(with_pixels=pixels)
    clip = ClipboardWatcher() if clipboard else None
    git = GitWatcher(git_repo) if git_repo else None
    console.print(f"backend: [bold]{cap.backend}[/]  pixels: {pixels}  git: {git_repo or '-'}  (Ctrl-C to stop)")
    end = time.time() + duration if duration else None
    n_st = n_dr = 0
    with _engine() as sg:
        if clip:
            clip.poll()  # prime: never ingest whatever was on the clipboard before we started
        try:
            while end is None or time.time() < end:
                frames = cap.capture_frames()
                copied = clip.poll() if clip else None
                if copied and frames:
                    frames[0].clipboard_copied = copied  # attribute to the frontmost window
                if git:
                    frames += git.poll()
                for f in frames:
                    r = sg.ingest_frame(f)
                    n_st += r.stored
                    n_dr += not r.stored
                console.print(f"\r{time.strftime('%H:%M:%S')} windows={len(frames)} stored={n_st} dropped={n_dr}", end="")
                time.sleep(interval)
        except KeyboardInterrupt:
            pass
    console.print()


@app.command()
def ingest(app_name: str = typer.Option(..., "--app", help="Application name."),
           category: str = typer.Option("terminal", help="terminal|browser|ide|dashboard|chat|other"),
           text: Optional[str] = typer.Option(None, help="Text to ingest (default: read stdin)."),
           title: str = "", url: str = "", doc_path: str = "",
           clipboard: Optional[str] = typer.Option(None, help="Text copied from this window.")):
    """Ingest a hook event (shell command output, IDE save, note). Text is scrubbed before storage."""
    body = text if text is not None else sys.stdin.read()
    with _engine() as sg:
        r = sg.ingest_text_event(app_name, category, body, title, url, doc_path, clipboard=clipboard, source="cli")
    _print_json({"stored": r.stored, "event_id": r.event_id, "reason": r.reason, "redactions": r.redactions,
                 "edges": [{"from": e.src, "relation": e.relation, "score": e.score} for e in r.edges]})


@app.command()
def search(query: str, k: int = typer.Option(10, "-k"), as_json: bool = typer.Option(False, "--json")):
    """4-way hybrid search (BM25 + dense + visual Patch-MaxSim + causal graph, fused by RRF)."""
    with _engine() as sg:
        t0 = time.perf_counter()
        hits = sg.search(query, k)
        el = (time.perf_counter() - t0) * 1e3
        if as_json:
            _print_json([h.to_dict() for h in hits])
            return
        t = Table(title=f"{len(hits)} hits in {el:.2f} ms")
        for c in ("#", "app", "title", "channels", "snippet"):
            t.add_column(c, overflow="fold")
        for h in hits:
            t.add_row(str(h.event_id), h.event["app"], (h.event["title"] or "")[:48],
                      " ".join(f"{k}:{v}" for k, v in h.ranks.items()), h.snippet[:70])
        console.print(t)


@app.command()
def why(target: str, as_json: bool = typer.Option(False, "--json")):
    """Explain why a file/symbol changed: trace the causal graph back to its root cause."""
    with _engine() as sg:
        res = sg.why(target)
    if as_json or not res.get("found"):
        _print_json(res)
        return
    console.print(f"[bold magenta]why {target}?[/]  ({res['elapsed_ms']} ms)")
    for r in res["root_causes"][:1]:
        console.print(f"  [bold red]root cause[/] #{r['event_id']} {escape('[' + r['app'] + ']')} "
                      f"{escape(r['window_title'] or '')}")
        if r.get("snippet"):
            console.print(f"     [dim]{escape(r['snippet'])}[/]")
    console.print("  [bold]chain[/]")
    for line in res["narrative"]:
        console.print(f"    {escape(line)}")
    console.print("\n[dim]mermaid:[/]")
    console.print(res["mermaid"], markup=False, highlight=False)


@app.command()
def visual(query: str = typer.Argument(""), k: int = typer.Option(5, "-k"), as_json: bool = typer.Option(False, "--json"),
           like: Optional[int] = typer.Option(None, "--like", help="Query by example: windows that look like event N.")):
    """Visual search by colour / layout description, or by example (--like EVENT_ID), with patch localization."""
    with _engine() as sg:
        if like is not None:
            _print_json(sg.visual_search_like(like, k))
            return
        res = sg.visual_search(query, k)
    if as_json:
        _print_json(res)
        return
    t = Table(title=f"visual: {query}")
    for c in ("#", "app", "title", "patch", "quadrant", "maxsim", "concepts"):
        t.add_column(c, overflow="fold")
    for r in res:
        p = r["patch"]
        t.add_row(str(r["event_id"]), r["app"], (r["window_title"] or "")[:40], f"({p['row']},{p['col']})",
                  p["quadrant"], f"{r['maxsim']:.3f}", ", ".join(k for k in p["visual_concepts"] if "_" not in k))
    console.print(t)


@app.command("capture-live")
def capture_live(ticks: int = typer.Option(5, help="Desktop ticks to capture."),
                 interval: float = typer.Option(1.0, help="Seconds between ticks."),
                 pixels: bool = typer.Option(True, help="Grab window pixels into RAM (wiped after encoding)."),
                 allow_clipboard: bool = typer.Option(False, "--allow-clipboard", help="Read the clipboard (pbpaste)."),
                 request_ax: bool = typer.Option(False, "--request-ax", help="Show the macOS Accessibility prompt."),
                 keep: bool = typer.Option(False, help="Keep the DB (in --home, or a temp dir that is printed)."),
                 as_json: bool = typer.Option(False, "--json")):
    """Capture real macOS windows (ctypes CoreGraphics + AX) into a temporary DB and self-test the pipeline."""
    from screengraph.capture.live import run_live_capture

    data_dir = Path(_STATE["home"]).expanduser() if (keep and _STATE["home"]) else None
    if keep and data_dir is None:
        import tempfile

        data_dir = Path(tempfile.mkdtemp(prefix="screengraph-live-"))
    res = run_live_capture(ticks, interval, data_dir=data_dir, pixels=pixels, allow_clipboard=allow_clipboard,
                           request_ax=request_ax, keep=keep)
    if keep and data_dir is not None:
        res["data_dir"] = str(data_dir)
    if as_json or res.get("status") != "ok":
        _print_json(res)
        return
    p = res["permissions"]
    console.print(f"[bold]live capture[/] {res['ticks']} ticks · windows/tick {res['windows_per_tick']} · "
                  f"permissions: screen-recording={p['screen_recording']} accessibility={p['accessibility']}")
    console.print(f"  frames {res['frames_seen']} (pixels {res['frames_with_pixels']}, AX text {res['frames_with_ax_text']})"
                  f" -> stored {res['events_stored']}, dropped by gate {res['frames_dropped_by_gate']}")
    console.print(f"  OS capture/tick p50 {res['os_capture_ms_per_tick'].get('p50_ms')} ms · ingest p50/p95/max "
                  f"{res['ingest_ms'].get('p50_ms')}/{res['ingest_ms'].get('p95_ms')}/{res['ingest_ms'].get('max_ms')} ms"
                  f" (<10 ms: {res['ingest_under_10ms']})")
    ok = res["zero_pixel_guarantee_holds"]
    console.print(f"  zero-pixel audit: image files on disk {res['image_files_on_disk']}, pixel blobs "
                  f"{res['pixel_blob_suspects']}, frames non-zero after ingest {res['frames_nonzero_after_ingest']} -> "
                  f"[{'green' if ok else 'red'}]{'PASS' if ok else 'FAIL'}[/]")
    if keep:
        console.print(f"  kept DB in {res.get('data_dir')}")


@app.command()
def workspace(live: bool = typer.Option(False, help="Also enumerate native windows now.")):
    """Current semantic state of all windows (zero-screenshot context handoff)."""
    frames = None
    if live:
        from screengraph.capture.os_hooks import NativeCapture

        frames = NativeCapture(with_pixels=False, with_ax=False).capture_frames()
    with _engine() as sg:
        _print_json(sg.active_workspace(frames))


@app.command()
def timeline(minutes: Optional[float] = typer.Option(None, help="Only the last N minutes."),
             causal_only: bool = typer.Option(False, help="Only causally linked events."),
             mermaid: bool = typer.Option(False, help="Print only the Mermaid graph.")):
    """Chronological narrative + Mermaid causal graph of the session."""
    with _engine() as sg:
        lo, hi = sg.store.time_range()
        t0 = (hi - minutes * 60) if (minutes and hi) else None
        res = sg.timeline(t0, hi, causal_only)
    if mermaid:
        console.print(res["mermaid"], markup=False, highlight=False)
        return
    for n in res["narrative"]:
        console.print(f"{time.strftime('%H:%M:%S', time.localtime(n['ts']))}  {escape(n['line'])}")


@app.command("active-workspace")
def active_workspace(live: bool = typer.Option(False, help="Also enumerate native windows now.")):
    """Alias of `workspace`."""
    workspace(live)


@app.command("privacy-audit")
def privacy_audit_cmd():
    """Alias of `audit`."""
    audit()


@app.command()
def audit():
    """Privacy audit: zero on-disk pixels, RAM wipe ledger, redaction counters."""
    with _engine() as sg:
        _print_json(sg.privacy_audit())


@app.command()
def status():
    """One-shot Rich dashboard."""
    from screengraph.interfaces.tui import render_status

    with _engine() as sg:
        render_status(sg, console)


@app.command()
def dash(simulate: bool = typer.Option(False, help="Animate by replaying the simulator into the DB."),
         seconds: Optional[float] = typer.Option(None, help="Exit after N seconds.")):
    """Live Rich TUI dashboard."""
    from screengraph.interfaces.tui import run_dashboard

    with _engine() as sg:
        run_dashboard(sg, simulate=simulate, duration_s=seconds)


@app.command("eval")
def eval_cmd(scale: int = typer.Option(10000, help="Events for the at-scale latency run (0 = skip)."),
             sweep: int = typer.Option(10, help="Extra simulator seeds for the robustness sweep (0 = skip)."),
             live: int = typer.Option(0, help="Also capture N real desktop ticks (macOS)."),
             out: Optional[str] = typer.Option(None, help="Report path (default evals/latest_report.json).")):
    """Run the deterministic benchmark suite."""
    try:
        from evals import benchmark_suite
    except ImportError:
        root = Path(__file__).resolve().parents[3]
        sys.path.insert(0, str(root))
        try:
            from evals import benchmark_suite
        except ImportError:
            console.print("[red]evals/benchmark_suite.py not found; run from a source checkout[/]")
            raise typer.Exit(1)
    argv = ["--scale", str(scale), "--sweep", str(sweep), "--live", str(live)] + (["--out", out] if out else [])
    benchmark_suite.main(argv)


@app.command()
def mcp():
    """Serve the MCP tools over stdio (JSON-RPC 2.0)."""
    from screengraph.interfaces.mcp_server import serve

    serve(_STATE["home"])


def main() -> None:
    app()


if __name__ == "__main__":
    main()
