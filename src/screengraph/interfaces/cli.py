"""screengraph CLI (Typer)."""

from __future__ import annotations

import inspect
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table
from typer.models import ArgumentInfo, CommandInfo, OptionInfo

from screengraph import __version__, i18n
from screengraph.i18n import t

app = typer.Typer(add_completion=False, no_args_is_help=True, help=t("help.app"))
console = Console()
_STATE: dict = {"home": None}


def _engine():
    from screengraph.config import Config
    from screengraph.engine import ScreenGraph

    cfg = Config(data_dir=Path(_STATE["home"]).expanduser()) if _STATE["home"] else Config()
    return ScreenGraph(cfg)


def _print_json(obj) -> None:
    console.print_json(json.dumps(obj, default=str))


def _check_lang(value: str | None) -> str | None:
    if value is not None and i18n.normalize(value) is None:
        raise typer.BadParameter(t("error.bad_lang", value=value, choices=", ".join(i18n.SUPPORTED)))
    return value


@app.callback()
def main_callback(
    home: Optional[str] = typer.Option(None, "--home", envvar="SCREENGRAPH_HOME",
                                       help=t("help.app.home")),
    lang: str | None = typer.Option(None, "--lang", metavar="|".join(i18n.SUPPORTED), callback=_check_lang,
                                    help=t("help.app.lang")),
):
    _STATE["home"] = home
    i18n.set_lang(lang)  # None -> $SCREENGRAPH_LANG, else English


@app.command(help=t("help.version"))
def version():
    console.print(f"screengraph-os {__version__}")


@app.command(help=t("help.simulate"))
def simulate(seed: int = typer.Option(7, help=t("help.simulate.seed")),
             ticks: int = typer.Option(450, help=t("help.simulate.ticks")),
             now: bool = typer.Option(True, help=t("help.simulate.now"))):
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
    detail = t("simulate.detail", seconds=el, stored=stored, dropped=dropped,
               ratio=dropped / max(1, stored + dropped), mb=st["db_bytes"] / 1e6, edges=st["edges"])
    console.print(f"[green]{t('simulate.done', total=stored + dropped)}[/] {detail}")


@app.command(help=t("help.capture"))
def capture(interval: float = typer.Option(2.0, help=t("help.capture.interval")),
            duration: float = typer.Option(60.0, help=t("help.capture.duration")),
            pixels: bool = typer.Option(False, help=t("help.capture.pixels")),
            git_repo: Optional[str] = typer.Option(None, help=t("help.capture.git_repo")),
            clipboard: bool = typer.Option(True, help=t("help.capture.clipboard"))):
    from screengraph.capture.os_hooks import ClipboardWatcher, GitWatcher, NativeCapture

    cap = NativeCapture(with_pixels=pixels)
    clip = ClipboardWatcher() if clipboard else None
    git = GitWatcher(git_repo) if git_repo else None
    console.print(t("capture.banner", backend=escape(str(cap.backend)), pixels=pixels, git=escape(git_repo or "-")))
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
                progress = t("capture.progress", windows=len(frames), stored=n_st, dropped=n_dr)
                console.print(f"\r{time.strftime('%H:%M:%S')} {progress}", end="")
                time.sleep(interval)
        except KeyboardInterrupt:
            pass
    console.print()


@app.command(help=t("help.ingest"))
def ingest(app_name: str = typer.Option(..., "--app", help=t("help.ingest.app_name")),
           category: str = typer.Option("terminal", help="terminal|browser|ide|dashboard|chat|other"),
           text: Optional[str] = typer.Option(None, help=t("help.ingest.text")),
           title: str = "", url: str = "", doc_path: str = "",
           clipboard: Optional[str] = typer.Option(None, help=t("help.ingest.clipboard"))):
    body = text if text is not None else sys.stdin.read()
    with _engine() as sg:
        r = sg.ingest_text_event(app_name, category, body, title, url, doc_path, clipboard=clipboard, source="cli")
    _print_json({"stored": r.stored, "event_id": r.event_id, "reason": r.reason, "redactions": r.redactions,
                 "edges": [{"from": e.src, "relation": e.relation, "score": e.score} for e in r.edges]})


@app.command(help=t("help.search"))
def search(query: str, k: int = typer.Option(10, "-k"), as_json: bool = typer.Option(False, "--json")):
    with _engine() as sg:
        t0 = time.perf_counter()
        hits = sg.search(query, k)
        el = (time.perf_counter() - t0) * 1e3
        if as_json:
            _print_json([h.to_dict() for h in hits])
            return
        tbl = Table(title=t("search.title", n=len(hits), ms=el))
        for c in ("#", t("col.app"), t("col.title"), t("col.channels"), t("col.snippet")):
            tbl.add_column(c, overflow="fold")
        for h in hits:
            tbl.add_row(str(h.event_id), h.event["app"], (h.event["title"] or "")[:48],
                        " ".join(f"{k}:{v}" for k, v in h.ranks.items()), h.snippet[:70])
        console.print(tbl)


@app.command(help=t("help.why"))
def why(target: str, as_json: bool = typer.Option(False, "--json")):
    with _engine() as sg:
        res = sg.why(target)
    if as_json or not res.get("found"):
        _print_json(res)
        return
    console.print(f"[bold magenta]{escape(t('why.header', target=target))}[/]  ({res['elapsed_ms']} ms)")
    for r in res["root_causes"][:1]:
        console.print(f"  [bold red]{t('why.root_cause')}[/] #{r['event_id']} {escape('[' + r['app'] + ']')} "
                      f"{escape(r['window_title'] or '')}")
        if r.get("snippet"):
            console.print(f"     [dim]{escape(r['snippet'])}[/]")
    console.print(f"  [bold]{t('why.chain')}[/]")
    for line in res["narrative"]:
        console.print(f"    {escape(line)}")
    console.print("\n[dim]mermaid:[/]")
    console.print(res["mermaid"], markup=False, highlight=False)


@app.command(help=t("help.visual"))
def visual(query: str = typer.Argument(""), k: int = typer.Option(5, "-k"), as_json: bool = typer.Option(False, "--json"),
           like: Optional[int] = typer.Option(None, "--like", help=t("help.visual.like"))):
    with _engine() as sg:
        if like is not None:
            _print_json(sg.visual_search_like(like, k))
            return
        res = sg.visual_search(query, k)
    if as_json:
        _print_json(res)
        return
    tbl = Table(title=escape(t("visual.title", query=query)))
    for c in ("#", t("col.app"), t("col.title"), t("col.patch"), t("col.quadrant"), "maxsim", t("col.concepts")):
        tbl.add_column(c, overflow="fold")
    for r in res:
        p = r["patch"]
        tbl.add_row(str(r["event_id"]), r["app"], (r["window_title"] or "")[:40], f"({p['row']},{p['col']})",
                    p["quadrant"], f"{r['maxsim']:.3f}", ", ".join(k for k in p["visual_concepts"] if "_" not in k))
    console.print(tbl)


@app.command("capture-live", help=t("help.capture-live"))
def capture_live(ticks: int = typer.Option(5, help=t("help.capture-live.ticks")),
                 interval: float = typer.Option(1.0, help=t("help.capture-live.interval")),
                 pixels: bool = typer.Option(True, help=t("help.capture-live.pixels")),
                 allow_clipboard: bool = typer.Option(False, "--allow-clipboard",
                                                      help=t("help.capture-live.allow_clipboard")),
                 request_ax: bool = typer.Option(False, "--request-ax", help=t("help.capture-live.request_ax")),
                 keep: bool = typer.Option(False, help=t("help.capture-live.keep")),
                 as_json: bool = typer.Option(False, "--json")):
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
    summary = t("live.summary", ticks=res["ticks"], per_tick=res["windows_per_tick"],
                screen=p["screen_recording"], ax=p["accessibility"])
    console.print(f"[bold]{t('live.header')}[/] {escape(summary)}")
    console.print("  " + escape(t("live.frames", seen=res["frames_seen"], pixels=res["frames_with_pixels"],
                                  ax=res["frames_with_ax_text"], stored=res["events_stored"],
                                  dropped=res["frames_dropped_by_gate"])))
    ing = res["ingest_ms"]
    console.print("  " + escape(t("live.latency", os_p50=res["os_capture_ms_per_tick"].get("p50_ms"),
                                  p50=ing.get("p50_ms"), p95=ing.get("p95_ms"), max=ing.get("max_ms"),
                                  under=res["ingest_under_10ms"])))
    ok = res["zero_pixel_guarantee_holds"]
    audit_line = t("live.audit", images=res["image_files_on_disk"], blobs=res["pixel_blob_suspects"],
                   nonzero=res["frames_nonzero_after_ingest"])
    console.print(f"  {escape(audit_line)} [{'green' if ok else 'red'}]{'PASS' if ok else 'FAIL'}[/]")
    if keep:
        console.print("  " + escape(t("live.kept", path=res.get("data_dir"))))


@app.command(help=t("help.workspace"))
def workspace(live: bool = typer.Option(False, help=t("help.workspace.live"))):
    frames = None
    if live:
        from screengraph.capture.os_hooks import NativeCapture

        frames = NativeCapture(with_pixels=False, with_ax=False).capture_frames()
    with _engine() as sg:
        _print_json(sg.active_workspace(frames))


@app.command(help=t("help.timeline"))
def timeline(minutes: Optional[float] = typer.Option(None, help=t("help.timeline.minutes")),
             causal_only: bool = typer.Option(False, help=t("help.timeline.causal_only")),
             mermaid: bool = typer.Option(False, help=t("help.timeline.mermaid"))):
    with _engine() as sg:
        lo, hi = sg.store.time_range()
        t0 = (hi - minutes * 60) if (minutes and hi) else None
        res = sg.timeline(t0, hi, causal_only)
    if mermaid:
        console.print(res["mermaid"], markup=False, highlight=False)
        return
    for n in res["narrative"]:
        console.print(f"{time.strftime('%H:%M:%S', time.localtime(n['ts']))}  {escape(n['line'])}")


@app.command("active-workspace", help=t("help.active-workspace"))
def active_workspace(live: bool = typer.Option(False, help=t("help.active-workspace.live"))):
    workspace(live)


@app.command("privacy-audit", help=t("help.privacy-audit"))
def privacy_audit_cmd():
    audit()


@app.command(help=t("help.audit"))
def audit():
    with _engine() as sg:
        _print_json(sg.privacy_audit())


@app.command(help=t("help.status"))
def status():
    from screengraph.interfaces.tui import render_status

    with _engine() as sg:
        render_status(sg, console)


@app.command(help=t("help.dash"))
def dash(simulate: bool = typer.Option(False, help=t("help.dash.simulate")),
         seconds: Optional[float] = typer.Option(None, help=t("help.dash.seconds"))):
    from screengraph.interfaces.tui import run_dashboard

    with _engine() as sg:
        run_dashboard(sg, simulate=simulate, duration_s=seconds)


@app.command("eval", help=t("help.eval"))
def eval_cmd(scale: int = typer.Option(10000, help=t("help.eval.scale")),
             sweep: int = typer.Option(10, help=t("help.eval.sweep")),
             live: int = typer.Option(0, help=t("help.eval.live")),
             out: Optional[str] = typer.Option(None, help=t("help.eval.out"))):
    try:
        from evals import benchmark_suite
    except ImportError:
        root = Path(__file__).resolve().parents[3]
        sys.path.insert(0, str(root))
        try:
            from evals import benchmark_suite
        except ImportError:
            console.print(f"[red]{escape(t('eval.missing'))}[/]")
            raise typer.Exit(1)
    argv = ["--scale", str(scale), "--sweep", str(sweep), "--live", str(live)] + (["--out", out] if out else [])
    benchmark_suite.main(argv)


@app.command(help=t("help.mcp"))
def mcp():
    from screengraph.interfaces.mcp_server import serve

    serve(_STATE["home"])


def _argv_lang(argv: list[str]) -> str | None:
    """The global ``--lang`` value, if given before the sub-command (so that ``--help`` is localized too)."""
    it = iter(argv)
    for tok in it:
        if tok == "--lang":
            return next(it, None)
        if tok.startswith("--lang="):
            return tok.split("=", 1)[1]
        if tok == "--home":
            next(it, None)
        elif not tok.startswith("-"):
            break
    return None


def localize_help() -> None:
    """Re-resolve every Typer help text from the catalog in the active language.

    Typer builds the Click command from these objects on each ``app()`` call, so this runs right before it.
    Help keys follow ``help.<command>`` and ``help.<command>.<parameter>`` (``app`` for the global options).
    """
    app.info.help = t("help.app")
    targets: list[tuple[str, CommandInfo | None, Callable[..., Any]]] = []
    if app.registered_callback is not None and app.registered_callback.callback is not None:
        targets.append(("app", None, app.registered_callback.callback))
    for ci in app.registered_commands:
        if ci.callback is not None:
            targets.append((ci.name or typer.main.get_command_name(ci.callback.__name__), ci, ci.callback))
    for name, info, callback in targets:
        if info is not None and i18n.has(f"help.{name}"):
            info.help = t(f"help.{name}")
        for pname, param in inspect.signature(callback).parameters.items():
            key = f"help.{name}.{pname}"
            if i18n.has(key) and isinstance(param.default, (OptionInfo, ArgumentInfo)):
                param.default.help = t(key)


def main() -> None:
    i18n.set_lang(i18n.normalize(_argv_lang(sys.argv[1:])))  # an invalid value is left for Click to reject
    localize_help()
    app()


if __name__ == "__main__":
    main()
