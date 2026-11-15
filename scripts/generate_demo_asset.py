"""Render the README terminal screenshots (English + Italian) from REAL engine / CLI output.

Everything runs on the deterministic workspace simulator (seed 7, 450 ticks) in temporary directories; nothing is
typed in by hand, no model is downloaded and the real screen is never captured.

* ``demo.svg``   - the README header: the active workspace, a visual patch query and two causal ``why`` traces,
  recorded from direct engine calls. Storage/legacy figures in the footer are read from evals/latest_report.json
  (produced by ``python -m evals.benchmark_suite``).
* ``why.svg``, ``search.svg``, ``status.svg`` - the actual ``screengraph`` CLI commands (Typer app invoked
  in-process with ``--lang``), with the CLI console swapped for a recording Rich console.

Output: ``docs/images/<lang>/<name>.svg`` for every language in ``screengraph.i18n.SUPPORTED``.

    python scripts/generate_demo_asset.py                 # all screenshots, en + it
    python scripts/generate_demo_asset.py --lang it       # only the Italian set
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from rich import box
from rich.console import Console, Group
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.terminal_theme import MONOKAI
from rich.text import Text

from screengraph import __version__, i18n
from screengraph.capture.simulator import WorkspaceSimulator
from screengraph.config import Config
from screengraph.engine import ScreenGraph
from screengraph.interfaces import cli

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "images"
CAT_STYLE = {"terminal": "green", "browser": "cyan", "ide": "blue", "dashboard": "magenta", "chat": "yellow"}
VISUAL_QUERY = "red memory spike chart in top-right quadrant"  # engine input: concept words stay English


def clip(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[: n - 1] + "…"


def recording_console(width: int, height: int | None = None) -> Console:
    return Console(record=True, width=width, height=height, force_terminal=True, color_system="truecolor",
                   file=io.StringIO())


def lang_flag(lang: str) -> str:
    return "" if lang == i18n.DEFAULT_LANG else f"--lang {lang} "


def prompt(cmd: str) -> Text:
    return Text.assemble(("❯ ", "bold green"), ("screengraph ", "bold white"), (cmd, "white"))


def save(con: Console, out: Path, title: str) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(con.export_svg(title=title, theme=MONOKAI), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------------------------- header demo
def why_tree(res: dict, target: str, lang: str) -> Group:
    """Root cause, then one line per causal hop: relation, edge score S, evidence, destination window."""
    pre, post = i18n.t("why.header", lang=lang, target="\0").split("\0")
    lines = [Text.assemble((pre, "bold magenta"), (target, "bold white"), (post + "  ", "bold magenta"),
                           (i18n.t("demo.traced", lang=lang, ms=res["elapsed_ms"]), "dim"))]
    root = res["root_causes"][0]
    lines.append(Text.assemble((f"  {i18n.t('why.root_cause', lang=lang)} ", "bold red"),
                               (f"#{root['event_id']} ", "red"),
                               (f"[{root['app']}] ", CAT_STYLE.get(root["app_category"], "white")),
                               (clip(root["window_title"], 40), "white"), ("  ", ""),
                               (clip((root.get("snippet") or "").strip(), 44), "grey50")))
    for c in res["chain"][1:]:
        via = max(c["via"], key=lambda v: v["score"]) if c["via"] else None
        t = Text("    ↳ ", "grey50")
        if via:
            ev = [e.split(":", 1)[-1] for e in via["evidence"]]
            evs = clip(", ".join(e for e in ev if not e.startswith(("xmodal_cos", "energy"))), 26)
            t.append(f"{via['relation']:<27}", "bold yellow")
            t.append(f"S={via['score']:.2f}  ", "grey70")
            t.append(f"{evs:<27}", "grey50")
        app = f"[{c['app']}] "
        t.append(app, CAT_STYLE.get(c["app_category"], "white"))
        t.append(clip(c["window_title"], 104 - 6 - 27 - 8 - 27 - len(app)), "white")
        lines.append(t)
    return Group(*lines)


def render_demo(lang: str, out: Path) -> Path:
    def tr(key: str, **kw: object) -> str:
        return i18n.t(key, lang=lang, **kw)

    flag = lang_flag(lang)
    report = json.loads((ROOT / "evals" / "latest_report.json").read_text())
    tmp = Path(tempfile.mkdtemp(prefix="screengraph-demo-"))
    con = recording_console(108)
    try:
        sg = ScreenGraph(Config.ephemeral(tmp))
        sim = WorkspaceSimulator(seed=7)
        t0 = time.perf_counter()
        stored = ticks = 0
        for tk in sim.session():
            ticks += 1
            stored += sg.ingest_frame(tk.frame).stored
        wall = time.perf_counter() - t0

        con.print(Panel(Text.assemble(
            ("screengraph-os ", "bold magenta"), (f"v{__version__}", "magenta"), (tr("demo.tagline"), "grey70"),
            (tr("demo.subtitle"), "grey50")),
            box=box.ROUNDED, border_style="magenta"))
        con.print(prompt(f"{flag}simulate"))
        con.print(Text.assemble(("  ", ""), (tr("demo.ticks", ticks=ticks), "white"), (" → ", "grey50"),
                                (tr("demo.dropped", dropped=ticks - stored), "green"), (" · ", "grey50"),
                                (tr("demo.stored", stored=stored), "white"), (" · ", "grey50"),
                                (f"{wall:.2f} s", "grey70")))
        con.print()

        con.print(prompt(f"{flag}active-workspace"))
        ws = sg.active_workspace()
        tbl = Table(box=box.SIMPLE_HEAD, header_style="bold grey70", pad_edge=False, expand=False)
        for col, wd in (("mon", 3), (tr("col.app"), 13), (tr("demo.col.window"), 30), (tr("demo.col.viewport"), 20),
                        (tr("demo.col.content"), 30)):
            tbl.add_column(col, no_wrap=True, overflow="ellipsis", width=wd)
        for w in sorted(ws["windows"], key=lambda w: (w["bbox"]["monitor_id"], w["bbox"]["x"])):
            b = w["bbox"]
            tbl.add_row(str(b["monitor_id"]), Text(w["app"], CAT_STYLE.get(w["app_category"], "white")),
                        escape(w["title"] or ""), f"{b['x']},{b['y']} {b['width']}×{b['height']}",
                        escape((w["summary"][-1] if w["summary"] else "").strip()))
        con.print(tbl)

        con.print(prompt(f'{flag}visual "{VISUAL_QUERY}"'))
        hit = sg.visual_search(VISUAL_QUERY, 1)[0]
        p = hit["patch"]
        sb = p["screen_bbox"]
        con.print(Text.assemble(("  #", "grey50"), (str(hit["event_id"]), "white"), ("  [Grafana] ", "magenta"),
                                (hit["window_title"], "white"), ("  MaxSim ", "grey50"), (f"{hit['maxsim']:.2f}", "white")))
        con.print(Text.assemble(("  patch ", "grey50"), (f"{tr('demo.patch_pos', row=p['row'], col=p['col'])} {p['quadrant']}", "bold cyan"),
                                (f"  {tr('demo.screen')} ", "grey50"),
                                (f"x={sb['x']} y={sb['y']} {sb['width']}×{sb['height']}", "white"),
                                (" " + tr("demo.monitor", n=hit["monitor_id"]), "grey50")))
        con.print(Text.assemble((f"  {tr('col.concepts')} ", "grey50"),
                                (" · ".join(k for k in p["visual_concepts"] if "_" not in k), "yellow"),
                                ("   " + tr("demo.pixel_only"), "grey50")))
        con.print()

        for target in ("k8s/payments-deployment.yaml", "db_pool.py"):
            con.print(prompt(f"{flag}why {target}"))
            con.print(why_tree(sg.why(target), target, lang))
            con.print()

        audit = sg.privacy_audit()
        g = report["gate_and_storage"]
        sw = report.get("robustness_seed_sweep", {})
        seeds = tr("demo.seeds", lo=sw["min"]["projected_30d_mb"], hi=sw["max"]["projected_30d_mb"]) if sw else ""
        foot = Text.assemble(
            (tr("demo.images", n=audit["disk"].get("image_magic_hits", 0)), "bold green"), ("  ·  ", "grey50"),
            (tr("demo.wipe_failures", n=audit["ram"]["verification_failures"]), "bold green"), ("  ·  ", "grey50"),
            (tr("demo.storage", mb=g["projected_30d_8h_mb"], seeds=seeds), "bold white"), (tr("demo.vs"), "grey50"),
            (f"{g['legacy_model_30d_8h_mb'] / 1000:.1f} GB", "red"), (tr("demo.video"), "grey50"))
        con.print(Panel(foot, title=tr("demo.footer_title"), title_align="left", box=box.ROUNDED,
                        border_style="grey50"))
        sg.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return save(con, out, "screengraph-os")


# ---------------------------------------------------------------------------------------------- CLI commands
def run_cli(home: Path, lang: str, args: list[str], con: Console) -> None:
    """Invoke the real Typer app in-process, printing through ``con`` instead of the terminal."""
    saved = cli.console
    cli.console = con
    try:
        cli.app(["--home", str(home), "--lang", lang, *args], standalone_mode=False)
    finally:
        cli.console = saved
        i18n.set_lang(None)


def render_cli(home: Path, lang: str, args: list[str], shown: str, out: Path, width: int,
               height: int | None = None) -> Path:
    con = recording_console(width, height)
    cmd = f"{lang_flag(lang)}{shown}"
    con.print(prompt(cmd))
    run_cli(home, lang, args, con)
    return save(con, out, f"screengraph {cmd}")


SHOTS: dict[str, Callable[[Path, str, Path], Path]] = {
    "why": lambda home, lang, out: render_cli(home, lang, ["why", "db_pool.py"], "why db_pool.py", out, 144),
    "search": lambda home, lang, out: render_cli(home, lang, ["search", "jwt expired", "-k", "4"],
                                                 'search "jwt expired" -k 4', out, 112),
    "status": lambda home, lang, out: render_cli(home, lang, ["status"], "status", out, 140, 40),
}


def main(argv: list[str] | None = None) -> list[Path]:
    ap = argparse.ArgumentParser(description="Render the README terminal screenshots from real engine/CLI output.")
    ap.add_argument("--lang", choices=i18n.SUPPORTED, action="append", help="Language(s) to render (default: all).")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    a = ap.parse_args(argv)
    langs = a.lang or list(i18n.SUPPORTED)
    written = [render_demo(lang, a.out_dir / lang / "demo.svg") for lang in langs]
    home = Path(tempfile.mkdtemp(prefix="screengraph-shots-"))
    try:  # one fixed-clock simulator session (`simulate --no-now`), queried in every language
        run_cli(home, i18n.DEFAULT_LANG, ["simulate", "--no-now"], recording_console(120))
        for lang in langs:
            for name, shot in SHOTS.items():
                written.append(shot(home, lang, a.out_dir / lang / f"{name}.svg"))
    finally:
        shutil.rmtree(home, ignore_errors=True)
    return written


if __name__ == "__main__":
    for p in main():
        print(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p)
