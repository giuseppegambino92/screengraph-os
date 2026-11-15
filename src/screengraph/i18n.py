"""Tiny message catalog for the human-facing CLI/TUI output (English + Italian).

Only text meant for people goes through :func:`t`: messages, table titles/headers, labels and help texts.
Machine-readable output (JSON, ``--json`` modes, MCP responses, relation names, evidence keys such as
``errors:``, captured window titles) is never translated.

Language resolution, highest priority first: :func:`set_lang` (the CLI's global ``--lang`` option), the
``SCREENGRAPH_LANG`` environment variable, then English. Unknown values in the environment fall back to English.
"""

from __future__ import annotations

import os
from string import Formatter

SUPPORTED: tuple[str, ...] = ("en", "it")
DEFAULT_LANG = "en"
ENV_VAR = "SCREENGRAPH_LANG"

_override: str | None = None

CATALOG: dict[str, dict[str, str]] = {
    "en": {
        # ---------------------------------------------------------------- CLI help (Typer)
        "help.app": "screengraph-os: zero-screenshot desktop memory + cross-app causal graph.",
        "help.app.home": "Data directory (default ~/.screengraph).",
        "help.app.lang": "Output language for human-readable text (env: SCREENGRAPH_LANG). JSON is never translated.",
        "help.version": "Print the version.",
        "help.simulate": "Replay the deterministic 60-minute multi-monitor developer session into the database.",
        "help.simulate.seed": "Simulator seed.",
        "help.simulate.ticks": "Window ticks to replay.",
        "help.simulate.now": "Anchor the session so it ends now.",
        "help.capture": "Capture real windows (best-effort native hooks) + clipboard + git working tree.",
        "help.capture.interval": "Seconds between polls.",
        "help.capture.duration": "Total seconds to run (0 = forever).",
        "help.capture.pixels": "Also grab window pixels (macOS + pyobjc; wiped after encoding).",
        "help.capture.git_repo": "Watch a git working tree for edits.",
        "help.capture.clipboard": "Watch the clipboard.",
        "help.ingest": "Ingest a hook event (shell command output, IDE save, note). Text is scrubbed before storage.",
        "help.ingest.app_name": "Application name.",
        "help.ingest.text": "Text to ingest (default: read stdin).",
        "help.ingest.clipboard": "Text copied from this window.",
        "help.search": "4-way hybrid search (BM25 + dense + visual Patch-MaxSim + causal graph, fused by RRF).",
        "help.why": "Explain why a file/symbol changed: trace the causal graph back to its root cause.",
        "help.visual": (
            "Visual search by colour / layout description, or by example (--like EVENT_ID), with patch localization."
        ),
        "help.visual.like": "Query by example: windows that look like event N.",
        "help.capture-live": (
            "Capture real macOS windows (ctypes CoreGraphics + AX) into a temporary DB and self-test the pipeline."
        ),
        "help.capture-live.ticks": "Desktop ticks to capture.",
        "help.capture-live.interval": "Seconds between ticks.",
        "help.capture-live.pixels": "Grab window pixels into RAM (wiped after encoding).",
        "help.capture-live.allow_clipboard": "Read the clipboard (pbpaste).",
        "help.capture-live.request_ax": "Show the macOS Accessibility prompt.",
        "help.capture-live.keep": "Keep the DB (in --home, or a temp dir that is printed).",
        "help.workspace": "Current semantic state of all windows (zero-screenshot context handoff).",
        "help.workspace.live": "Also enumerate native windows now.",
        "help.timeline": "Chronological narrative + Mermaid causal graph of the session.",
        "help.timeline.minutes": "Only the last N minutes.",
        "help.timeline.causal_only": "Only causally linked events.",
        "help.timeline.mermaid": "Print only the Mermaid graph.",
        "help.active-workspace": "Alias of `workspace`.",
        "help.active-workspace.live": "Also enumerate native windows now.",
        "help.privacy-audit": "Alias of `audit`.",
        "help.audit": "Privacy audit: zero on-disk pixels, RAM wipe ledger, redaction counters.",
        "help.status": "One-shot Rich dashboard.",
        "help.dash": "Live Rich TUI dashboard.",
        "help.dash.simulate": "Animate by replaying the simulator into the DB.",
        "help.dash.seconds": "Exit after N seconds.",
        "help.eval": "Run the deterministic benchmark suite.",
        "help.eval.scale": "Events for the at-scale latency run (0 = skip).",
        "help.eval.sweep": "Extra simulator seeds for the robustness sweep (0 = skip).",
        "help.eval.live": "Also capture N real desktop ticks (macOS).",
        "help.eval.out": "Report path (default evals/latest_report.json).",
        "help.mcp": "Serve the MCP tools over stdio (JSON-RPC 2.0).",
        # ---------------------------------------------------------------- CLI output
        "simulate.done": "replayed {total} ticks",
        "simulate.detail": (
            "in {seconds:.2f}s: {stored} stored, {dropped} dropped by the idle gate ({ratio:.1%}); "
            "db {mb:.2f} MB, {edges} causal edges"
        ),
        "capture.banner": "backend: [bold]{backend}[/]  pixels: {pixels}  git: {git}  (Ctrl-C to stop)",
        "capture.progress": "windows={windows} stored={stored} dropped={dropped}",
        "search.title": "{n} hits in {ms:.2f} ms",
        "col.app": "app",
        "col.title": "title",
        "col.channels": "channels",
        "col.snippet": "snippet",
        "col.patch": "patch",
        "col.quadrant": "quadrant",
        "col.concepts": "concepts",
        "why.header": "why {target}?",
        "why.root_cause": "root cause",
        "why.chain": "chain",
        "visual.title": "visual: {query}",
        "live.header": "live capture",
        "live.summary": (
            "{ticks} ticks · windows/tick {per_tick} · permissions: screen-recording={screen} accessibility={ax}"
        ),
        "live.frames": "frames {seen} (pixels {pixels}, AX text {ax}) -> stored {stored}, dropped by gate {dropped}",
        "live.latency": "OS capture/tick p50 {os_p50} ms · ingest p50/p95/max {p50}/{p95}/{max} ms (<10 ms: {under})",
        "live.audit": (
            "zero-pixel audit: image files on disk {images}, pixel blobs {blobs}, "
            "frames non-zero after ingest {nonzero} ->"
        ),
        "live.kept": "kept DB in {path}",
        "eval.missing": "evals/benchmark_suite.py not found; run from a source checkout",
        "error.bad_lang": "'{value}' is not a supported language (choose from: {choices}).",
        # ---------------------------------------------------------------- TUI
        "tui.tagline": "zero-screenshot spatio-temporal desktop memory",
        "tui.no_viewports": "no viewports yet",
        "tui.map_footer": "monitors: {monitors}  ·  viewports: {viewports}  ·  screenshots stored: 0",
        "tui.col.mon": "mon",
        "tui.col.app": "app",
        "tui.col.doc": "title / document",
        "tui.col.bbox": "bbox",
        "tui.col.changed": "changed",
        "tui.why": "why {doc}?",
        "tui.root_cause": "ROOT CAUSE",
        "tui.no_chains": "no causal chains yet - try `screengraph simulate`",
        "tui.events_edges": "events / causal edges",
        "tui.database": "database",
        "tui.footprint": "footprint",
        "tui.kb_hour": "{kb:.0f} KB/hour",
        "tui.projected": "projected 30d x 8h",
        "tui.video_model": "continuous video model",
        "tui.gb_hour": "{gb:.2f} GB/hour",
        "tui.reduction": "reduction",
        "tui.frames": "frames seen / dropped",
        "tui.buffers_zeroed": "RAM buffers zeroed",
        "tui.bytes_zeroed": "bytes zeroed",
        "tui.wipe_failures": "wipe verify failures",
        "tui.pixels_on_disk": "pixels on disk",
        "tui.zero_files": "0 files",
        "tui.redacted": "redacted {kind}",
        "tui.panel.viewports": "viewports",
        "tui.panel.storage": "storage",
        "tui.panel.chains": "causal chains (root cause -> edit)",
        "tui.panel.windows": "active windows",
        "tui.panel.privacy": "privacy",
        # ---------------------------------------------------------------- README screenshots (scripts/)
        "demo.tagline": "   zero-screenshot desktop memory · cross-app causal graph · MCP\n",
        "demo.subtitle": (
            "deterministic simulator, seed 7 — real engine output, regenerated by scripts/generate_demo_asset.py"
        ),
        "demo.ticks": "{ticks} window ticks",
        "demo.dropped": "{dropped} dropped by the idle gate",
        "demo.stored": "{stored} semantic events stored",
        "demo.col.window": "window",
        "demo.col.viewport": "viewport x,y w×h",
        "demo.col.content": "latest content",
        "demo.traced": "traced in {ms:.1f} ms",
        "demo.patch_pos": "(row {row}, col {col})",
        "demo.screen": "screen",
        "demo.monitor": "monitor {n}",
        "demo.pixel_only": "(pixel-only frame: no text changed)",
        "demo.images": "{n} image files on disk",
        "demo.wipe_failures": "{n} wipe failures",
        "demo.seeds": " (seeds {lo:.0f}–{hi:.0f})",
        "demo.storage": "{mb:.1f} MB/30 d int8{seeds}",
        "demo.vs": "  vs ",
        "demo.video": " modelled video",
        "demo.footer_title": "privacy & storage",
    },
    "it": {
        # ---------------------------------------------------------------- CLI help (Typer)
        "help.app": "screengraph-os: memoria del desktop senza screenshot + grafo causale tra app.",
        "help.app.home": "Directory dei dati (default ~/.screengraph).",
        "help.app.lang": "Lingua dei testi leggibili (env: SCREENGRAPH_LANG). Il JSON non viene mai tradotto.",
        "help.version": "Mostra la versione.",
        "help.simulate": "Riproduce nel database la sessione di sviluppo deterministica di 60 minuti su più monitor.",
        "help.simulate.seed": "Seed del simulatore.",
        "help.simulate.ticks": "Numero di tick delle finestre da riprodurre.",
        "help.simulate.now": "Ancora la sessione in modo che termini adesso.",
        "help.capture": "Cattura le finestre reali (hook nativi best-effort) + appunti + working tree git.",
        "help.capture.interval": "Secondi tra un polling e l'altro.",
        "help.capture.duration": "Durata totale in secondi (0 = senza limite).",
        "help.capture.pixels": "Cattura anche i pixel delle finestre (macOS + pyobjc; azzerati dopo l'encoding).",
        "help.capture.git_repo": "Osserva le modifiche in un working tree git.",
        "help.capture.clipboard": "Osserva gli appunti.",
        "help.ingest": (
            "Acquisisce un evento da hook (output di un comando shell, salvataggio nell'IDE, nota). "
            "Il testo viene ripulito prima della memorizzazione."
        ),
        "help.ingest.app_name": "Nome dell'applicazione.",
        "help.ingest.text": "Testo da acquisire (default: letto da stdin).",
        "help.ingest.clipboard": "Testo copiato da questa finestra.",
        "help.search": "Ricerca ibrida a 4 vie (BM25 + dense + Patch-MaxSim visivo + grafo causale, fuse con RRF).",
        "help.why": "Spiega perché un file/simbolo è cambiato: risale il grafo causale fino alla root cause.",
        "help.visual": (
            "Ricerca visiva per descrizione di colori / layout, o per esempio (--like EVENT_ID), "
            "con localizzazione della patch."
        ),
        "help.visual.like": "Query by example: finestre simili all'evento N.",
        "help.capture-live": (
            "Cattura le finestre reali di macOS (CoreGraphics via ctypes + AX) in un DB temporaneo "
            "ed esegue un self-test della pipeline."
        ),
        "help.capture-live.ticks": "Numero di tick del desktop da catturare.",
        "help.capture-live.interval": "Secondi tra un tick e l'altro.",
        "help.capture-live.pixels": "Cattura i pixel delle finestre in RAM (azzerati dopo l'encoding).",
        "help.capture-live.allow_clipboard": "Legge gli appunti (pbpaste).",
        "help.capture-live.request_ax": "Mostra la richiesta di autorizzazione Accessibilità di macOS.",
        "help.capture-live.keep": "Conserva il DB (in --home, oppure in una directory temporanea che viene stampata).",
        "help.workspace": "Stato semantico attuale di tutte le finestre (passaggio di contesto senza screenshot).",
        "help.workspace.live": "Elenca anche ora le finestre native.",
        "help.timeline": "Narrazione cronologica + grafo causale Mermaid della sessione.",
        "help.timeline.minutes": "Solo gli ultimi N minuti.",
        "help.timeline.causal_only": "Solo gli eventi collegati causalmente.",
        "help.timeline.mermaid": "Stampa solo il grafo Mermaid.",
        "help.active-workspace": "Alias di `workspace`.",
        "help.active-workspace.live": "Elenca anche ora le finestre native.",
        "help.privacy-audit": "Alias di `audit`.",
        "help.audit": (
            "Audit della privacy: zero pixel su disco, registro degli azzeramenti in RAM, contatori degli oscuramenti."
        ),
        "help.status": "Dashboard Rich istantanea.",
        "help.dash": "Dashboard TUI Rich in tempo reale.",
        "help.dash.simulate": "Anima la dashboard riproducendo il simulatore nel DB.",
        "help.dash.seconds": "Esce dopo N secondi.",
        "help.eval": "Esegue la suite di benchmark deterministica.",
        "help.eval.scale": "Eventi per il test di latenza su larga scala (0 = salta).",
        "help.eval.sweep": "Seed aggiuntivi del simulatore per il test di robustezza (0 = salta).",
        "help.eval.live": "Cattura anche N tick reali del desktop (macOS).",
        "help.eval.out": "Percorso del report (default evals/latest_report.json).",
        "help.mcp": "Espone i tool MCP su stdio (JSON-RPC 2.0).",
        # ---------------------------------------------------------------- CLI output
        "simulate.done": "riprodotti {total} tick",
        "simulate.detail": (
            "in {seconds:.2f}s: {stored} memorizzati, {dropped} scartati dal gate di inattività "
            "({ratio:.1%}); db {mb:.2f} MB, {edges} archi causali"
        ),
        "capture.banner": "backend: [bold]{backend}[/]  pixel: {pixels}  git: {git}  (Ctrl-C per interrompere)",
        "capture.progress": "finestre={windows} memorizzati={stored} scartati={dropped}",
        "search.title": "{n} risultati in {ms:.2f} ms",
        "col.app": "app",
        "col.title": "titolo",
        "col.channels": "canali",
        "col.snippet": "snippet",
        "col.patch": "patch",
        "col.quadrant": "quadrante",
        "col.concepts": "concetti",
        "why.header": "perché {target}?",
        "why.root_cause": "root cause",
        "why.chain": "catena",
        "visual.title": "ricerca visiva: {query}",
        "live.header": "cattura live",
        "live.summary": (
            "{ticks} tick · finestre/tick {per_tick} · permessi: registrazione-schermo={screen} accessibilità={ax}"
        ),
        "live.frames": (
            "frame {seen} (con pixel {pixels}, con testo AX {ax}) -> memorizzati {stored}, scartati dal gate {dropped}"
        ),
        "live.latency": "cattura OS/tick p50 {os_p50} ms · ingest p50/p95/max {p50}/{p95}/{max} ms (<10 ms: {under})",
        "live.audit": (
            "audit zero-pixel: file immagine su disco {images}, blob di pixel {blobs}, "
            "frame non azzerati dopo l'ingest {nonzero} ->"
        ),
        "live.kept": "DB conservato in {path}",
        "eval.missing": "evals/benchmark_suite.py non trovato; eseguire da un checkout dei sorgenti",
        "error.bad_lang": "'{value}' non è una lingua supportata (scegli tra: {choices}).",
        # ---------------------------------------------------------------- TUI
        "tui.tagline": "memoria spazio-temporale del desktop senza screenshot",
        "tui.no_viewports": "ancora nessun viewport",
        "tui.map_footer": "monitor: {monitors}  ·  viewport: {viewports}  ·  screenshot salvati: 0",
        "tui.col.mon": "mon",
        "tui.col.app": "app",
        "tui.col.doc": "titolo / documento",
        "tui.col.bbox": "bbox",
        "tui.col.changed": "modificato",
        "tui.why": "perché {doc}?",
        "tui.root_cause": "ROOT CAUSE",
        "tui.no_chains": "ancora nessuna catena causale - prova `screengraph simulate`",
        "tui.events_edges": "eventi / archi causali",
        "tui.database": "database",
        "tui.footprint": "occupazione",
        "tui.kb_hour": "{kb:.0f} KB/ora",
        "tui.projected": "proiezione 30 g x 8 h",
        "tui.video_model": "modello video continuo",
        "tui.gb_hour": "{gb:.2f} GB/ora",
        "tui.reduction": "riduzione",
        "tui.frames": "frame visti / scartati",
        "tui.buffers_zeroed": "buffer RAM azzerati",
        "tui.bytes_zeroed": "byte azzerati",
        "tui.wipe_failures": "verifiche di azzeramento fallite",
        "tui.pixels_on_disk": "pixel su disco",
        "tui.zero_files": "0 file",
        "tui.redacted": "oscurati {kind}",
        "tui.panel.viewports": "viewport",
        "tui.panel.storage": "storage",
        "tui.panel.chains": "catene causali (root cause -> modifica)",
        "tui.panel.windows": "finestre attive",
        "tui.panel.privacy": "privacy",
        # ---------------------------------------------------------------- README screenshots (scripts/)
        "demo.tagline": "   memoria del desktop senza screenshot · grafo causale tra app · MCP\n",
        "demo.subtitle": (
            "simulatore deterministico, seed 7 — output reale, rigenerato da scripts/generate_demo_asset.py"
        ),
        "demo.ticks": "{ticks} tick di finestra",
        "demo.dropped": "{dropped} scartati dal gate di inattività",
        "demo.stored": "{stored} eventi semantici memorizzati",
        "demo.col.window": "finestra",
        "demo.col.viewport": "viewport x,y l×a",
        "demo.col.content": "ultimo contenuto",
        "demo.traced": "tracciato in {ms:.1f} ms",
        "demo.patch_pos": "(riga {row}, col {col})",
        "demo.screen": "schermo",
        "demo.monitor": "monitor {n}",
        "demo.pixel_only": "(frame solo pixel: nessun testo cambiato)",
        "demo.images": "{n} immagini su disco",
        "demo.wipe_failures": "{n} wipe falliti",
        "demo.seeds": " (seed {lo:.0f}–{hi:.0f})",
        "demo.storage": "{mb:.1f} MB/30 g int8{seeds}",
        "demo.vs": "  vs ",
        "demo.video": " di video stimato",
        "demo.footer_title": "privacy e storage",
    },
}


def normalize(value: str | None) -> str | None:
    """Map ``"it"``, ``"IT"``, ``"it_IT.UTF-8"``, ``"en-US"`` ... to a supported code, else ``None``."""
    if not value:
        return None
    code = value.strip().lower().replace("-", "_").split(".")[0].split("_")[0]
    return code if code in SUPPORTED else None


def set_lang(lang: str | None) -> str:
    """Force the output language (``None`` = back to ``$SCREENGRAPH_LANG`` / English). Returns the active one."""
    global _override
    if lang is not None and normalize(lang) is None:
        raise ValueError(f"unsupported language {lang!r}; choose one of {', '.join(SUPPORTED)}")
    _override = normalize(lang)
    return get_lang()


def get_lang() -> str:
    return _override or normalize(os.environ.get(ENV_VAR)) or DEFAULT_LANG


def has(key: str) -> bool:
    return key in CATALOG[DEFAULT_LANG]


def t(key: str, /, *, lang: str | None = None, **kwargs: object) -> str:
    """Translate ``key`` into the active (or given) language and fill ``{placeholders}`` from ``kwargs``.

    Missing keys in a non-default language fall back to English; an unknown key raises ``KeyError``.
    """
    code = normalize(lang) or get_lang()
    msg = CATALOG[code].get(key)
    if msg is None:
        msg = CATALOG[DEFAULT_LANG][key]
    return msg.format(**kwargs) if kwargs else msg


def placeholders(msg: str) -> set[str]:
    """Names of the ``{fields}`` used by a catalog message (format specs ignored)."""
    return {name for _, name, _, _ in Formatter().parse(msg) if name}
