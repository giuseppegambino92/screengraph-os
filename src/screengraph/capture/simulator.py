"""Deterministic multi-window, multi-monitor workspace simulator.

Renders realistic *structural* pixel grids for five app archetypes at 1/8 of
logical resolution (one grid pixel ~ one 8 px character cell) together with
their accessibility trees:

* Terminal  (iTerm2)    dark, ANSI-coloured log lines, stack traces, selections
* Browser   (Chrome)    search results, docs, CI badges, tree diagrams, modals
* IDE       (VS Code)   syntax-coloured code, explorer sidebar, blue status bar
* Dashboard (Grafana)   2x2 panels: line charts (with optional spike), bar chart
* Chat      (Slack)     aubergine sidebar, avatars, message threads

Layout: monitor 0 is a 5120x1440 ultra-wide, monitor 1 a 2560x1440.

``WorkspaceSimulator.session()`` replays a scripted 60-minute developer session
(450 window ticks, ~85% idle/cursor-only ticks, 4 multi-hop debugging chains,
6 visual artifacts, 15 injected secrets) and yields frames plus ground truth.
Everything is seeded: same seed -> byte-identical frames.
"""

from __future__ import annotations

import base64
import json
import string
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional

import numpy as np

from screengraph.capture.models import AXNode, BoundingBox, ViewportFrame

SCALE = 8
LINE_H = 5

# ---------------------------------------------------------------------------
# Glyph model: each character is a 1-px column whose height depends on its shape
# ---------------------------------------------------------------------------
_HLUT = np.zeros(256, np.uint8)
for _c in range(256):
    _ch = chr(_c)
    if _ch.isspace():
        _HLUT[_c] = 0
    elif _ch.isupper() or _ch.isdigit() or _ch in "bdfhklt!?/|\\()[]{}#$%&@":
        _HLUT[_c] = 3
    elif _ch in ".,:;-_'\"`~^*=+<>":
        _HLUT[_c] = 1
    else:
        _HLUT[_c] = 2

Color = tuple[int, int, int]

# palette
TERM_BG, TERM_FG, TERM_RED, TERM_GREEN, TERM_YELLOW, TERM_CYAN, TERM_DIM = (
    (28, 28, 32), (200, 200, 200), (230, 80, 80), (80, 200, 120), (230, 200, 80), (100, 180, 230), (120, 120, 125))
SEL_BG = (78, 78, 88)
WHITE, PAGE_FG, LINK, MUTED = (250, 250, 250), (40, 40, 40), (26, 13, 171), (110, 110, 110)
CHROME_BAR, CHROME_TAB = (222, 225, 230), (240, 240, 242)
IDE_BG, IDE_SIDE, IDE_FG, IDE_KW, IDE_STR, IDE_TYPE, IDE_COMMENT, IDE_NUM, IDE_STATUS = (
    (30, 30, 30), (37, 37, 38), (212, 212, 212), (197, 134, 192), (206, 145, 120), (78, 201, 176),
    (106, 153, 85), (181, 206, 168), (0, 122, 204))
DASH_BG, PANEL_BG, AXIS = (17, 18, 23), (24, 27, 31), (60, 60, 64)
G_BLUE, G_RED, G_GREEN, G_CYAN = (87, 148, 242), (242, 73, 92), (115, 191, 105), (80, 200, 220)
SLACK_SIDE, SLACK_FG = (74, 21, 75), (29, 28, 29)
TREE_BLUE, CONNECTOR = (40, 110, 220), (150, 150, 155)
BADGE_RED, BADGE_GREEN, MODAL_ORANGE = (215, 58, 73), (40, 167, 69), (245, 140, 30)
AVATARS = [(224, 30, 90), (46, 182, 125), (236, 178, 46), (54, 197, 240), (120, 80, 200)]


class Canvas:
    def __init__(self, h: int, w: int, bg: Color):
        self.h, self.w = h, w
        self.px = np.empty((h, w, 3), np.uint8)
        self.px[:] = bg

    def rect(self, x: int, y: int, w: int, h: int, color: Color) -> None:
        x0, y0, x1, y1 = max(0, x), max(0, y), min(self.w, x + w), min(self.h, y + h)
        if x1 > x0 and y1 > y0:
            self.px[y0:y1, x0:x1] = color

    def text(self, x: int, y: int, s: str, color: Color, max_w: Optional[int] = None) -> tuple[int, int, int, int]:
        if y < 0 or y + 3 > self.h or x >= self.w:
            return (x, y, 0, 3)
        n = min(len(s), (max_w if max_w is not None else self.w - x), self.w - x)
        if n <= 0:
            return (x, y, 0, 3)
        hts = _HLUT[np.frombuffer(s[:n].encode("ascii", "replace"), np.uint8)]
        row = self.px[y:y + 3, x:x + n]
        for r in range(3):
            row[r][hts >= 3 - r] = color
        return (x, y, n, 3)

    def vline(self, x: int, y0: int, y1: int, color: Color) -> None:
        if 0 <= x < self.w:
            a, b = sorted((y0, y1))
            self.px[max(0, a):min(self.h, b + 1), x] = color

    def hline(self, y: int, x0: int, x1: int, color: Color) -> None:
        if 0 <= y < self.h:
            a, b = sorted((x0, x1))
            self.px[y, max(0, a):min(self.w, b + 1)] = color

    def polyline(self, xs: np.ndarray, ys: np.ndarray, color: Color) -> None:
        prev = int(ys[0])
        for x, y in zip(xs, ys):
            y = int(y)
            self.vline(int(x), prev, y, color)
            prev = y


@dataclass
class Artifact:
    name: str
    kind: str
    rects: list[tuple[int, int, int, int]] = field(default_factory=list)  # (x, y, w, h) grid px
    points: list[tuple[int, int]] = field(default_factory=list)  # (x, y) grid px

    def gt_patches(self, h: int, w: int, rows: int = 4, cols: int = 4, min_cover: float = 0.3) -> set[tuple[int, int]]:
        out = set()
        for (x, y) in self.points:
            out.add((min(int(y * rows / h), rows - 1), min(int(x * cols / w), cols - 1)))
        ph, pw = h / rows, w / cols
        for (x, y, rw, rh) in self.rects:
            for r in range(rows):
                for c in range(cols):
                    oy = max(0.0, min(y + rh, (r + 1) * ph) - max(y, r * ph))
                    ox = max(0.0, min(x + rw, (c + 1) * pw) - max(x, c * pw))
                    if oy * ox >= min_cover * ph * pw:
                        out.add((r, c))
        return out


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------
class SimWindow:
    category = "other"
    app = "App"
    bundle_id = ""

    def __init__(self, window_id: str, bbox: BoundingBox, pid: int, z: int = 0):
        self.window_id, self.bbox, self.pid, self.z = window_id, bbox, pid, z
        self.gh, self.gw = bbox.height // SCALE, bbox.width // SCALE

    @property
    def title(self) -> str:
        return self.app

    url = ""
    doc_path = ""

    def render(self) -> tuple[np.ndarray, list[AXNode], list[Artifact]]:
        raise NotImplementedError

    def titlebar(self, cv: Canvas, nodes: list[AXNode]) -> None:
        cv.rect(0, 0, cv.w, 3, (58, 58, 60))
        cv.text(max(2, cv.w // 2 - len(self.title) // 2), 0, self.title, (190, 190, 190))


class TerminalWindow(SimWindow):
    category, app, bundle_id = "terminal", "iTerm2", "com.googlecode.iterm2"

    def __init__(self, *a, cwd: str = "~/src/acme-api", **k):
        super().__init__(*a, **k)
        self.cwd = cwd
        self.lines: list[tuple[str, Color]] = []
        self.selection: Optional[str] = None
        self.running = "zsh"

    @property
    def title(self) -> str:
        return f"{self.running} — {self.cwd}"

    def run(self, cmd: str, output: list, running: Optional[str] = None) -> None:
        self.selection = None
        self.lines.append((f"dev@mbp {self.cwd} % {cmd}", TERM_GREEN))
        for o in output:
            self.lines.append(o if isinstance(o, tuple) else (o, TERM_FG))
        self.running = running or "zsh"

    def clear(self) -> None:
        self.lines, self.selection = [], None

    def select(self, text: str) -> None:
        self.selection = text

    def render(self):
        cv = Canvas(self.gh, self.gw, TERM_BG)
        nodes: list[AXNode] = []
        self.titlebar(cv, nodes)
        top = 5
        max_lines = (self.gh - top - 2) // LINE_H
        visible = self.lines[-max_lines:]
        for i, (txt, col) in enumerate(visible):
            y = top + i * LINE_H
            if self.selection and self.selection in txt:
                sx = 1 + txt.index(self.selection)
                cv.rect(sx, y - 1, min(len(self.selection), self.gw - sx), 5, SEL_BG)
            bb = cv.text(1, y, txt, col)
            nodes.append(AXNode("AXStaticText", txt[: self.gw - 1], bb))
        cv.text(1, top + len(visible) * LINE_H, "dev@mbp %", TERM_GREEN)
        if self.selection:
            nodes.append(AXNode("AXSelectedText", self.selection, (1, top, len(self.selection), 3), focused=True))
        return cv.px, nodes, []


@dataclass
class Page:
    url: str
    title: str
    blocks: list[tuple] = field(default_factory=list)


class BrowserWindow(SimWindow):
    category, app, bundle_id = "browser", "Google Chrome", "com.google.Chrome"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.page = Page("about:blank", "New Tab")

    @property
    def title(self) -> str:
        return f"{self.page.title} - Google Chrome"

    @property
    def url(self) -> str:  # type: ignore[override]
        return self.page.url

    def navigate(self, page: Page) -> None:
        self.page = page

    def render(self):
        cv = Canvas(self.gh, self.gw, WHITE)
        nodes: list[AXNode] = []
        arts: list[Artifact] = []
        self.titlebar(cv, nodes)
        cv.rect(0, 3, cv.w, 4, CHROME_BAR)
        cv.rect(2, 3, 40, 4, CHROME_TAB)
        cv.text(4, 4, self.page.title[:36], PAGE_FG)
        cv.rect(0, 7, cv.w, 6, CHROME_TAB)
        cv.rect(8, 8, cv.w - 16, 4, (255, 255, 255))
        bb = cv.text(10, 9, self.page.url, MUTED, max_w=cv.w - 20)
        nodes.append(AXNode("AXTextField", self.page.url, bb))
        y, x0, maxw = 16, 6, cv.w - 12
        for blk in self.page.blocks:
            kind = blk[0]
            if y >= cv.h - 4:
                break
            if kind in ("h1", "p", "code", "link", "muted"):
                txt = blk[1]
                col = {"h1": PAGE_FG, "p": PAGE_FG, "code": (60, 60, 60), "link": LINK, "muted": MUTED}[kind]
                if kind == "code":
                    cv.rect(x0 - 1, y - 1, maxw + 2, LINE_H * (1 + len(txt) // maxw), (238, 238, 240))
                chunks = [txt[i:i + maxw] for i in range(0, max(len(txt), 1), maxw)]
                for ch in chunks:
                    bb = cv.text(x0, y, ch, col)
                    if kind == "h1":
                        cv.text(x0, y + 1, ch, col)  # bold: double-struck
                    nodes.append(AXNode("AXHeading" if kind == "h1" else "AXLink" if kind == "link" else "AXStaticText",
                                        ch + (f" {blk[2]}" if kind == "link" and len(blk) > 2 else ""), bb))
                    y += LINE_H + (2 if kind == "h1" else 0)
            elif kind == "badge_row":
                _, label, status = blk
                bb = cv.text(x0, y, label, PAGE_FG)
                nodes.append(AXNode("AXStaticText", label, bb))
                bx = x0 + len(label) + 4
                col = BADGE_RED if status == "failing" else BADGE_GREEN
                cv.rect(bx, y - 1, 16, 5, col)
                cv.text(bx + 2, y, status, WHITE, max_w=12)
                nodes.append(AXNode("AXStaticText", status, (bx, y - 1, 16, 5)))
                if status == "failing":
                    arts.append(Artifact("red_badge", "badge", [], [(bx + 8, y + 1)]))
                y += LINE_H + 3
            elif kind == "tree":
                _, label = blk
                y = self._tree(cv, x0, y, maxw, arts, nodes, label)
            elif kind == "modal":
                _, mtitle, mbody = blk
                mw, mh = 100, 40
                mx, my = (cv.w - mw) // 2, (cv.h - mh) // 2
                cv.rect(mx, my, mw, mh, MODAL_ORANGE)
                bb = cv.text(mx + 4, my + 4, mtitle, WHITE, max_w=mw - 8)
                nodes.append(AXNode("AXDialog", mtitle, bb, focused=True))
                for i, ln in enumerate(mbody):
                    bb = cv.text(mx + 4, my + 12 + i * LINE_H, ln, WHITE, max_w=mw - 8)
                    nodes.append(AXNode("AXStaticText", ln, bb))
                cv.rect(mx + mw - 26, my + mh - 9, 20, 6, (255, 236, 210))
                nodes.append(AXNode("AXButton", "Dismiss", (mx + mw - 26, my + mh - 9, 20, 6)))
                arts.append(Artifact("orange_modal", "modal", [(mx, my, mw, mh)], [(mx + mw // 2, my + mh // 2)]))
            y += 1
        return cv.px, nodes, arts

    def _tree(self, cv, x0, y, maxw, arts, nodes, label) -> int:
        bw, bh = 18, 7
        levels = [1, 2, 4]
        centers: list[list[tuple[int, int]]] = []
        boxes = []
        span = min(maxw, 150)
        left = x0 + (maxw - span) // 2
        for li, n in enumerate(levels):
            ly = y + li * (bh + 9)
            row = []
            for k in range(n):
                cx = left + int((k + 0.5) * span / n)
                row.append((cx, ly))
            centers.append(row)
        for li in range(1, len(levels)):
            for k, (cx, ly) in enumerate(centers[li]):
                px_, py_ = centers[li - 1][k * len(centers[li - 1]) // len(centers[li])]
                mid = py_ + bh + 4
                cv.vline(px_, py_ + bh, mid, CONNECTOR)
                cv.hline(mid, px_, cx, CONNECTOR)
                cv.vline(cx, mid, ly - 1, CONNECTOR)
        names = ["gateway", "payments-api", "ledger-svc", "postgres", "kafka", "fraud-ml", "redis"]
        ni = 0
        for row in centers:
            for (cx, ly) in row:
                cv.rect(cx - bw // 2, ly, bw, bh, TREE_BLUE)
                cv.text(cx - bw // 2 + 2, ly + 2, names[ni % len(names)][:14], WHITE)
                nodes.append(AXNode("AXImage", names[ni % len(names)], (cx - bw // 2, ly, bw, bh)))
                boxes.append((cx - bw // 2, ly, bw, bh))
                ni += 1
        total_h = len(levels) * (bh + 9)
        xs = [b[0] for b in boxes]
        arts.append(Artifact("blue_tree", "tree", [(min(xs), y, max(xs) + bw - min(xs), total_h)],
                             [(b[0] + bw // 2, b[1] + bh // 2) for b in boxes]))
        nodes.append(AXNode("AXImage", label, (left, y, span, total_h)))
        return y + total_h + 2


def _syntax(line: str) -> list[tuple[str, Color]]:
    """Very small tokenizer for code colouring."""
    kws = {"def", "return", "import", "from", "class", "if", "else", "let", "const", "fn", "pub", "async",
           "await", "with", "as", "try", "except", "raise", "for", "in", "impl", "use", "mut", "self", "export"}
    out: list[tuple[str, Color]] = []
    stripped = line.lstrip()
    if stripped.startswith(("#", "//")):
        return [(line, IDE_COMMENT)]
    i = 0
    while i < len(line):
        ch = line[i]
        if ch in "\"'":
            j = line.find(ch, i + 1)
            j = len(line) - 1 if j < 0 else j
            out.append((line[i:j + 1], IDE_STR))
            i = j + 1
            continue
        if ch.isalpha() or ch == "_":
            j = i
            while j < len(line) and (line[j].isalnum() or line[j] == "_"):
                j += 1
            word = line[i:j]
            col = IDE_KW if word in kws else IDE_TYPE if word[:1].isupper() else IDE_FG
            out.append((word, col))
            i = j
            continue
        if ch.isdigit():
            j = i
            while j < len(line) and (line[j].isalnum() or line[j] == "."):
                j += 1
            out.append((line[i:j], IDE_NUM))
            i = j
            continue
        out.append((ch, IDE_FG))
        i += 1
    return out


class IDEWindow(SimWindow):
    category, app, bundle_id = "ide", "Code", "com.microsoft.VSCode"

    def __init__(self, *a, project: str = "acme-api", **k):
        super().__init__(*a, **k)
        self.project = project
        self.files = ["src/app/db_pool.py", "src/auth/middleware.ts", "src/scheduler.rs",
                      "k8s/payments-deployment.yaml", "config/payments.py", ".env.local", "README.md", "CHANGELOG.md"]
        self.path = "README.md"
        self.code: list[str] = []
        self.dirty = False

    @property
    def title(self) -> str:
        base = self.path.rsplit("/", 1)[-1]
        return f"{'● ' if self.dirty else ''}{base} — {self.project} — Visual Studio Code"

    @property
    def doc_path(self) -> str:  # type: ignore[override]
        return self.path

    def open(self, path: str, code: list[str], dirty: bool = False) -> None:
        self.path, self.code, self.dirty = path, code, dirty

    def render(self):
        cv = Canvas(self.gh, self.gw, IDE_BG)
        nodes: list[AXNode] = []
        self.titlebar(cv, nodes)
        side_w = 30
        cv.rect(0, 3, side_w, cv.h - 6, IDE_SIDE)
        cv.text(2, 5, "EXPLORER", (150, 150, 150))
        for i, f in enumerate(self.files):
            col = (230, 230, 230) if f == self.path else (160, 160, 160)
            bb = cv.text(3, 11 + i * LINE_H, f.rsplit("/", 1)[-1], col, max_w=side_w - 4)
            nodes.append(AXNode("AXRow", f, bb))
        cv.rect(side_w, 3, cv.w - side_w, 4, (45, 45, 46))
        cv.text(side_w + 2, 4, self.path.rsplit("/", 1)[-1], (220, 220, 220))
        x0 = side_w + 8
        max_lines = (cv.h - 10 - 4) // LINE_H
        for i, ln in enumerate(self.code[:max_lines]):
            y = 9 + i * LINE_H
            cv.text(side_w + 1, y, f"{i + 1:>3}", (90, 90, 90))
            x = x0
            for tok, col in _syntax(ln):
                cv.text(x, y, tok, col)
                x += len(tok)
            nodes.append(AXNode("AXStaticText", ln, (x0, y, len(ln), 3)))
        cv.rect(0, cv.h - 3, cv.w, 3, IDE_STATUS)
        cv.text(2, cv.h - 3, f"main  Ln 1, Col 1  {self.path}", WHITE)
        return cv.px, nodes, []


class DashboardWindow(SimWindow):
    category, app, bundle_id = "dashboard", "Grafana", "com.grafana.desktop"

    PANELS = [("requests/s · gateway", G_BLUE, "line"), ("payments-api memory (GiB)", G_RED, "line"),
              ("CI build duration (min)", G_GREEN, "bar"), ("p95 latency (ms) · checkout", G_CYAN, "line")]

    def __init__(self, *a, seed: int = 0, **k):
        super().__init__(*a, **k)
        self.rng = np.random.default_rng(seed)
        self.version = 0
        self.spike_at: Optional[float] = None  # relative x position of memory spike, None = no spike
        self._data = self._gen()

    @property
    def title(self) -> str:
        return "Production Overview - Grafana"

    @property
    def url(self) -> str:  # type: ignore[override]
        return "https://grafana.acme.dev/d/prod-overview"

    def _gen(self) -> list[np.ndarray]:
        out = []
        for _, _, kind in self.PANELS:
            if kind == "bar":
                out.append(self.rng.uniform(0.25, 0.9, 8))
            else:
                walk = np.cumsum(self.rng.normal(0, 0.012, 200))
                out.append(np.clip(0.4 + walk - walk.mean(), 0.2, 0.6))
        return out

    def refresh(self) -> None:
        self.version += 1
        self._data = self._gen()
        if self.spike_at is not None:
            self.spike_at = max(0.15, self.spike_at - 0.25)  # spike scrolls left as time passes

    def render(self):
        cv = Canvas(self.gh, self.gw, DASH_BG)
        nodes: list[AXNode] = []
        arts: list[Artifact] = []
        self.titlebar(cv, nodes)
        cv.text(2, 5, "Production Overview  ·  Last 1 hour", (200, 200, 200))
        nodes.append(AXNode("AXHeading", "Production Overview · Last 1 hour", (2, 5, 40, 3)))
        top, gap = 10, 1
        pw, ph = (cv.w - 3 * gap) // 2, (cv.h - top - 3 * gap) // 2
        for i, (name, col, kind) in enumerate(self.PANELS):
            r, c = divmod(i, 2)
            x, y = gap + c * (pw + gap), top + gap + r * (ph + gap)
            cv.rect(x, y, pw, ph, PANEL_BG)
            bb = cv.text(x + 2, y + 2, name, (200, 200, 200), max_w=pw - 4)
            nodes.append(AXNode("AXHeading", name, bb))
            ax0, ay0, aw, ah = x + 4, y + 9, pw - 7, ph - 13
            cv.vline(ax0 - 1, ay0, ay0 + ah, AXIS)
            cv.hline(ay0 + ah, ax0 - 1, ax0 + aw, AXIS)
            data = self._data[i]
            if kind == "bar":
                n = len(data)
                bw_ = max(2, aw // (n * 2))
                for k, v in enumerate(data):
                    bh_ = max(3, int(v * (ah - 2)))
                    cv.rect(ax0 + 2 + k * (2 * bw_), ay0 + ah - bh_, bw_, bh_, col)
                arts.append(Artifact("green_bars", "bar_chart", [(ax0, ay0, aw, ah)], []))
            else:
                xs = np.arange(ax0, ax0 + aw)
                vals = np.interp(np.linspace(0, len(data) - 1, aw), np.arange(len(data)), data)
                spike_pt = None
                if i == 1 and self.spike_at is not None:
                    sx = int(self.spike_at * aw)
                    vals = vals.copy()
                    vals[sx - 1:sx + 2] = [0.75, 0.97, 0.8]
                    spike_pt = sx
                ys = ay0 + ah - 1 - (vals * (ah - 2)).astype(int)
                cv.polyline(xs, ys, col)
                if i == 1:
                    if spike_pt is not None:
                        px_ = ax0 + spike_pt
                        arts.append(Artifact("red_spike", "spike", [],
                                             [(px_, int(ys[spike_pt])), (px_, int(np.median(ys)))]))
                    else:
                        arts.append(Artifact("red_line", "line_chart", [(ax0, ay0, aw, ah)], []))
                if i == 3:
                    arts.append(Artifact("cyan_line", "line_chart", [(ax0, ay0, aw, ah)], []))
        return cv.px, nodes, arts


class ChatWindow(SimWindow):
    category, app, bundle_id = "chat", "Slack", "com.tinyspeck.slackmacgap"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.channel = "#eng-backend"
        self.messages: list[tuple[str, str]] = []
        self.draft = ""

    @property
    def title(self) -> str:
        return f"{self.channel} - Acme - Slack"

    def post(self, author: str, text: str, channel: Optional[str] = None) -> None:
        if channel:
            self.channel = channel
        self.messages.append((author, text))

    def render(self):
        cv = Canvas(self.gh, self.gw, WHITE)
        nodes: list[AXNode] = []
        self.titlebar(cv, nodes)
        side = 28
        cv.rect(0, 3, side, cv.h - 3, SLACK_SIDE)
        for i, ch in enumerate(["#eng-backend", "#rust-help", "#incidents", "#random"]):
            cv.text(2, 7 + i * LINE_H, ch, (220, 200, 220) if ch == self.channel else (170, 140, 170), max_w=side - 3)
        cv.text(side + 3, 5, self.channel, SLACK_FG)
        nodes.append(AXNode("AXHeading", self.channel, (side + 3, 5, len(self.channel), 3)))
        x0, maxw = side + 13, cv.w - side - 16
        rendered: list[tuple[str, str, list[str]]] = []
        for author, text in self.messages:
            chunks = [text[i:i + maxw] for i in range(0, max(1, len(text)), maxw)]
            rendered.append((author, text, chunks))
        budget = (cv.h - 12 - 8) // LINE_H
        shown, used = [], 0
        for item in reversed(rendered):
            need = 1 + len(item[2]) + 1
            if used + need > budget:
                break
            shown.append(item)
            used += need
        y = 12
        for author, text, chunks in reversed(shown):
            av = AVATARS[sum(map(ord, author)) % len(AVATARS)]
            cv.rect(side + 3, y, 8, 8, av)
            bb = cv.text(x0, y, author, SLACK_FG)
            cv.text(x0, y + 1, author, SLACK_FG)
            y += LINE_H
            for ch in chunks:
                col = LINK if ch.startswith("http") else SLACK_FG
                bb = cv.text(x0, y, ch, col)
                y += LINE_H
            nodes.append(AXNode("AXStaticText", f"{author}: {text}", (x0, y - LINE_H * len(chunks), maxw, LINE_H * len(chunks))))
            y += 2
        cv.rect(side + 3, cv.h - 8, cv.w - side - 6, 6, (245, 245, 245))
        if self.draft:
            cv.text(side + 5, cv.h - 6, self.draft, SLACK_FG)
        return cv.px, nodes, []


# ---------------------------------------------------------------------------
# Fake secrets (generated at runtime: no credential-shaped literals in source)
# ---------------------------------------------------------------------------
def _rand(rng: np.random.Generator, alphabet: str, n: int) -> str:
    return "".join(alphabet[i] for i in rng.integers(0, len(alphabet), n))


def _luhn_complete(prefix: str, rng: np.random.Generator, length: int = 16) -> str:
    body = prefix + "".join(str(d) for d in rng.integers(0, 10, length - len(prefix) - 1))
    total = 0
    for i, ch in enumerate(reversed(body)):
        d = int(ch)
        if i % 2 == 0:
            d *= 2
            d -= 9 if d > 9 else 0
        total += d
    return body + str((10 - total % 10) % 10)


def fake_secrets(seed: int = 7) -> dict[str, str]:
    rng = np.random.default_rng(seed + 1000)
    an = string.ascii_letters + string.digits
    up = string.ascii_uppercase + string.digits
    b64 = an + "+/"

    def b64url(obj) -> str:
        return base64.urlsafe_b64encode(json.dumps(obj, separators=(",", ":")).encode()).decode().rstrip("=")

    jwt = ".".join([b64url({"alg": "HS256", "typ": "JWT"}), b64url({"sub": "u_8812", "exp": 1731499000}),
                    _rand(rng, an + "-_", 43)])
    pem_body = [_rand(rng, b64, 64) for _ in range(3)]
    sym = "!@#$%^&*"
    pw_chars = list(rng.permutation(list("ZqxTLpWmRvKj") + list("84297613") + list(sym))[:20])
    return {
        "openai_key": "sk-" + "proj-" + _rand(rng, an, 48),
        "aws_access_key": "AK" + "IA" + _rand(rng, up, 16),
        "aws_secret_key": _rand(rng, b64, 40),
        "jwt": jwt,
        "github_token": "gh" + "p_" + _rand(rng, an, 36),
        "stripe_key": "sk" + "_live_" + _rand(rng, an, 24),
        "slack_token": "xo" + "xb-" + "-".join(_rand(rng, string.digits, n) for n in (12, 13)) + "-" + _rand(rng, an, 24),
        "google_api_key": "AI" + "za" + _rand(rng, an + "-_", 35),
        "ssh_private_key": "\n".join(["-----BEGIN OPENSSH " + "PRIVATE KEY-----", *pem_body, "-----END OPENSSH " + "PRIVATE KEY-----"]),
        "chat_password": "".join(pw_chars),
        "db_password": "hunter2-" + _rand(rng, an, 10),
        "credit_card": _luhn_complete("4539", rng),
        "cli_password": "Tr0ub4dor-" + _rand(rng, an, 8),
        "anthropic_key": "sk-" + "ant-api03-" + _rand(rng, an + "-_", 60),
        "api_header_key": _rand(rng, b64, 40),
    }


# ---------------------------------------------------------------------------
# Session script
# ---------------------------------------------------------------------------
@dataclass
class Action:
    t: float
    window: str
    fn: Callable[[], None]
    copy: Optional[str] = None
    chain: Optional[str] = None
    role: str = ""
    secrets: tuple[str, ...] = ()
    label: str = ""


@dataclass
class SimTick:
    index: int
    ts: float
    kind: str  # "change" | "idle"
    frame: ViewportFrame
    action: Optional[Action]
    artifacts: list[dict]


class WorkspaceSimulator:
    """Deterministic multi-monitor workspace with a scripted developer session."""

    def __init__(self, seed: int = 7, n_ticks: int = 450, duration_s: float = 3600.0, t0: float = 1_731_500_000.0):
        self.seed, self.n_ticks, self.duration, self.t0 = seed, n_ticks, duration_s, t0
        self.rng = np.random.default_rng(seed)
        self.secrets = fake_secrets(seed)
        m0w, m1x = 5120, 5120
        self.windows: dict[str, SimWindow] = {
            "term": TerminalWindow("term", BoundingBox(0, 0, 1280, 1440, 0), pid=4101, z=1),
            "browser": BrowserWindow("browser", BoundingBox(1280, 0, 1920, 1440, 0), pid=4102, z=2),
            "ide": IDEWindow("ide", BoundingBox(3200, 0, m0w - 3200, 1440, 0), pid=4103, z=3),
            "dash": DashboardWindow("dash", BoundingBox(m1x, 0, 1280, 1440, 1), pid=4104, z=4, seed=seed),
            "chat": ChatWindow("chat", BoundingBox(m1x + 1280, 0, 1280, 1440, 1), pid=4105, z=5),
        }
        self.actions = self._script()
        self.focus = "term"

    # -- frames ------------------------------------------------------------
    def frame(self, wid: str, ts: float, copy: Optional[str] = None, cursor: Optional[tuple[int, int]] = None,
              caret: bool = False) -> tuple[ViewportFrame, list[dict]]:
        w = self.windows[wid]
        px, nodes, arts = w.render()
        if caret:
            px[-6:-3, 12:13] = (230, 230, 230)
        if cursor is not None:
            cy, cx = cursor
            px[cy:cy + 3, cx:cx + 2] = (255, 255, 255)
            px[cy + 3:cy + 4, cx:cx + 1] = (0, 0, 0)
        gh, gw = px.shape[:2]
        art_info = [{"name": a.name, "kind": a.kind, "patches": sorted(a.gt_patches(gh, gw))} for a in arts]
        f = ViewportFrame(
            window_id=w.window_id, app=w.app, title=w.title, bbox=w.bbox, ts=ts, app_category=w.category,
            bundle_id=w.bundle_id, pid=w.pid, z_index=w.z, url=w.url, doc_path=w.doc_path, ax_nodes=nodes,
            clipboard_copied=copy, pixels=px, source="sim",
        )
        return f, art_info

    def snapshot(self, ts: Optional[float] = None) -> list[ViewportFrame]:
        """Current frame of every window (multi-monitor viewport decomposition)."""
        ts = self.t0 if ts is None else ts
        return [self.frame(wid, ts)[0] for wid in self.windows]

    def session(self, changes_only: bool = False) -> Iterator[SimTick]:
        dt = self.duration / self.n_ticks
        slots: dict[int, Action] = {}
        for a in sorted(self.actions, key=lambda a: a.t):
            k = int(round(a.t / dt))
            while k in slots:
                k += 1
            slots[k] = a
        weights = {"term": 0.3, "browser": 0.25, "ide": 0.25, "dash": 0.1, "chat": 0.1}
        wids = list(weights)
        probs = np.array([weights[w] for w in wids])
        for i in range(self.n_ticks):
            ts = self.t0 + i * dt + float(self.rng.uniform(0, 0.5))
            act = slots.get(i)
            if act is not None:
                act.fn()
                self.focus = act.window
                frame, arts = self.frame(act.window, ts, copy=act.copy)
                yield SimTick(i, ts, "change", frame, act, arts)
                continue
            if changes_only:
                continue
            wid = self.focus if self.rng.random() < 0.6 else wids[int(self.rng.choice(len(wids), p=probs))]
            w = self.windows[wid]
            r = self.rng.random()
            cursor = (int(self.rng.integers(10, w.gh - 10)), int(self.rng.integers(10, w.gw - 10))) if r < 0.7 else None
            caret = wid in ("term", "ide") and self.rng.random() < 0.5
            frame, arts = self.frame(wid, ts, cursor=cursor, caret=caret)
            yield SimTick(i, ts, "idle", frame, None, arts)

    # -- ground truth helpers ---------------------------------------------------
    CHAIN_TARGETS = {"C1": "src/app/db_pool.py", "C2": "src/auth/middleware.ts", "C3": "src/scheduler.rs",
                     "C4": "k8s/payments-deployment.yaml"}
    VISUAL_QUERIES = {
        "red_spike": "red memory spike chart in top-right quadrant",
        "blue_tree": "blue architecture tree diagram",
        "orange_modal": "orange warning modal",
        "green_bars": "green bar chart of build durations",
        "red_badge": "red failing badge",
        "cyan_line": "cyan latency line chart",
    }

    # -- the script --------------------------------------------------------------
    def _script(self) -> list[Action]:
        T, B, I, D, C = (self.windows[k] for k in ("term", "browser", "ide", "dash", "chat"))
        S = self.secrets
        A: list[Action] = []

        def add(t, win, fn, **kw):
            A.append(Action(t, win, fn, **kw))

        R, Y, D_, CY = TERM_RED, TERM_YELLOW, TERM_DIM, TERM_CYAN
        # ---- boot: every window's first frame ------------------------------------------
        add(0, "term", lambda: T.run("git status", ["On branch main", "Your branch is up to date with 'origin/main'.",
                                                     "nothing to commit, working tree clean"]), label="boot")
        add(8, "browser", lambda: B.navigate(Page("https://news.ycombinator.com/", "Hacker News", [
            ("h1", "Hacker News"), ("link", "1. Show HN: A tiny SQL engine in 500 lines", "https://github.com/x/tiny-sql"),
            ("link", "2. The unreasonable effectiveness of boring technology", "https://boringtech.dev/post"),
            ("link", "3. Rust 1.83 released", "https://blog.rust-lang.org/2024/11/28/Rust-1.83.0.html"),
            ("muted", "312 points by pg 4 hours ago | 120 comments")])), label="boot")
        add(16, "ide", lambda: I.open("README.md", ["# acme-api", "", "Payments and auth backend for Acme.",
                                                     "", "## Running locally", "", "    make dev", "    make test"]),
            label="boot")
        add(24, "dash", lambda: None, label="boot")
        add(32, "chat", lambda: C.post("priya", "morning! standup at 10:15 today"), label="boot")

        # ---- secrets + distractors ------------------------------------------------------
        add(60, "term", lambda: T.run(f"export OPENAI_API_KEY={S['openai_key']}", []) or
            T.run(f"export SLACK_BOT_TOKEN={S['slack_token']}", []), secrets=("openai_key", "slack_token"))
        add(100, "chat", lambda: C.post("priya", "reminder: payments retro moved to thursday"))
        add(140, "ide", lambda: I.open(".env.local", [
            "# local only - never commit", f"AWS_ACCESS_KEY_ID={S['aws_access_key']}",
            f"AWS_SECRET_ACCESS_KEY={S['aws_secret_key']}", f"DB_PASSWORD={S['db_password']}",
            f"ANTHROPIC_API_KEY={S['anthropic_key']}", "LOG_LEVEL=debug"]),
            secrets=("aws_access_key", "aws_secret_key", "db_password", "anthropic_key"))

        # ---- chain C1: SQLite busy -> search -> docs -> db_pool.py --------------------------
        add(180, "term", lambda: T.run("pytest tests/test_pool.py -x", [
            "============ test session starts ============", "collected 6 items", "",
            ("tests/test_pool.py ..F", TERM_FG), ("______ test_concurrent_writes ______", R),
            "Traceback (most recent call last):",
            '  File "src/app/db_pool.py", line 88, in acquire',
            "    conn.execute(\"BEGIN IMMEDIATE\")",
            ("app.errors.SQLiteBusyException: database is locked", R),
            ("FAILED tests/test_pool.py::test_concurrent_writes - SQLiteBusyException", R),
            ("======= 1 failed, 2 passed in 1.92s =======", R)], running="pytest"),
            chain="C1", role="root")
        add(212, "dash", lambda: D.refresh())
        add(236, "term", lambda: T.select("SQLiteBusyException: database is locked"),
            copy="SQLiteBusyException: database is locked", chain="C1", role="copy")
        add(268, "chat", lambda: C.post("marco", "anyone up for lunch at 12:30?"))
        add(300, "browser", lambda: B.navigate(Page(
            "https://www.google.com/search?q=SQLiteBusyException%3A+database+is+locked",
            "SQLiteBusyException: database is locked - Google Search", [
                ("muted", "SQLiteBusyException: database is locked"),
                ("link", "PRAGMA busy_timeout - SQLite", "https://www.sqlite.org/pragma.html#pragma_busy_timeout"),
                ("p", "busy_timeout sets a busy handler that sleeps when a table is locked ..."),
                ("link", "Write-Ahead Logging - SQLite", "https://www.sqlite.org/wal.html"),
                ("link", "database is locked with concurrent writers - Stack Overflow",
                 "https://stackoverflow.com/questions/2740806")])), chain="C1", role="search")
        add(356, "browser", lambda: B.navigate(Page(
            "https://www.sqlite.org/pragma.html#pragma_busy_timeout", "Pragma statements supported by SQLite", [
                ("h1", "PRAGMA busy_timeout"),
                ("code", "PRAGMA busy_timeout = milliseconds;"),
                ("p", "Query or change the setting of the busy timeout. Sets a busy handler that sleeps for a "
                      "specified amount of time when a table is locked (SQLITE_BUSY)."),
                ("p", "Combine with journal_mode=WAL so readers do not block writers."),
                ("code", "PRAGMA journal_mode = WAL;")])), chain="C1", role="docs")
        add(392, "chat", lambda: C.post("priya", "heads up: CI queue is slow this morning"))
        add(430, "ide", lambda: I.open("src/app/db_pool.py", [
            "import sqlite3", "", "class Pool:", "    def acquire(self):",
            "        conn = sqlite3.connect(self.path, timeout=5.0)",
            "        conn.execute(\"PRAGMA journal_mode=WAL\")",
            "        conn.execute(\"PRAGMA busy_timeout = 5000\")",
            "        conn.execute(\"BEGIN IMMEDIATE\")", "        return conn"], dirty=True), chain="C1", role="fix")
        add(470, "term", lambda: T.run("pytest tests/test_pool.py -x", [
            ("tests/test_pool.py ......", TERM_GREEN), ("======= 6 passed in 0.41s =======", TERM_GREEN)]),
            label="C1 effect")
        add(520, "chat", lambda: C.post("priya", "nice, CI is green again"))
        add(560, "browser", lambda: B.navigate(Page("https://news.ycombinator.com/item?id=4242", "Boring technology | HN", [
            ("h1", "The unreasonable effectiveness of boring technology"),
            ("p", "Choose boring technology: postgres, cron and a queue get you surprisingly far.")])))
        add(600, "dash", lambda: D.refresh())
        add(640, "term", lambda: T.run("cat ~/.ssh/id_ed25519", S["ssh_private_key"].split("\n")),
            secrets=("ssh_private_key",))
        add(680, "term", lambda: T.clear() or T.run(
            f"git remote set-url origin https://{S['github_token']}@github.com/acme/api.git", []),
            secrets=("github_token",))
        add(704, "term", lambda: T.run(f"echo $OPENAI_API_KEY", [S["openai_key"]]) or T.select(S["openai_key"]),
            copy=S["openai_key"], label="secret copy")
        add(728, "chat", lambda: C.post("me", f"key for the eval box: {S['openai_key']}", channel="#eng-backend"),
            label="secret paste")
        add(760, "browser", lambda: B.navigate(Page("http://localhost:3000/checkout", "Checkout · Acme", [
            ("h1", "Checkout"), ("p", "Email: jane.doe@example.com"),
            ("p", f"Card number: {S['credit_card'][:4]} {S['credit_card'][4:8]} {S['credit_card'][8:12]} {S['credit_card'][12:]}"),
            ("p", "Expiry 12/28  CVC ***")])), secrets=("credit_card",))
        add(800, "ide", lambda: I.open("config/payments.py", [
            "import os", "", f"STRIPE_SECRET_KEY = \"{S['stripe_key']}\"", "CURRENCY = \"EUR\"",
            "RETRY_LIMIT = 3"]), secrets=("stripe_key",))
        add(840, "dash", lambda: D.refresh())

        # ---- chain C2: JWT expired -> search -> README -> middleware.ts ---------------------
        add(900, "term", lambda: T.clear() or T.run("npm run dev", [
            ("ready - started server on 0.0.0.0:3000, url: http://localhost:3000", CY),
            "GET /api/v1/session 401 12ms",
            ("TokenExpiredError: jwt expired", R),
            "    at verify (node_modules/jsonwebtoken/verify.js:152:21)",
            "    at verifyToken (src/auth/middleware.ts:41:11)"], running="node"), chain="C2", role="root")
        add(950, "term", lambda: T.run(
            f"curl -s -H \"Authorization: Bearer {S['jwt']}\" localhost:3000/api/v1/session",
            ['{"error":"TokenExpiredError: jwt expired"}']), chain="C2", role="root", secrets=("jwt",))
        add(990, "term", lambda: T.select("TokenExpiredError: jwt expired"), copy="TokenExpiredError: jwt expired",
            chain="C2", role="copy")
        add(1020, "chat", lambda: C.post("marco", "fyi I'll touch middleware.ts later today for the request logging"),
            label="hard negative")
        add(1060, "browser", lambda: B.navigate(Page(
            "https://www.google.com/search?q=TokenExpiredError%3A+jwt+expired",
            "TokenExpiredError: jwt expired - Google Search", [
                ("muted", "TokenExpiredError: jwt expired"),
                ("link", "auth0/node-jsonwebtoken: JsonWebToken implementation for node.js",
                 "https://github.com/auth0/node-jsonwebtoken#readme"),
                ("p", "clockTolerance: number of seconds to tolerate when checking the nbf and exp claims"),
                ("link", "jwt expired error even though token is fresh - Stack Overflow",
                 "https://stackoverflow.com/questions/52184472")])), chain="C2", role="search")
        add(1110, "browser", lambda: B.navigate(Page(
            "https://github.com/auth0/node-jsonwebtoken#readme", "auth0/node-jsonwebtoken - GitHub", [
                ("h1", "jsonwebtoken"), ("h1", "jwt.verify(token, secretOrPublicKey, [options, callback])"),
                ("p", "clockTolerance: number of seconds to tolerate when checking the nbf and exp claims, "
                      "to deal with small amounts of clock skew."),
                ("p", "Error TokenExpiredError: thrown if the token is expired."),
                ("code", "jwt.verify(token, secret, { clockTolerance: 30 })")])), chain="C2", role="docs")
        add(1140, "dash", lambda: D.refresh())
        add(1180, "ide", lambda: I.open("src/auth/middleware.ts", [
            "import jwt from 'jsonwebtoken'", "", "export function verifyToken(token: string) {",
            "  // tolerate small clock skew between auth and api pods",
            "  return jwt.verify(token, process.env.JWT_SECRET, { clockTolerance: 30 })", "}"], dirty=True),
            chain="C2", role="fix")
        add(1230, "term", lambda: T.run("npm run dev", ["GET /api/v1/session 200 9ms"], running="node"),
            label="C2 effect")
        add(1270, "chat", lambda: C.post("priya", "design review for ledger v2 at 3pm"))
        add(1320, "browser", lambda: B.navigate(Page("http://localhost:3000/__devtools/network", "DevTools - Network", [
            ("h1", "Request Headers"), ("p", "accept: application/json"),
            ("p", f"x-api-key: {S['api_header_key']}"), ("p", "user-agent: Mozilla/5.0")])),
            secrets=("api_header_key",))
        add(1380, "browser", lambda: B.navigate(Page(
            f"https://maps.googleapis.com/maps/api/staticmap?center=Berlin&key={S['google_api_key']}",
            "staticmap (640x640)", [("p", "Map preview")])), secrets=("google_api_key",))
        add(1440, "ide", lambda: I.open("README.md", ["# acme-api", "", "## Architecture",
                                                     "See the wiki for the payments service tree."]))
        add(1500, "browser", lambda: B.navigate(Page("https://grafana.com/orgs/acme/billing", "Billing · Grafana Cloud", [
            ("h1", "Billing"), ("p", "Usage this month: 41% of quota"),
            ("modal", "Warning: trial ending", ["Your Grafana Cloud trial", "ends in 3 days.", "Add a payment method."])])),
            label="visual orange_modal")
        add(1540, "browser", lambda: B.navigate(Page("https://wiki.acme.dev/payments/architecture",
                                                     "Payments architecture - Acme Wiki", [
            ("h1", "Payments service architecture"),
            ("p", "Request flow from the edge gateway down to storage and streaming."),
            ("tree", "payments service tree diagram"),
            ("p", "Owners: payments team. Last reviewed 2024-10-02.")])), label="visual blue_tree")
        add(1600, "dash", lambda: D.refresh())

        # ---- chain C3: rustc E0502 -> chat -> docs link -> scheduler.rs ---------------------
        e0502 = "error[E0502]: cannot borrow `self.queue` as mutable because it is also borrowed as immutable"
        add(1700, "term", lambda: T.clear() or T.run("cargo build", [
            ("   Compiling scheduler v0.4.0 (/Users/dev/src/scheduler)", TERM_GREEN), (e0502, R),
            ("   --> src/scheduler.rs:142:9", CY), "    |",
            "141 |         let head = self.queue.peek();", "142 |         self.queue.push(job);",
            ("For more information about this error, try `rustc --explain E0502`.", Y),
            ("error: could not compile `scheduler` due to previous error", R)]), chain="C3", role="root")
        add(1745, "term", lambda: T.select(e0502), copy=e0502, chain="C3", role="copy")
        add(1790, "chat", lambda: C.post("me", f"anyone seen this? {e0502}", channel="#rust-help"),
            chain="C3", role="ask")
        add(1860, "chat", lambda: C.post("marco", "classic one - see https://doc.rust-lang.org/error_codes/E0502.html "
                                                  "and clone the handle before you push"), chain="C3", role="answer")
        add(1900, "browser", lambda: B.navigate(Page("https://doc.rust-lang.org/error_codes/E0502.html",
                                                     "E0502 - Error codes index", [
            ("h1", "Error code E0502"),
            ("p", "A variable already borrowed as immutable was borrowed as mutable."),
            ("code", "let y = &a; bar(&mut a); // error: cannot borrow `a` as mutable"),
            ("p", "To fix this error, ensure that you don't have any other references to the variable before "
                  "trying to access it mutably.")])), chain="C3", role="docs")
        add(1950, "dash", lambda: D.refresh())
        add(2010, "ide", lambda: I.open("src/scheduler.rs", [
            "impl Scheduler {", "    pub fn enqueue(&mut self, job: Job) {",
            "        // fix E0502: end the immutable borrow before mutating",
            "        let head = self.queue.peek().cloned();", "        self.queue.push(job);",
            "        self.notify(head);", "    }", "}"], dirty=True), chain="C3", role="fix")
        add(2060, "term", lambda: T.run("cargo build", [
            ("    Finished dev [unoptimized + debuginfo] target(s) in 4.21s", TERM_GREEN)]), label="C3 effect")
        add(2100, "chat", lambda: C.post("me", "thanks marco, that was it"))
        add(2200, "browser", lambda: B.navigate(Page("https://news.ycombinator.com/", "Hacker News", [
            ("h1", "Hacker News"), ("link", "1. Postgres as a queue, revisited", "https://example.org/pg-queue"),
            ("link", "2. Why we left Kubernetes", "https://example.org/left-k8s")])))
        add(2260, "ide", lambda: I.open("CHANGELOG.md", ["## 0.9.3", "- pool: enable WAL and busy timeout",
                                                         "- auth: tolerate 30s clock skew"]))
        add(2320, "term", lambda: T.clear() or T.run(f"mysql -h db.internal -u app --password={S['cli_password']}",
                                                      ["ERROR 2003 (HY000): Can't connect to MySQL server"]),
            secrets=("cli_password",))
        add(2380, "chat", lambda: C.post("priya", f"staging db creds rotated, new one: {S['chat_password']}",
                                         channel="#eng-backend"), secrets=("chat_password",))
        add(2440, "browser", lambda: B.navigate(Page("https://github.com/acme/api/actions", "Actions · acme/api", [
            ("h1", "All workflows"), ("badge_row", "unit-tests          main  #1842", "passing"),
            ("badge_row", "integration-tests   main  #1842", "failing"),
            ("badge_row", "lint                main  #1842", "passing"),
            ("p", "integration-tests failed after 6m 12s")])), label="visual red_badge")

        # ---- chain C4: memory spike -> kubectl OOMKilled -> search -> docs -> deployment.yaml ----
        def spike():
            D.refresh()
            D.spike_at = 0.72
        add(2600, "dash", spike, chain="C4", role="root")
        add(2660, "term", lambda: T.clear() or T.run("kubectl get pods -n payments", [
            "NAME                          READY   STATUS      RESTARTS      AGE",
            ("payments-api-7d9f8b-x2k      0/1     OOMKilled   3 (12s ago)   41m", R),
            "ledger-svc-5c4b7-q8m          1/1     Running     0             3d"]), chain="C4", role="symptom")
        add(2700, "term", lambda: T.run("kubectl describe pod payments-api-7d9f8b-x2k -n payments", [
            "    Last State:     Terminated", ("      Reason:       OOMKilled", R), "      Exit Code:    137",
            "    Limits:", "      memory:  512Mi",
            ("  Warning  OOMKilling  kernel-monitor  Memory cgroup out of memory: Killed process 4312 (payments-api)", Y)]),
            chain="C4", role="symptom")
        add(2740, "term", lambda: T.select("Memory cgroup out of memory: Killed process"),
            copy="Memory cgroup out of memory: Killed process", chain="C4", role="copy")
        add(2790, "browser", lambda: B.navigate(Page(
            "https://www.google.com/search?q=Memory+cgroup+out+of+memory%3A+Killed+process+kubernetes",
            "Memory cgroup out of memory: Killed process kubernetes - Google Search", [
                ("muted", "Memory cgroup out of memory: Killed process kubernetes"),
                ("link", "Resource Management for Pods and Containers | Kubernetes",
                 "https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/"),
                ("p", "If a container exceeds its memory limit it is terminated (OOMKilled) ..."),
                ("link", "Exit code 137 OOMKilled - Stack Overflow", "https://stackoverflow.com/questions/59729917")])),
            chain="C4", role="search")
        add(2850, "browser", lambda: B.navigate(Page(
            "https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/",
            "Resource Management for Pods and Containers | Kubernetes", [
                ("h1", "Resource Management for Pods and Containers"),
                ("p", "If a container exceeds its memory limit, the kernel kills it and the status is OOMKilled."),
                ("code", "resources:  requests: { memory: 512Mi }  limits: { memory: 1Gi }"),
                ("p", "Set requests to typical usage and limits above the observed peak.")])), chain="C4", role="docs")
        add(2900, "chat", lambda: C.post("priya", "is payments flapping for anyone else?", channel="#incidents"),
            label="hard negative")
        add(2950, "ide", lambda: I.open("k8s/payments-deployment.yaml", [
            "apiVersion: apps/v1", "kind: Deployment", "metadata:", "  name: payments-api", "spec:",
            "  template:", "    spec:", "      containers:", "        - name: payments-api",
            "          # raise limit: pods were OOMKilled at 512Mi under peak load",
            "          resources:", "            requests: { memory: 768Mi }", "            limits: { memory: 1Gi }"],
            dirty=True), chain="C4", role="fix")
        add(3000, "term", lambda: T.run("kubectl apply -f k8s/payments-deployment.yaml",
                                        ["deployment.apps/payments-api configured"]), label="C4 effect")
        add(3060, "dash", lambda: D.refresh())
        add(3120, "chat", lambda: C.post("me", "payments memory looks healthy after the limit bump", channel="#incidents"))
        add(3200, "browser", lambda: B.navigate(Page("https://grafana.com/docs/grafana/latest/alerting/",
                                                     "Alerting | Grafana documentation", [
            ("h1", "Grafana Alerting"), ("p", "Alert rules evaluate queries and fire notifications.")])))
        add(3300, "ide", lambda: I.open("CHANGELOG.md", ["## 0.9.4", "- k8s: payments memory limit 1Gi",
                                                         "- scheduler: fix borrow in enqueue"]))
        add(3400, "term", lambda: T.clear() or T.run("git commit -am \"fix: pool, auth skew, scheduler, k8s limits\"",
                                                      ["[main 9f2c1ab] fix: pool, auth skew, scheduler, k8s limits",
                                                       " 5 files changed, 18 insertions(+), 6 deletions(-)"]))
        add(3480, "chat", lambda: C.post("marco", "shipping the logging change tomorrow"))
        add(3560, "dash", lambda: D.refresh())
        return A


def secret_inventory(sim: WorkspaceSimulator) -> dict[str, str]:
    return dict(sim.secrets)
