"""Cross-platform window / accessibility / clipboard / git introspection with graceful fallbacks.

Native capture is best-effort and permission-gated by the OS:

* macOS  - Quartz ``CGWindowListCopyWindowInfo`` for viewports (needs ``pyobjc``;
           window titles need Screen Recording permission), ``AXUIElement`` text
           (needs Accessibility permission), optional ``CGWindowListCreateImage``
           pixels. Without pyobjc: ``lsappinfo`` front-app metadata only.
* Linux  - ``wmctrl -lGp`` / ``xdotool`` (X11); Wayland compositors generally do
           not expose other clients' windows, so only the clipboard works there.
* Windows- ``user32`` via ctypes: EnumWindows / GetWindowText / GetWindowRect.
* All    - clipboard (pbpaste / wl-paste / xclip / Win32), git working tree.

Pixels obtained natively are copied into a writable numpy buffer at 1/8 scale
and handed to the EphemeralFrameBuffer like simulator frames. The OS-owned
source bitmap (e.g. an immutable CFData) cannot be wiped by us; it is released
immediately. The deterministic simulator lives in ``simulator.py`` and is
re-exported here.
"""

from __future__ import annotations

import hashlib
import os
import platform
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from screengraph.capture.models import AXNode, BoundingBox, ViewportFrame
from screengraph.capture.simulator import WorkspaceSimulator, fake_secrets  # noqa: F401  (re-export)

_CATEGORY_HINTS = [
    ("terminal", ("terminal", "iterm", "kitty", "alacritty", "wezterm", "konsole", "ghostty", "warp", "tilix", "xterm")),
    ("browser", ("chrome", "safari", "firefox", "arc", "edge", "brave", "vivaldi", "chromium", "opera")),
    ("ide", ("code", "cursor", "pycharm", "intellij", "xcode", "vim", "zed", "sublime", "windsurf", "goland",
             "webstorm", "rustrover", "emacs", "git")),
    ("chat", ("slack", "discord", "teams", "messages", "telegram", "signal", "whatsapp", "zulip", "mattermost")),
    ("dashboard", ("grafana", "datadog", "activity monitor", "kibana", "new relic", "htop", "btop")),
]


def categorize(app: str, title: str = "") -> str:
    s = f"{app} {title}".lower()
    for cat, keys in _CATEGORY_HINTS:
        if any(k in s for k in keys):
            return cat
    return "other"


def _run(cmd: list[str], timeout: float = 2.0) -> Optional[str]:
    if not shutil.which(cmd[0]):
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return out.stdout if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


@dataclass
class WindowInfo:
    window_id: str
    app: str
    title: str
    bbox: BoundingBox
    pid: int = 0
    z_index: int = 0
    bundle_id: str = ""


# ---------------------------------------------------------------------------
# Window enumeration
# ---------------------------------------------------------------------------
class NativeCapture:
    def __init__(self, with_pixels: bool = False, with_ax: bool = True, scale: int = 8):
        self.system = platform.system()
        self.with_pixels, self.with_ax, self.scale = with_pixels, with_ax, scale
        self._quartz = None
        if self.system == "Darwin":
            try:
                import Quartz  # type: ignore

                self._quartz = Quartz
            except Exception:
                self._quartz = None

    @property
    def backend(self) -> str:
        if self.system == "Darwin":
            return "quartz" if self._quartz else "lsappinfo"
        if self.system == "Linux":
            return "wmctrl" if shutil.which("wmctrl") else ("xdotool" if shutil.which("xdotool") else "none")
        if self.system == "Windows":
            return "win32"
        return "none"

    def list_windows(self) -> list[WindowInfo]:
        try:
            if self.system == "Darwin":
                return self._mac_windows()
            if self.system == "Linux":
                return self._linux_windows()
            if self.system == "Windows":
                return self._win_windows()
        except Exception:
            return []
        return []

    # -- macOS
    def _mac_windows(self) -> list[WindowInfo]:
        Q = self._quartz
        if Q is None:
            front = _run(["lsappinfo", "front"])
            if not front:
                return []
            info = _run(["lsappinfo", "info", "-only", "name", front.strip()]) or ""
            m = re.search(r'^\s*"([^"]+)"', info)  # first line: "AppName" ASN:0x0-0x...: (in front)
            if not m:
                m = re.search(r'"(?:LSDisplayName|name)"\s*=\s*"([^"]+)"', info)
            if not m:
                return []
            name = m.group(1)
            return [WindowInfo(f"mac:{name}", name, name, BoundingBox(0, 0, 0, 0))]
        opts = Q.kCGWindowListOptionOnScreenOnly | Q.kCGWindowListExcludeDesktopElements
        out = []
        for z, w in enumerate(Q.CGWindowListCopyWindowInfo(opts, Q.kCGNullWindowID) or []):
            if int(w.get("kCGWindowLayer", 0)) != 0:
                continue
            b = w.get("kCGWindowBounds", {})
            bb = BoundingBox(int(b.get("X", 0)), int(b.get("Y", 0)), int(b.get("Width", 0)), int(b.get("Height", 0)))
            if bb.width < 100 or bb.height < 100:
                continue
            out.append(WindowInfo(f"mac:{w.get('kCGWindowNumber')}", str(w.get("kCGWindowOwnerName", "")),
                                  str(w.get("kCGWindowName", "") or ""), bb, int(w.get("kCGWindowOwnerPID", 0)), z))
        return out

    def _mac_pixels(self, info: WindowInfo) -> Optional[np.ndarray]:
        Q = self._quartz
        if Q is None:
            return None
        wid = int(info.window_id.split(":", 1)[1])
        img = Q.CGWindowListCreateImage(Q.CGRectNull, Q.kCGWindowListOptionIncludingWindow, wid,
                                        Q.kCGWindowImageBoundsIgnoreFraming | Q.kCGWindowImageNominalResolution)
        if img is None:
            return None
        w, h, bpr = Q.CGImageGetWidth(img), Q.CGImageGetHeight(img), Q.CGImageGetBytesPerRow(img)
        data = Q.CGDataProviderCopyData(Q.CGImageGetDataProvider(img))
        raw = np.frombuffer(data, np.uint8).reshape(h, bpr // 4, 4)[:, :w]
        s = self.scale
        small = np.ascontiguousarray(raw[::s, ::s, 2::-1])  # BGRA -> RGB, 1/scale, writable copy
        del raw, data, img
        return small

    def _mac_ax(self, info: WindowInfo, max_nodes: int = 400) -> list[AXNode]:
        try:
            import ApplicationServices as AS  # type: ignore
        except Exception:
            return []
        app = AS.AXUIElementCreateApplication(info.pid)
        err, win = AS.AXUIElementCopyAttributeValue(app, "AXFocusedWindow", None)
        if err or win is None:
            return []
        nodes: list[AXNode] = []
        stack = [(win, 0)]
        while stack and len(nodes) < max_nodes:
            el, depth = stack.pop()
            for attr in ("AXValue", "AXTitle", "AXDescription"):
                e2, v = AS.AXUIElementCopyAttributeValue(el, attr, None)
                if not e2 and isinstance(v, str) and v.strip():
                    _, role = AS.AXUIElementCopyAttributeValue(el, "AXRole", None)
                    nodes.append(AXNode(str(role or "AX"), v.strip()[:2000]))
                    break
            if depth < 12:
                e3, kids = AS.AXUIElementCopyAttributeValue(el, "AXChildren", None)
                if not e3 and kids:
                    stack.extend((k, depth + 1) for k in kids)
        return nodes

    # -- Linux
    def _linux_windows(self) -> list[WindowInfo]:
        out = _run(["wmctrl", "-lGp"])
        res = []
        if out:
            for z, line in enumerate(out.splitlines()):
                parts = line.split(None, 8)
                if len(parts) < 9:
                    continue
                wid, _desk, pid, x, y, w, h, _host, title = parts
                app = title.rsplit(" - ", 1)[-1] if " - " in title else title.split()[0] if title else "unknown"
                res.append(WindowInfo(f"x11:{wid}", app, title, BoundingBox(int(x), int(y), int(w), int(h)), int(pid), z))
            return res
        name = _run(["xdotool", "getactivewindow", "getwindowname"])
        if name:
            return [WindowInfo("x11:active", name.strip().rsplit(" - ", 1)[-1], name.strip(), BoundingBox(0, 0, 0, 0))]
        return []

    # -- Windows
    def _win_windows(self) -> list[WindowInfo]:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        res: list[WindowInfo] = []

        @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        def cb(hwnd, _):
            if not user32.IsWindowVisible(hwnd):
                return True
            n = user32.GetWindowTextLengthW(hwnd)
            if n == 0:
                return True
            buf = ctypes.create_unicode_buffer(n + 1)
            user32.GetWindowTextW(hwnd, buf, n + 1)
            r = wintypes.RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(r))
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            title = buf.value
            app = title.rsplit(" - ", 1)[-1]
            res.append(WindowInfo(f"win:{hwnd}", app, title,
                                  BoundingBox(r.left, r.top, r.right - r.left, r.bottom - r.top), pid.value, len(res)))
            return True

        user32.EnumWindows(cb, 0)
        return res

    # -- frames
    def capture_frames(self, ts: Optional[float] = None) -> list[ViewportFrame]:
        ts = time.time() if ts is None else ts
        frames = []
        for info in self.list_windows():
            nodes: list[AXNode] = []
            if self.with_ax and self.system == "Darwin":
                try:
                    nodes = self._mac_ax(info)
                except Exception:
                    nodes = []
            if not nodes and info.title:
                nodes = [AXNode("AXWindowTitle", info.title)]
            px = None
            if self.with_pixels and self.system == "Darwin":
                try:
                    px = self._mac_pixels(info)
                except Exception:
                    px = None
            frames.append(ViewportFrame(
                window_id=info.window_id, app=info.app, title=info.title or info.app, bbox=info.bbox, ts=ts,
                app_category=categorize(info.app, info.title), bundle_id=info.bundle_id, pid=info.pid,
                z_index=info.z_index, ax_nodes=nodes, pixels=px, source=f"native:{self.backend}"))
        return frames


# ---------------------------------------------------------------------------
# Clipboard
# ---------------------------------------------------------------------------
def read_clipboard() -> Optional[str]:
    system = platform.system()
    if system == "Darwin":
        return _run(["pbpaste"])
    if system == "Linux":
        return _run(["wl-paste", "--no-newline"]) or _run(["xclip", "-o", "-selection", "clipboard"])
    if system == "Windows":
        try:
            import ctypes

            u, k = ctypes.windll.user32, ctypes.windll.kernel32  # type: ignore[attr-defined]
            if not u.OpenClipboard(0):
                return None
            try:
                h = u.GetClipboardData(13)  # CF_UNICODETEXT
                if not h:
                    return None
                p = k.GlobalLock(h)
                try:
                    return ctypes.wstring_at(p)
                finally:
                    k.GlobalUnlock(h)
            finally:
                u.CloseClipboard()
        except Exception:
            return None
    return None


class ClipboardWatcher:
    """Reports clipboard text when it changes (compared by hash; contents are not retained)."""

    def __init__(self, reader=read_clipboard):
        self._reader = reader
        self._last: Optional[str] = None

    def poll(self) -> Optional[str]:
        txt = self._reader()
        if not txt:
            return None
        h = hashlib.blake2b(txt.encode("utf-8", "ignore"), digest_size=16).hexdigest()
        if h == self._last:
            return None
        first = self._last is None
        self._last = h
        return None if first else txt[:4000]


# ---------------------------------------------------------------------------
# Git working tree
# ---------------------------------------------------------------------------
class GitWatcher:
    """Turns working-tree modifications into IDE-edit events (added lines as AX text)."""

    SKIP_DIRS = (".venv/", "venv/", "node_modules/", "__pycache__/", ".git/", "dist/", "build/", "target/")

    def __init__(self, repo: str | Path, max_lines: int = 60, max_files: int = 25, max_file_bytes: int = 256_000):
        self.repo = str(Path(repo).expanduser().resolve())
        self.max_lines = max_lines
        self.max_files = max_files  # bound work per poll (a fresh clone / unignored venv can list thousands)
        self.max_file_bytes = max_file_bytes
        self._seen: dict[str, str] = {}

    def poll(self, ts: Optional[float] = None) -> list[ViewportFrame]:
        ts = time.time() if ts is None else ts
        status = _run(["git", "-C", self.repo, "status", "--porcelain=v1", "--untracked-files=all"], timeout=5)
        if status is None:
            return []
        frames = []
        for line in status.splitlines():
            if len(frames) >= self.max_files:
                break
            path = line[3:].strip().strip('"').split(" -> ")[-1]
            if not path or path.endswith("/") or any(f"/{d}" in f"/{path}" for d in self.SKIP_DIRS):
                continue
            try:
                full = Path(self.repo, path)
                if full.is_file() and full.stat().st_size > self.max_file_bytes:
                    continue
            except OSError:
                continue
            diff = _run(["git", "-C", self.repo, "diff", "-U0", "--", path], timeout=5) or ""
            if not diff and line.startswith("??"):
                try:
                    diff = "\n".join("+" + l for l in Path(self.repo, path).read_text(errors="ignore").splitlines()[:self.max_lines])
                except OSError:
                    continue
            h = hashlib.blake2b(diff.encode("utf-8", "ignore"), digest_size=16).hexdigest()
            if self._seen.get(path) == h:
                continue
            self._seen[path] = h
            added = [l[1:] for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++")][: self.max_lines]
            nodes = [AXNode("AXStaticText", l, (0, i * 10, 1000, 10)) for i, l in enumerate(added)]
            repo_name = os.path.basename(self.repo)
            frames.append(ViewportFrame(
                window_id=f"git:{self.repo}", app="git", title=f"{os.path.basename(path)} — {repo_name} (working tree)",
                bbox=BoundingBox(0, 0, 1000, max(40, 10 * len(nodes))), ts=ts, app_category="ide",
                doc_path=path, ax_nodes=nodes, source="git"))
        return frames
