"""Live macOS capture without pyobjc: CoreGraphics / CoreFoundation / ApplicationServices via ctypes.

* Window viewports: ``CGWindowListCopyWindowInfo`` (on-screen, layer 0): window number, owner app,
  owner pid, bounds; titles are present only with Screen Recording permission.
* Pixels: ``CGWindowListCreateImage`` straight into RAM (no file, no subprocess). The OS hands us a
  CFData copy of the bitmap; we area-average it 1/8 into a fresh writable numpy array (the
  EphemeralFrameBuffer the engine wipes), then ``memset`` the CFData bytes to 0 before ``CFRelease``.
  The CGImage's own backing store belongs to CoreGraphics and can only be released.
* Text: ``AXUIElement`` traversal of each app's focused window when the process is Accessibility-
  trusted (``AXIsProcessTrusted``). We never trigger the permission prompt unless asked (``request_ax``).
* Clipboard: only with ``allow_clipboard=True`` (``pbpaste``), attributed to the frontmost window.

``run_live_capture`` returns an aggregate summary that contains no window titles, app names or text,
so it can be committed to a report.
"""

from __future__ import annotations

import ctypes
import platform
import shutil
import subprocess
import tempfile
import time
from ctypes import POINTER, Structure, byref, c_bool, c_char_p, c_double, c_int32, c_long, c_uint32, c_void_p
from pathlib import Path
from typing import Optional

import numpy as np

from screengraph.capture.models import AXNode, BoundingBox, ViewportFrame
from screengraph.capture.os_hooks import categorize

_UTF8 = 0x08000100
kCGWindowListOptionOnScreenOnly = 1
kCGWindowListExcludeDesktopElements = 16
kCGWindowListOptionIncludingWindow = 8
kCGWindowImageBoundsIgnoreFraming = 1
kCGWindowImageNominalResolution = 16


class CGRect(Structure):
    _fields_ = [("x", c_double), ("y", c_double), ("w", c_double), ("h", c_double)]


class _CF:
    """Minimal typed bindings (loaded lazily; macOS only)."""

    def __init__(self):
        self.cg = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
        self.cf = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
        self.ax = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices")
        f = self._fn
        cf, cg, ax = self.cf, self.cg, self.ax
        self.ArrayCount = f(cf, "CFArrayGetCount", c_long, [c_void_p])
        self.ArrayAt = f(cf, "CFArrayGetValueAtIndex", c_void_p, [c_void_p, c_long])
        self.DictGet = f(cf, "CFDictionaryGetValue", c_void_p, [c_void_p, c_void_p])
        self.StrCreate = f(cf, "CFStringCreateWithCString", c_void_p, [c_void_p, c_char_p, c_uint32])
        self.StrGet = f(cf, "CFStringGetCString", c_bool, [c_void_p, c_char_p, c_long, c_uint32])
        self.StrLen = f(cf, "CFStringGetLength", c_long, [c_void_p])
        self.NumGet = f(cf, "CFNumberGetValue", c_bool, [c_void_p, c_long, c_void_p])
        self.Release = f(cf, "CFRelease", None, [c_void_p])
        self.TypeID = f(cf, "CFGetTypeID", c_long, [c_void_p])
        self.StrTypeID = f(cf, "CFStringGetTypeID", c_long, [])()
        self.ArrTypeID = f(cf, "CFArrayGetTypeID", c_long, [])()
        self.DataLen = f(cf, "CFDataGetLength", c_long, [c_void_p])
        self.DataPtr = f(cf, "CFDataGetBytePtr", c_void_p, [c_void_p])
        self.RectFromDict = f(cg, "CGRectMakeWithDictionaryRepresentation", c_bool, [c_void_p, POINTER(CGRect)])
        self.WinInfo = f(cg, "CGWindowListCopyWindowInfo", c_void_p, [c_uint32, c_uint32])
        self.CreateImage = f(cg, "CGWindowListCreateImage", c_void_p, [CGRect, c_uint32, c_uint32, c_uint32]) \
            if hasattr(cg, "CGWindowListCreateImage") else None
        self.ImgW = f(cg, "CGImageGetWidth", ctypes.c_size_t, [c_void_p])
        self.ImgH = f(cg, "CGImageGetHeight", ctypes.c_size_t, [c_void_p])
        self.ImgBPR = f(cg, "CGImageGetBytesPerRow", ctypes.c_size_t, [c_void_p])
        self.ImgBPP = f(cg, "CGImageGetBitsPerPixel", ctypes.c_size_t, [c_void_p])
        self.ImgProvider = f(cg, "CGImageGetDataProvider", c_void_p, [c_void_p])
        self.ProviderCopy = f(cg, "CGDataProviderCopyData", c_void_p, [c_void_p])
        self.ScreenAccess = f(cg, "CGPreflightScreenCaptureAccess", c_bool, []) \
            if hasattr(cg, "CGPreflightScreenCaptureAccess") else None
        self.AXTrusted = f(ax, "AXIsProcessTrusted", c_bool, [])
        self.AXApp = f(ax, "AXUIElementCreateApplication", c_void_p, [c_int32])
        self.AXCopy = f(ax, "AXUIElementCopyAttributeValue", c_int32, [c_void_p, c_void_p, POINTER(c_void_p)])
        self._keys: dict[str, int] = {}

    @staticmethod
    def _fn(lib, name, res, args):
        fn = getattr(lib, name)
        fn.restype, fn.argtypes = res, args
        return fn

    def key(self, s: str) -> int:
        k = self._keys.get(s)
        if k is None:
            k = self._keys[s] = self.StrCreate(None, s.encode(), _UTF8)  # kept for process lifetime
        return k

    def pystr(self, ref) -> str:
        if not ref or self.TypeID(ref) != self.StrTypeID:
            return ""
        n = self.StrLen(ref) * 4 + 1
        buf = ctypes.create_string_buffer(n)
        return buf.value.decode("utf-8", "replace") if self.StrGet(ref, buf, n, _UTF8) else ""

    def pyint(self, ref) -> int:
        v = ctypes.c_int64()
        if ref:
            self.NumGet(ref, 4, byref(v))  # kCFNumberSInt64Type
        return int(v.value)

    def request_ax_prompt(self) -> bool:  # pragma: no cover - interactive OS dialog
        fn = self._fn(self.ax, "AXIsProcessTrustedWithOptions", c_bool, [c_void_p])
        create = self._fn(self.cf, "CFDictionaryCreate", c_void_p,
                          [c_void_p, POINTER(c_void_p), POINTER(c_void_p), c_long, c_void_p, c_void_p])
        k = c_void_p.in_dll(self.ax, "kAXTrustedCheckOptionPrompt")
        t = c_void_p.in_dll(self.cf, "kCFBooleanTrue")
        kcb = ctypes.addressof(c_void_p.in_dll(self.cf, "kCFTypeDictionaryKeyCallBacks"))
        vcb = ctypes.addressof(c_void_p.in_dll(self.cf, "kCFTypeDictionaryValueCallBacks"))
        keys, vals = (c_void_p * 1)(k.value), (c_void_p * 1)(t.value)
        d = create(None, keys, vals, 1, kcb, vcb)
        try:
            return bool(fn(d))
        finally:
            self.Release(d)


class MacLiveInspector:
    def __init__(self, scale: int = 8, with_pixels: bool = True, with_ax: bool = True, min_size: int = 120):
        if platform.system() != "Darwin":
            raise RuntimeError("MacLiveInspector requires macOS")
        self.cf = _CF()
        self.scale, self.with_pixels, self.with_ax, self.min_size = scale, with_pixels, with_ax, min_size

    def permissions(self) -> dict:
        return {"screen_recording": bool(self.cf.ScreenAccess()) if self.cf.ScreenAccess else None,
                "accessibility": bool(self.cf.AXTrusted()),
                "cg_window_image_available": self.cf.CreateImage is not None}

    # -- viewports -------------------------------------------------------------
    def windows(self) -> list[dict]:
        cf = self.cf
        arr = cf.WinInfo(kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements, 0)
        out = []
        if not arr:
            return out
        try:
            for i in range(cf.ArrayCount(arr)):
                d = cf.ArrayAt(arr, i)
                if cf.pyint(cf.DictGet(d, cf.key("kCGWindowLayer"))) != 0:
                    continue
                r = CGRect()
                cf.RectFromDict(cf.DictGet(d, cf.key("kCGWindowBounds")), byref(r))
                if r.w < self.min_size or r.h < self.min_size:
                    continue
                out.append({"number": cf.pyint(cf.DictGet(d, cf.key("kCGWindowNumber"))),
                            "owner": cf.pystr(cf.DictGet(d, cf.key("kCGWindowOwnerName"))),
                            "title": cf.pystr(cf.DictGet(d, cf.key("kCGWindowName"))),
                            "pid": cf.pyint(cf.DictGet(d, cf.key("kCGWindowOwnerPID"))),
                            "bbox": (int(r.x), int(r.y), int(r.w), int(r.h)), "z": len(out)})
        finally:
            cf.Release(arr)
        return out

    # -- pixels ------------------------------------------------------------------
    def grab(self, number: int) -> Optional[np.ndarray]:
        """Window bitmap -> (H/s, W/s, 3) uint8 RGB area-average, entirely in RAM. Source bytes are zeroed."""
        cf = self.cf
        if cf.CreateImage is None:
            return None
        null = CGRect(float("inf"), float("inf"), 0.0, 0.0)  # CGRectNull: the window's own bounds
        img = cf.CreateImage(null, kCGWindowListOptionIncludingWindow, number,
                             kCGWindowImageBoundsIgnoreFraming | kCGWindowImageNominalResolution)
        if not img:
            return None
        data = None
        try:
            w, h, bpr, bpp = cf.ImgW(img), cf.ImgH(img), cf.ImgBPR(img), cf.ImgBPP(img)
            if bpp != 32 or w < self.scale or h < self.scale:
                return None
            data = cf.ProviderCopy(cf.ImgProvider(img))
            n = cf.DataLen(data)
            ptr = cf.DataPtr(data)
            raw = np.ctypeslib.as_array(ctypes.cast(ptr, POINTER(ctypes.c_uint8)), shape=(n,))
            s = self.scale
            H, W = (h // s) * s, (w // s) * s
            bgra = raw[: h * bpr].reshape(h, bpr)[:H, : W * 4].reshape(H // s, s, W // s, s, 4)
            acc = bgra[..., :3].sum(axis=(1, 3), dtype=np.uint32)  # area average per s x s block
            out = np.empty((H // s, W // s, 3), np.uint8)
            np.floor_divide(acc[..., ::-1], s * s, out=out, casting="unsafe")  # BGR -> RGB
            acc.fill(0)
            ctypes.memset(ptr, 0, n)  # wipe our private CFData copy of the bitmap before release
            return out
        finally:
            if data:
                cf.Release(data)
            cf.Release(img)

    # -- accessibility text ---------------------------------------------------------
    def ax_text(self, pid: int, max_nodes: int = 300, budget_s: float = 0.05) -> list[str]:
        cf = self.cf
        if not cf.AXTrusted():
            return []
        app = cf.AXApp(pid)
        if not app:
            return []
        out: list[str] = []
        owned = [app]
        try:
            win = c_void_p()
            if cf.AXCopy(app, cf.key("AXFocusedWindow"), byref(win)) != 0 or not win.value:
                return []
            owned.append(win.value)
            stack, t0 = [(win.value, 0)], time.perf_counter()
            while stack and len(out) < max_nodes and time.perf_counter() - t0 < budget_s:
                el, depth = stack.pop()
                for attr in ("AXValue", "AXTitle", "AXDescription"):
                    v = c_void_p()
                    if cf.AXCopy(el, cf.key(attr), byref(v)) == 0 and v.value:
                        owned.append(v.value)
                        txt = cf.pystr(v.value).strip()
                        if txt:
                            out.append(txt[:2000])
                            break
                if depth < 12:
                    kids = c_void_p()
                    if cf.AXCopy(el, cf.key("AXChildren"), byref(kids)) == 0 and kids.value:
                        owned.append(kids.value)
                        if cf.TypeID(kids.value) == cf.ArrTypeID:
                            stack.extend((cf.ArrayAt(kids.value, k), depth + 1) for k in range(cf.ArrayCount(kids.value)))
        finally:
            for ref in owned:
                cf.Release(ref)
        return out

    # -- frames ----------------------------------------------------------------------
    def capture(self, ts: Optional[float] = None) -> tuple[list[ViewportFrame], dict]:
        ts = time.time() if ts is None else ts
        stats = {"enumerate_ms": 0.0, "grab_ms": [], "ax_ms": [], "pixel_frames": 0, "ax_frames": 0}
        t0 = time.perf_counter()
        wins = self.windows()
        stats["enumerate_ms"] = (time.perf_counter() - t0) * 1e3
        frames = []
        for w in wins:
            x, y, ww, hh = w["bbox"]
            px = None
            if self.with_pixels:
                t1 = time.perf_counter()
                px = self.grab(w["number"])
                stats["grab_ms"].append((time.perf_counter() - t1) * 1e3)
                stats["pixel_frames"] += px is not None
            lines: list[str] = []
            if self.with_ax:
                t2 = time.perf_counter()
                lines = self.ax_text(w["pid"])
                stats["ax_ms"].append((time.perf_counter() - t2) * 1e3)
                stats["ax_frames"] += bool(lines)
            gh = px.shape[0] if px is not None else max(1, hh // self.scale)
            gw = px.shape[1] if px is not None else max(1, ww // self.scale)
            nodes = [AXNode("AXStaticText", l, (0, int(i * gh / max(1, len(lines))), gw, max(1, gh // max(1, len(lines)))))
                     for i, l in enumerate(lines)]
            if not nodes and w["title"]:
                nodes = [AXNode("AXWindowTitle", w["title"], (0, 0, gw, 3))]
            frames.append(ViewportFrame(
                window_id=f"mac:{w['number']}", app=w["owner"] or "unknown", title=w["title"] or w["owner"],
                bbox=BoundingBox(x, y, ww, hh, 0), ts=ts, app_category=categorize(w["owner"], w["title"]),
                pid=w["pid"], z_index=w["z"], ax_nodes=nodes, pixels=px, source="native:ctypes-cg"))
        return frames, stats


def _read_clipboard() -> Optional[str]:
    try:
        r = subprocess.run(["pbpaste"], capture_output=True, text=True, timeout=2)
        return r.stdout if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _lat(a: list[float]) -> dict:
    if not a:
        return {"n": 0}
    x = np.asarray(a)
    return {"n": len(a), "p50_ms": round(float(np.percentile(x, 50)), 3), "p95_ms": round(float(np.percentile(x, 95)), 3),
            "max_ms": round(float(x.max()), 3)}


def run_live_capture(ticks: int = 5, interval: float = 1.0, data_dir: Optional[Path] = None, pixels: bool = True,
                     allow_clipboard: bool = False, request_ax: bool = False, keep: bool = False,
                     engine=None) -> dict:
    """Capture real desktop ticks into a (temporary) DB and return an aggregate, title-free summary."""
    if platform.system() != "Darwin":
        return {"status": "skipped", "reason": f"live capture implemented for macOS; this is {platform.system()}"}
    from screengraph.capture.os_hooks import ClipboardWatcher
    from screengraph.config import Config
    from screengraph.engine import ScreenGraph

    tmp = None
    owns_engine = engine is None
    if engine is None:
        if data_dir is None:
            tmp = Path(tempfile.mkdtemp(prefix="screengraph-live-"))
            data_dir = tmp
        engine = ScreenGraph(Config(data_dir=Path(data_dir)))
    insp = MacLiveInspector(with_pixels=pixels)
    if request_ax and not insp.cf.AXTrusted():
        insp.cf.request_ax_prompt()
    perms = insp.permissions()
    clip = ClipboardWatcher(_read_clipboard) if allow_clipboard else None
    if clip:
        clip.poll()  # prime: never ingest what was on the clipboard before we started
    ingest_ms, gate_ms, encode_ms, cap_ms, grab, axms = [], [], [], [], [], []
    frames_seen = stored = dropped = pix = axf = 0
    nonzero_after = 0
    windows_per_tick = []
    try:
        for k in range(ticks):
            t0 = time.perf_counter()
            frames, st = insp.capture()
            cap_ms.append((time.perf_counter() - t0) * 1e3)
            grab += st["grab_ms"]
            axms += st["ax_ms"]
            pix += st["pixel_frames"]
            axf += st["ax_frames"]
            windows_per_tick.append(len(frames))
            copied = clip.poll() if clip else None
            if copied and frames:
                frames[0].clipboard_copied = copied
            for f in frames:
                r = engine.ingest_frame(f)
                frames_seen += 1
                stored += r.stored
                dropped += not r.stored
                ingest_ms.append(r.timings_ms.get("total_ms", 0.0))
                gate_ms.append(r.timings_ms.get("gate_ms", 0.0))
                if "encode_bq_wipe_ms" in r.timings_ms:
                    encode_ms.append(r.timings_ms["encode_bq_wipe_ms"])
                if f.pixels is not None and f.pixels.any():
                    nonzero_after += 1
            if k < ticks - 1:
                time.sleep(max(0.0, interval - (time.perf_counter() - t0)))
        ws = engine.active_workspace()
        audit = engine.privacy_audit()
        return {
            "status": "ok", "platform": platform.platform(), "backend": "ctypes CoreGraphics/CoreFoundation/AX",
            "permissions": perms, "ticks": ticks, "interval_s": interval,
            "windows_per_tick": windows_per_tick, "frames_seen": frames_seen, "frames_with_pixels": pix,
            "frames_with_ax_text": axf, "events_stored": stored, "frames_dropped_by_gate": dropped,
            "os_capture_ms_per_tick": _lat(cap_ms), "os_window_grab_ms": _lat(grab), "os_ax_ms": _lat(axms),
            "ingest_ms": _lat(ingest_ms), "gate_ms": _lat(gate_ms), "encode_bq_wipe_ms": _lat(encode_ms),
            "ingest_under_10ms": bool(ingest_ms) and max(ingest_ms) < 10.0,
            "active_workspace_windows": len(ws["windows"]),
            "image_files_on_disk": audit["disk"].get("image_magic_hits", 0) + audit["disk"].get("image_extension_hits", 0),
            "pixel_blob_suspects": audit["sqlite"]["pixel_blob_suspects"],
            "frames_nonzero_after_ingest": nonzero_after,
            "ram_wipe_verification_failures": audit["ram"]["verification_failures"],
            "zero_pixel_guarantee_holds": audit["zero_pixel_guarantee_holds"] and nonzero_after == 0,
            "clipboard_enabled": allow_clipboard,
            "note": "aggregate only: no window titles, app names or text are included in this summary",
        }
    finally:
        if owns_engine:
            engine.close()
        if tmp is not None and not keep:
            shutil.rmtree(tmp, ignore_errors=True)
