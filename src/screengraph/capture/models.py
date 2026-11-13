"""Capture-layer schemas.

Hot-path objects are plain dataclasses (cheap to construct, no validation
overhead inside the <10 ms ingestion budget). Pydantic is used at the MCP
boundary instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

APP_CATEGORIES = ("terminal", "browser", "ide", "dashboard", "chat", "other")


@dataclass(frozen=True)
class BoundingBox:
    x: int
    y: int
    width: int
    height: int
    monitor_id: int = 0

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height, "monitor_id": self.monitor_id}


@dataclass
class AXNode:
    """One visible node of a window's accessibility tree.

    ``bbox`` is (x, y, w, h) in *window-local* pixel coordinates of the frame's
    pixel grid, so text can be assigned to spatial patches.
    """

    role: str
    text: str
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    focused: bool = False


@dataclass
class ViewportFrame:
    """A single raw observation of one window. Holds pixels ONLY in RAM."""

    window_id: str
    app: str
    title: str
    bbox: BoundingBox
    ts: float
    app_category: str = "other"
    bundle_id: str = ""
    pid: int = 0
    z_index: int = 0
    url: str = ""
    doc_path: str = ""
    ax_nodes: list[AXNode] = field(default_factory=list)
    clipboard_copied: Optional[str] = None  # text copied *from this window* during this tick
    pixels: Optional[np.ndarray] = None  # (H, W, 3) uint8 -- ephemeral, zeroed after feature extraction
    source: str = "sim"

    def ax_text(self) -> str:
        return "\n".join(n.text for n in self.ax_nodes if n.text)


@dataclass
class VisualGridPatch:
    row: int
    col: int
    vector: np.ndarray
    concepts: dict[str, float] = field(default_factory=dict)

    def screen_bbox(self, window: BoundingBox, rows: int, cols: int) -> BoundingBox:
        pw, ph = window.width / cols, window.height / rows
        return BoundingBox(
            int(window.x + self.col * pw), int(window.y + self.row * ph), int(pw), int(ph), window.monitor_id
        )


@dataclass
class ViewportEvent:
    """A persisted, privacy-scrubbed semantic window state (a causal-graph node)."""

    id: int
    ts: float
    window_id: str
    app: str
    app_category: str
    title: str
    text: str
    url: str = ""
    doc_path: str = ""
    entities: dict = field(default_factory=dict)
    bbox: Optional[BoundingBox] = None
    clipboard_hash: Optional[str] = None
    source: str = "sim"

    def label(self, n: int = 48) -> str:
        head = self.title or self.app
        return f"{self.app}: {head[:n]}"
