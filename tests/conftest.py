from __future__ import annotations

import hashlib

import numpy as np
import pytest

from screengraph.capture.simulator import WorkspaceSimulator
from screengraph.config import Config
from screengraph.engine import ScreenGraph


def pixel_fingerprints(px: np.ndarray, k: int = 4, width: int = 48) -> list[bytes]:
    """High-variance raw byte windows of a frame: if any of them shows up on disk, pixels leaked."""
    flat = px.reshape(px.shape[0], -1)
    out = []
    scores = [(len(set(row[i:i + width].tobytes())), r, i) for r, row in enumerate(flat)
              for i in range(0, flat.shape[1] - width, width)]
    scores.sort(reverse=True)
    for _, r, i in scores[:k]:
        out.append(flat[r, i:i + width].tobytes())
    return out


@pytest.fixture(scope="session")
def session_run(tmp_path_factory):
    """One full deterministic 450-tick session, ingested once and shared by tests."""
    d = tmp_path_factory.mktemp("session")
    sg = ScreenGraph(Config.ephemeral(d))
    sim = WorkspaceSimulator(seed=7)
    ticks, fingerprints, frame_digests = [], [], set()
    for tk in sim.session():
        px = tk.frame.pixels
        if tk.kind == "change":
            fingerprints += pixel_fingerprints(px)
            frame_digests.add(hashlib.sha256(px.tobytes()).hexdigest())
        r = sg.ingest_frame(tk.frame)
        ticks.append({"tick": tk, "result": r, "zero_after": not tk.frame.pixels.any()})
    yield {"sg": sg, "sim": sim, "ticks": ticks, "dir": d, "fingerprints": fingerprints, "digests": frame_digests}
    sg.close()


@pytest.fixture(scope="session")
def session_run_ui(tmp_path_factory):
    """Same session with the OPTIONAL simulator-tuned UI-object detectors (ablation)."""
    d = tmp_path_factory.mktemp("session_ui")
    sg = ScreenGraph(Config.ephemeral(d, ui_concepts=True))
    sim = WorkspaceSimulator(seed=7)
    ticks = []
    for tk in sim.session():
        r = sg.ingest_frame(tk.frame)
        ticks.append({"tick": tk, "result": r, "zero_after": not tk.frame.pixels.any()})
    yield {"sg": sg, "sim": sim, "ticks": ticks, "dir": d}
    sg.close()


@pytest.fixture()
def engine(tmp_path):
    sg = ScreenGraph(Config.ephemeral(tmp_path))
    yield sg
    sg.close()
