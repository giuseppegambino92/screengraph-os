"""Central configuration: thresholds, decay constants, dimensions, paths, privacy settings."""

from __future__ import annotations

import math
import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def default_data_dir() -> Path:
    env = os.environ.get("SCREENGRAPH_HOME")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".screengraph"


@dataclass
class Config:
    # ---- storage -------------------------------------------------------
    data_dir: Path = field(default_factory=default_data_dir)
    db_name: str = "screengraph.db"

    # ---- capture / perceptual gate --------------------------------------
    phash_threshold: int = 4  # max confident-bit Hamming distance for "visually identical" (global pHash)
    quadrant_threshold: int = 2  # same, per 2x2 quadrant pHash (localized changes, e.g. a chart spike)
    dhash_threshold: int = 6  # max Hamming bits for dHash (gradient structure)
    phash_size: int = 32  # DCT input resolution (32x32 -> 8x8 low frequencies)

    # ---- embeddings -----------------------------------------------------
    dim: int = 256  # full embedding dimension
    matryoshka_dims: tuple[int, ...] = (64, 128, 256)
    patch_store_dim: int = 128  # Matryoshka prefix persisted as int8 (+ per-window scale) for rescoring
    prefilter_dim: int = 64  # Matryoshka prefix used for the 1-bit Hamming funnel
    grid_rows: int = 4
    grid_cols: int = 4
    visual_weight: float = 1.0  # perceptual colour concepts (+ optional UI layer / adapter)
    text_weight: float = 0.8  # positioned AX tokens
    descriptor_weight: float = 0.6  # generic Spectral-HOG-JL descriptor
    morphology: bool = True  # cross-modal morphological concept bank (generic primitives; default ON)
    ui_concepts: bool = False  # OPTIONAL hand-written UI-object detectors (ablation; simulator-tuned)
    vision_adapter: Optional[str] = None  # e.g. "open_clip:ViT-B-32/laion2b_s34b_b79k" if installed
    embed_seed: int = 0x5C2E

    # ---- causal graph (Hawkes-style kernel) ------------------------------
    causal_horizon_s: float = 300.0
    decay_lambda: float = math.log(2) / 90.0  # half-life of 90 s
    w_time: float = 0.25
    w_sem: float = 0.20
    w_bridge: float = 0.45
    w_trans: float = 0.10
    tau_causal: float = 0.40
    mention_discount: float = 0.35  # chat message that merely names a file/symbol (no error/clipboard/link)
    sem_only_min: float = 0.55  # an edge with no entity bridge needs at least this cosine
    hawkes_mu: float = 0.15  # background (immigrant) intensity in parent responsibility
    max_parents: int = 6
    entity_df_window: int = 200  # recent nodes used for entity document-frequency pruning
    entity_df_max: float = 0.35  # entities seen in more than this share of recent nodes are ignored

    # ---- retrieval ------------------------------------------------------
    rrf_k: int = 60
    list_depth: int = 50
    prefilter_candidates: int = 256

    # ---- privacy ---------------------------------------------------------
    entropy_threshold: float = 4.2
    redact_emails: bool = True
    redact_ips: bool = True
    hmac_key: bytes | None = None  # if None, loaded/created at data_dir/.hmac_key (0600)

    @property
    def db_path(self) -> Path:
        return Path(self.data_dir) / self.db_name

    def ensure_dirs(self) -> None:
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)

    def resolve_hmac_key(self) -> bytes:
        """Per-install key for deterministic-but-unguessable redaction hashes.

        The key never enters the database, so a stolen DB cannot be used to
        brute-force low-entropy secrets (card numbers, emails) from their tags.
        """
        if self.hmac_key is not None:
            return self.hmac_key
        self.ensure_dirs()
        key_path = Path(self.data_dir) / ".hmac_key"
        if key_path.exists():
            self.hmac_key = key_path.read_bytes()
        else:
            key = secrets.token_bytes(32)
            fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as fh:
                fh.write(key)
            self.hmac_key = key
        return self.hmac_key

    @classmethod
    def ephemeral(cls, data_dir: str | Path, **overrides) -> "Config":
        """Config for tests/benchmarks: fixed HMAC key, isolated directory."""
        cfg = cls(data_dir=Path(data_dir), hmac_key=b"screengraph-deterministic-test-key", **overrides)
        cfg.ensure_dirs()
        return cfg
