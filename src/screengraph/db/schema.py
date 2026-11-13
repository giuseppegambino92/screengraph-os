"""SQLite schema: WAL journal, external-content FTS5, binary vector blobs, causal edges, privacy audit.

No table has a column that can hold pixels. The largest blob per event is the
int8 patch matrix (P x 128 bytes = 2 KiB for a 4x4 grid); the privacy audit
asserts that no blob in the database exceeds ``MAX_VECTOR_BLOB_BYTES``.
"""

from __future__ import annotations

import sqlite3

SCHEMA_VERSION = 2
MAX_VECTOR_BLOB_BYTES = 16 * 1024

PRAGMAS = (
    # must precede WAL on a new file: a ~2.6 KB vector row wastes 37% of a 4 KB page (1 row/page);
    # 8 KB pages hold three rows
    "PRAGMA page_size=8192",
    "PRAGMA journal_mode=WAL",
    "PRAGMA synchronous=NORMAL",
    "PRAGMA temp_store=MEMORY",
    "PRAGMA foreign_keys=ON",
    "PRAGMA cache_size=-16000",
    "PRAGMA mmap_size=134217728",
)

DDL = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id             INTEGER PRIMARY KEY,
    ts             REAL    NOT NULL,
    window_id      TEXT    NOT NULL,
    app            TEXT    NOT NULL,
    app_category   TEXT    NOT NULL,
    bundle_id      TEXT,
    title          TEXT,
    pid            INTEGER,
    monitor_id     INTEGER,
    x INTEGER, y INTEGER, w INTEGER, h INTEGER,
    z_index        INTEGER,
    url            TEXT,
    doc_path       TEXT,
    body           TEXT,              -- scrubbed accessibility text (never raw)
    entities       TEXT,              -- JSON: scrubbed entity sets used for causal bridges
    clipboard_hash TEXT,              -- keyed HMAC of scrubbed, normalized clipboard text
    phash          INTEGER,
    dhash          INTEGER,
    source         TEXT,
    gate_reason    TEXT
);
CREATE INDEX IF NOT EXISTS events_ts      ON events(ts);
CREATE INDEX IF NOT EXISTS events_window  ON events(window_id, ts);
CREATE INDEX IF NOT EXISTS events_docpath ON events(doc_path);

CREATE VIRTUAL TABLE IF NOT EXISTS events_fts USING fts5(
    title, body, entities, app,
    content='events', content_rowid='id',
    tokenize="unicode61 tokenchars '_'"
);

CREATE VIRTUAL TABLE IF NOT EXISTS events_fts_vocab USING fts5vocab(events_fts, 'row');

CREATE TABLE IF NOT EXISTS event_vectors (
    event_id   INTEGER PRIMARY KEY REFERENCES events(id) ON DELETE CASCADE,
    dim        INTEGER NOT NULL,
    grid_rows  INTEGER NOT NULL,
    grid_cols  INTEGER NOT NULL,
    global_i8    BLOB NOT NULL, -- d x int8 (symmetric scalar quantization)
    global_scale REAL NOT NULL, -- dequantization scale
    global_bq    BLOB NOT NULL, -- d bits (uint64 words)
    patch_bq     BLOB NOT NULL, -- P x 64 bits: sign code of the 64-d Matryoshka prefix (stage-1 funnel)
    patch_i8     BLOB NOT NULL, -- P x store_dim int8 (Matryoshka prefix, stage-2 asymmetric rescoring)
    patch_scale  REAL NOT NULL, -- per-window dequantization scale
    concepts     TEXT           -- JSON: per-patch perceptual concepts (explainability)
);

CREATE TABLE IF NOT EXISTS causal_edges (
    src       INTEGER NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    dst       INTEGER NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    score     REAL NOT NULL,     -- linear causal score
    prob      REAL NOT NULL,     -- Hawkes branching responsibility P(parent = src | dst)
    relation  TEXT NOT NULL,
    c_time    REAL, c_sem REAL, c_bridge REAL, c_trans REAL,
    evidence  TEXT,              -- JSON list of bridging entities (scrubbed)
    PRIMARY KEY (src, dst)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS edges_dst ON causal_edges(dst);

CREATE TABLE IF NOT EXISTS app_transitions (
    src_cat TEXT NOT NULL,
    dst_cat TEXT NOT NULL,
    count   REAL NOT NULL,
    PRIMARY KEY (src_cat, dst_cat)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS redaction_log (
    id       INTEGER PRIMARY KEY,
    ts       REAL NOT NULL,
    event_id INTEGER,
    kind     TEXT NOT NULL,
    category TEXT NOT NULL,
    tag      TEXT NOT NULL       -- keyed hash tag only; the secret is never stored
);

CREATE TABLE IF NOT EXISTS privacy_audit (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class SchemaMismatch(RuntimeError):
    pass


def connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    for p in PRAGMAS:
        conn.execute(p)
    row = None
    try:
        row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    except sqlite3.OperationalError:
        pass
    if row is not None and int(row[0]) != SCHEMA_VERSION:
        conn.close()
        raise SchemaMismatch(f"{path} uses schema v{row[0]}, this build needs v{SCHEMA_VERSION}; "
                             f"use a fresh --home (vectors are not convertible: v2 changed the encoder)")
    conn.executescript(DDL)
    conn.execute("INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
    return conn
