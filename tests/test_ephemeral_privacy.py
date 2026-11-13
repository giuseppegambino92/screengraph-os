"""RAM zeroing, zero pixels on disk, and Luhn / entropy / format secret redaction."""

from __future__ import annotations

import hashlib
import os
import sqlite3

import numpy as np
import pytest

from screengraph.capture.ephemeral import EphemeralFrameBuffer, ZeroingLedger
from screengraph.privacy.audit import scan_directory
from screengraph.privacy.scrubber import Scrubber, luhn_valid, shannon_entropy

KEY = b"unit-test-key"


# ---------------------------------------------------------------- RAM buffer
def test_buffer_zeroed_on_exit_and_ledger_records():
    led = ZeroingLedger()
    px = np.random.default_rng(0).integers(1, 255, (90, 80, 3), dtype=np.uint8)
    with EphemeralFrameBuffer(px, led) as buf:
        assert buf.pixels.any()
        s = buf.scratch((90, 80), np.float32)
        s[:] = 7.0
        t = buf.track(px.astype(np.float32))
    assert not px.any() and not s.any() and not t.any()
    snap = led.snapshot()
    assert snap["buffers_zeroed"] == 3 and snap["verification_failures"] == 0
    assert snap["bytes_zeroed"] == px.nbytes + s.nbytes + t.nbytes
    assert snap["chain_head"] != "0" * 64


def test_buffer_zeroed_even_when_extraction_raises():
    px = np.full((10, 10, 3), 200, np.uint8)
    with pytest.raises(RuntimeError):
        with EphemeralFrameBuffer(px, ZeroingLedger()):
            raise RuntimeError("encoder crashed")
    assert not px.any()


def test_buffer_inaccessible_after_wipe_and_rejects_readonly():
    px = np.ones((4, 4, 3), np.uint8)
    b = EphemeralFrameBuffer(px, ZeroingLedger())
    with b:
        pass
    with pytest.raises(RuntimeError):
        _ = b.pixels
    ro = np.ones((4, 4, 3), np.uint8)
    ro.flags.writeable = False
    with pytest.raises(ValueError):
        EphemeralFrameBuffer(ro)


def test_ledger_hash_chain_is_deterministic_and_order_sensitive():
    a, b, c = ZeroingLedger(), ZeroingLedger(), ZeroingLedger()
    for n in (10, 20, 30):
        a.record(n, True)
        b.record(n, True)
    for n in (30, 20, 10):
        c.record(n, True)
    assert a.chain_head == b.chain_head != c.chain_head


# --------------------------------------------------------- whole-session proof
def test_every_frame_wiped_after_ingest(session_run):
    assert all(t["zero_after"] for t in session_run["ticks"])
    led = session_run["sg"].ledger.snapshot()
    assert led["frames_wrapped"] == len(session_run["ticks"])
    assert led["verification_failures"] == 0


def test_zero_image_files_and_no_pixel_bytes_on_disk(session_run):
    sg = session_run["sg"]
    sg.store.size_bytes(checkpoint=True)
    scan = scan_directory(session_run["dir"])
    assert scan["image_magic_hits"] == 0 and scan["image_extension_hits"] == 0
    blob = b"".join(open(os.path.join(dp, f), "rb").read()
                    for dp, _, fs in os.walk(session_run["dir"]) for f in fs)
    # cryptographic check 1: no file on disk is a raw frame
    for dp, _, fs in os.walk(session_run["dir"]):
        for f in fs:
            digest = hashlib.sha256(open(os.path.join(dp, f), "rb").read()).hexdigest()
            assert digest not in session_run["digests"]
    # check 2: no high-variance 48-byte pixel window of any semantic frame occurs anywhere on disk
    leaked = [fp for fp in session_run["fingerprints"] if fp in blob]
    assert not leaked, f"{len(leaked)} pixel fingerprints found on disk"


def test_sqlite_has_no_pixel_sized_blobs(session_run):
    audit = session_run["sg"].privacy_audit()
    assert audit["zero_pixel_guarantee_holds"]
    assert audit["sqlite"]["pixel_blob_suspects"] == 0
    assert audit["sqlite"]["max_blob_bytes"] <= 16 * 1024


def test_all_injected_secrets_absent_from_database(session_run):
    sg, sim = session_run["sg"], session_run["sim"]
    sg.store.size_bytes(checkpoint=True)
    raw = open(sg.store.path, "rb").read()
    conn = sqlite3.connect(sg.store.path)
    texts = "\n".join(str(v) for (v,) in conn.execute("SELECT body || title || url || entities FROM events"))
    conn.close()
    assert len(sim.secrets) == 15
    for name, secret in sim.secrets.items():
        parts = [secret] if "\n" not in secret else [l for l in secret.split("\n") if "PRIVATE KEY" not in l]
        for p in parts:
            assert p.encode() not in raw, name
            assert p not in texts, name


def test_secret_copied_between_apps_keeps_a_linkable_tag(session_run):
    sg, sim = session_run["sg"], session_run["sim"]
    tag = sg.scrubber.tag_for("SECRET", sim.secrets["openai_key"])
    evs = [e for e in sg.store.timeline(0, 1e13, 10000) if tag in e["body"]]
    cats = {e["app_category"] for e in evs}
    assert {"terminal", "chat"} <= cats
    ids = {e["id"] for e in evs}
    cross = [e for e in sg.store.edges() if e["src"] in ids and e["dst"] in ids and e["relation"] == "CLIPBOARD_TRANSFER"]
    assert cross, "secret transfer terminal -> chat should be a CLIPBOARD_TRANSFER edge"


# ---------------------------------------------------------------- scrubber
def test_luhn():
    assert luhn_valid("4539578763621486")
    assert not luhn_valid("4539578763621487")
    assert not luhn_valid("1234")


def test_shannon_entropy():
    assert shannon_entropy("") == 0.0
    assert shannon_entropy("aaaa") == 0.0
    assert abs(shannon_entropy("ab") - 1.0) < 1e-9
    assert abs(shannon_entropy("abcdefghijklmnop") - 4.0) < 1e-9


@pytest.mark.parametrize("text,kind", [
    ("export OPENAI_API_KEY=" + "sk-" + "proj-" + "Ab3dEf6hIj9kLm2nOp5qRs8tUv1wXy4z", "openai_key"),
    ("aws " + "AK" + "IA" + "IOSFODNN7EXAMPLQ", "aws_access_key"),
    ("aws_secret_access_key = " + "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "aws_secret_key"),
    ("tok " + "eyJhbGciOiJIUzI1NiJ9" + ".eyJzdWIiOiIxMjM0NTYifQ.dozjgNryP4J3jVmNHl0w5N", "jwt"),
    ("Authorization: Bearer abcdefghijklmnop1234567890", "bearer_token"),
    ("remote https://" + "gh" + "p_" + "a" * 36 + "@github.com/x/y.git", "github_token"),
    ("card 4539 5787 6362 1486 ok", "credit_card"),
    ("DB_PASSWORD=hunter2-hunter2", "password_assignment"),
    ("mysql --password=Tr0ub4dor-xyz", "cli_secret_flag"),
    ("mail jane.doe@example.com", "email"),
    ("rotated to Zq8!xT4#Lp2@Wm9$Rv6^ ok", "high_entropy_token"),
    ("staging pw: Zq8!xT4#Lp2@Wm9$Rv6^ ok", "credential_context"),
    # regression (seed sweep): 19 distinct chars + sentence punctuation must not slip under the length gate
    ("staging db creds rotated, new one: 8Kx@7RmvZ4$6p&#3L29!", "credential_context"),
    ("then 8Kx@7RmvZ4$6p&#3L29! was used", "high_entropy_token"),
    ("x-api-key: Q2hhbmdlTWUhIFRoaXMgaXMgYSB0ZXN0IGtleSE9", "auth_header"),
    # regression (found by the seed sweep): base64 secrets containing '/' are not file paths
    ("x-api-key: /RpZood4l1HbRV3F3lCLxa/2r3fFbsuzbZ8sClP1", "auth_header"),
    ("SERVICE_TOKEN=qDUTVBAq7v5s6/48zUuk3/w3nChsKoVdlN3eG8gN", "password_assignment"),
    ("SIGNING_SEED=qDUTVBAq7v5s6/48zUuk3/w3nChsKoVdlN3eG8gN", "high_entropy_context"),
    ("blob qDUTVBAq7v5s6/48zUuk3/w3nChsKoVdlN3eG8gN end", "high_entropy_token"),
])
def test_each_secret_kind_is_redacted(text, kind):
    s = Scrubber(KEY)
    res = s.scrub(text)
    equivalent = {"cli_secret_flag": {"cli_secret_flag", "password_assignment"}}.get(kind, {kind})
    assert equivalent & {r.kind for r in res.redactions}, (kind, res.text)
    secret = text[res.redactions[0].start:res.redactions[0].end]
    assert secret not in res.text
    assert "<REDACTED_" in res.text


def test_pem_block_across_ax_lines_is_redacted():
    s = Scrubber(KEY)
    lines = ["$ cat key", "-----BEGIN OPENSSH " + "PRIVATE KEY-----", "b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAAB",
             "AAAAMwAAAAtzc2gtZWQyNTUxOQAAACD2ExampleExampleExampleExampleExample", "-----END OPENSSH " + "PRIVATE KEY-----", "$"]
    out, reds = s.scrub_lines(lines)
    assert len(out) == len(lines)
    assert any(r.kind == "pem_private_key" for r in reds)
    assert all("b3BlbnNz" not in l and "ExampleExample" not in l for l in out)
    assert out[0] == "$ cat key" and out[-1] == "$"


def test_tags_are_deterministic_keyed_and_idempotent():
    a, b = Scrubber(KEY), Scrubber(b"another-key")
    t = "token sk-" + "proj-" + "Z" * 30
    r1, r2 = a.scrub(t).text, a.scrub(t).text
    assert r1 == r2  # same secret -> same tag (cross-window linkability)
    assert r1 != b.scrub(t).text  # tags depend on the install key (no offline brute force)
    assert a.scrub(r1).text == r1  # scrubbing is idempotent


def test_no_false_positives_on_dev_artifacts():
    s = Scrubber(KEY)
    benign = [
        "File \"src/app/db_pool.py\", line 88, in acquire",
        "commit 9f2c1ab4d7e8a0b1c2d3e4f5a6b7c8d9e0f1a2b3",
        "13:02:11 GET /api/v1/session 401 12ms http://localhost:3000",
        "error[E0502]: cannot borrow `self.queue` as mutable",
        "SQLiteBusyException: database is locked",
        "kubectl describe pod payments-api-7d9f8b-x2k -n payments",
    ]
    for line in benign:
        assert s.scrub(line).text == line, line


def test_scrubber_fuzz_random_secret_sets():
    """200 independently generated secret sets, each embedded in realistic context lines."""
    from screengraph.capture.simulator import fake_secrets

    s = Scrubber(KEY)
    for seed in range(200):
        sec = fake_secrets(seed)
        lines = [f"x-api-key: {sec['api_header_key']}", f"AWS_SECRET_ACCESS_KEY={sec['aws_secret_key']}",
                 f"DB_PASSWORD={sec['db_password']}", f"mysql --password={sec['cli_password']}",
                 f"staging db creds rotated, new one: {sec['chat_password']}", f"rotated to {sec['chat_password']} ok",
                 f"export OPENAI_API_KEY={sec['openai_key']}", f"curl -H \"Authorization: Bearer {sec['jwt']}\"",
                 sec["google_api_key"], sec["slack_token"], sec["github_token"], sec["stripe_key"],
                 sec["anthropic_key"], sec["aws_access_key"], f"card {sec['credit_card']}"]
        out, _ = s.scrub_lines(lines)
        joined = "\n".join(out)
        for name, v in sec.items():
            if "\n" not in v:
                assert v not in joined, (seed, name)



def test_schema_guard_and_page_size(tmp_path):
    from screengraph.db.schema import SchemaMismatch, connect

    p = str(tmp_path / "x.db")
    c = connect(p)
    assert c.execute("PRAGMA page_size").fetchone()[0] == 8192
    c.execute("UPDATE meta SET value='1' WHERE key='schema_version'")
    c.close()
    with pytest.raises(SchemaMismatch):
        connect(p)
