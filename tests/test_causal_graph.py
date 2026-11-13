"""Terminal -> clipboard -> browser -> IDE causal inference and backward "why" tracing."""

from __future__ import annotations

from collections import defaultdict

from screengraph.capture.simulator import WorkspaceSimulator
from screengraph.config import Config
from screengraph.engine import ScreenGraph
from screengraph.graph.causal_engine import extract_entities

T0 = 1_731_500_000.0


def test_entity_extraction():
    e = extract_entities(
        'dev@mbp ~/src/acme-api % pytest\n  File "src/app/db_pool.py", line 88\n'
        "app.errors.SQLiteBusyException: database is locked\nGET /api/v1/session 401 http://localhost:3000/x\n"
        "see https://www.sqlite.org/pragma.html#pragma_busy_timeout and PRAGMA busy_timeout\n"
        "error[E0502]: cannot borrow  payments-api-7d9f8b-x2k  <REDACTED_SECRET:0123456789>")
    assert "db_pool.py" in e["paths"]
    assert {"sqlitebusyexception", "e0502"} <= set(e["errors"])
    assert "sqlite.org/pragma.html" in e["urls"]
    assert "localhost:3000" in e["hosts"]
    assert "/api/v1/session" in e["routes"]
    assert "busy_timeout" in e["symbols"]
    assert {"payments-api-7d9f8b-x2k", "payments-api"} <= set(e["resources"])
    assert "<REDACTED_SECRET:0123456789>" in e["secrets"]
    assert "acme-api" not in e["resources"]  # shell prompt is chrome, not content
    assert "error" in e["flags"]


def _micro(engine: ScreenGraph):
    r = {}
    r["err"] = engine.ingest_text_event(
        "iTerm2", "terminal", 'pytest\n  File "src/app/db_pool.py", line 88, in acquire\n'
        "app.errors.SQLiteBusyException: database is locked", window_id="term", ts=T0)
    r["copy"] = engine.ingest_text_event(
        "iTerm2", "terminal", 'pytest\n  File "src/app/db_pool.py", line 88, in acquire\n'
        "app.errors.SQLiteBusyException: database is locked\n[selected] SQLiteBusyException: database is locked",
        window_id="term", ts=T0 + 20, clipboard="SQLiteBusyException: database is locked")
    r["chat"] = engine.ingest_text_event("Slack", "chat", "priya: lunch at 12?", window_id="chat", ts=T0 + 30)
    r["search"] = engine.ingest_text_event(
        "Chrome", "browser", "SQLiteBusyException: database is locked\n"
        "PRAGMA busy_timeout - SQLite https://www.sqlite.org/pragma.html#pragma_busy_timeout",
        url="https://www.google.com/search?q=SQLiteBusyException%3A+database+is+locked", window_id="br", ts=T0 + 50)
    r["docs"] = engine.ingest_text_event(
        "Chrome", "browser", "PRAGMA busy_timeout = milliseconds;\nCombine with journal_mode=WAL",
        url="https://www.sqlite.org/pragma.html#pragma_busy_timeout", window_id="br", ts=T0 + 90)
    r["fix"] = engine.ingest_text_event(
        "Code", "ide", 'conn.execute("PRAGMA journal_mode=WAL")\nconn.execute("PRAGMA busy_timeout = 5000")',
        doc_path="src/app/db_pool.py", window_id="ide", ts=T0 + 140)
    return r


def test_cross_app_edges_and_relation_types(engine):
    r = _micro(engine)
    rel = {(e["src"], e["dst"]): e["relation"] for e in engine.store.edges()}
    E = {k: v.event_id for k, v in r.items()}
    assert rel.get((E["copy"], E["search"])) == "CLIPBOARD_TRANSFER"
    assert rel.get((E["search"], E["docs"])) == "LINK_FOLLOW"
    assert rel.get((E["docs"], E["fix"])) == "RESEARCH_TO_CODE_EDIT"
    assert rel.get((E["err"], E["fix"])) == "ERROR_TO_CODE_EDIT"
    # pure temporal proximity is not causation: the lunch message links to nothing
    assert not any(E["chat"] in k for k in rel)


def test_why_traces_back_to_the_terminal_error(engine):
    r = _micro(engine)
    res = engine.why("src/app/db_pool.py")
    assert res["found"]
    assert res["root_causes"][0]["event_id"] == r["err"].event_id
    chain = {c["event_id"] for c in res["chain"]}
    assert {r[k].event_id for k in ("err", "copy", "search", "docs", "fix")} <= chain
    assert r["chat"].event_id not in chain
    assert "graph LR" in res["mermaid"] and "CLIPBOARD_TRANSFER" in res["mermaid"]


def test_why_by_symbol_resolves_to_the_edit(engine):
    r = _micro(engine)
    res = engine.why("busy_timeout")
    assert res["target_events"][0]["event_id"] == r["fix"].event_id
    assert res["root_causes"][0]["event_id"] == r["err"].event_id


def test_hawkes_responsibilities_are_a_distribution(engine):
    _micro(engine)
    by_dst = defaultdict(float)
    for e in engine.store.edges():
        assert 0 < e["prob"] <= 1 and e["score"] >= engine.cfg.tau_causal
        by_dst[e["dst"]] += e["prob"]
    assert all(s < 1.0 for s in by_dst.values())  # remainder = background (immigrant) probability mu


def test_horizon_limits_edges(engine):
    engine.ingest_text_event("iTerm2", "terminal", "app.errors.SQLiteBusyException: database is locked",
                             window_id="t", ts=T0)
    late = engine.ingest_text_event("Chrome", "browser", "SQLiteBusyException: database is locked", window_id="b",
                                    ts=T0 + engine.cfg.causal_horizon_s + 5)
    assert late.stored and late.edges == []


def test_forward_walk_finds_effects(engine):
    r = _micro(engine)
    eff = {n for n, _, _ in engine.graph.walk_forward(r["err"].event_id)}
    assert r["fix"].event_id in eff


def test_graph_survives_restart(tmp_path):
    cfg = Config.ephemeral(tmp_path)
    with ScreenGraph(cfg) as sg:
        r = _micro(sg)
    with ScreenGraph(Config.ephemeral(tmp_path)) as sg2:
        res = sg2.why("db_pool.py")
        assert res["root_causes"][0]["event_id"] == r["err"].event_id
        assert sg2.dense.n == 6 and sg2.patches.n == 6


def _roles(ticks):
    roles = defaultdict(dict)
    for t in ticks:
        a = t["tick"].action
        if a and a.chain and t["result"].stored:
            roles[a.chain].setdefault(a.role, []).append(t["result"].event_id)
    return roles


def test_full_session_root_cause_tracing_default_encoder(session_run):
    """Default (generic) encoder. C4's true root is a canvas-only Grafana memory spike: without a learned or
    hand-tuned spike detector nothing marks that frame as an anomaly, so the root lands on the first *textual*
    failure (kubectl OOMKilled), one hop downstream. Chains must still be 100% precise (no hard negatives)."""
    sg, roles = session_run["sg"], _roles(session_run["ticks"])
    for cid, path in WorkspaceSimulator.CHAIN_TARGETS.items():
        res = sg.why(path)
        root = res["root_causes"][0]["event_id"]
        gold_roots = set(roles[cid].get("root", [])) | (set(roles[cid].get("copy", [])) if cid != "C4" else set())
        if cid == "C4":
            assert root in gold_roots | set(roles[cid]["symptom"]), cid
        else:
            assert root in gold_roots, cid
        gold_chain = {e for role, ids in roles[cid].items() if role != "fix" for e in ids}
        pred = {c["event_id"] for c in res["chain"]} - set(roles[cid]["fix"])
        assert pred <= gold_chain, (cid, pred - gold_chain)  # 100% chain precision
        missing = gold_chain - pred
        assert not missing or (cid == "C4" and missing <= set(roles[cid]["root"])), (cid, missing)


def test_full_session_root_cause_tracing_ui_concepts_ablation(session_run_ui):
    sg, roles = session_run_ui["sg"], _roles(session_run_ui["ticks"])
    for cid, path in WorkspaceSimulator.CHAIN_TARGETS.items():
        res = sg.why(path)
        gold_roots = set(roles[cid].get("root", [])) | (set(roles[cid].get("copy", [])) if cid != "C4" else set())
        assert res["root_causes"][0]["event_id"] in gold_roots, cid
        gold_chain = {e for role, ids in roles[cid].items() if role != "fix" for e in ids}
        pred = {c["event_id"] for c in res["chain"]} - set(roles[cid]["fix"])
        assert pred == gold_chain, (cid, pred ^ gold_chain)


def test_relation_types_present_in_session(session_run):
    rels = {e["relation"] for e in session_run["sg"].store.edges()}
    for r in ("CLIPBOARD_TRANSFER", "ERROR_TO_RESEARCH", "RESEARCH_TO_CODE_EDIT", "UI_TO_LOG_CORRELATION",
              "SEMANTIC_CONTINUATION", "LINK_FOLLOW", "ERROR_TO_CODE_EDIT"):
        assert r in rels, r


def test_mention_only_chat_is_discounted(engine):
    T = T0
    err = engine.ingest_text_event("iTerm2", "terminal", "TokenExpiredError: jwt expired\n at verifyToken (src/auth/middleware.ts:41:11)",
                                   window_id="term", ts=T)
    chat = engine.ingest_text_event("Slack", "chat", "marco: fyi I'll touch middleware.ts later for logging",
                                    window_id="chat", ts=T + 30)
    fix = engine.ingest_text_event("Code", "ide", "jwt.verify(token, secret, { clockTolerance: 30 })",
                                   doc_path="src/auth/middleware.ts", window_id="ide", ts=T + 90)
    srcs = {e.src for e in fix.edges}
    assert err.event_id in srcs and chat.event_id not in srcs
    res = engine.why("src/auth/middleware.ts")
    assert chat.event_id not in {c["event_id"] for c in res["chain"]}
    # the same chat source WITH error content keeps full strength
    help_ = engine.ingest_text_event("Slack", "chat", "anyone seen PoolTimeoutError in src/db/pool.py ?",
                                     window_id="chat2", ts=T + 200)
    fix2 = engine.ingest_text_event("Code", "ide", "POOL_TIMEOUT = 30", doc_path="src/db/pool.py",
                                    window_id="ide", ts=T + 260)
    assert help_.event_id in {e.src for e in fix2.edges}
