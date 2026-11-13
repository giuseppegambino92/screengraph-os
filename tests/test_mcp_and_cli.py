"""End-to-end: all MCP tools over JSON-RPC (direct + stdio) and every CLI command."""

from __future__ import annotations

import io
import json

import pytest
from typer.testing import CliRunner

from screengraph.config import Config
from screengraph.engine import ScreenGraph
from screengraph.interfaces.cli import app
from screengraph.interfaces.mcp_server import TOOLS, MCPServer


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    d = tmp_path_factory.mktemp("mcp")
    runner = CliRunner()
    res = runner.invoke(app, ["--home", str(d), "simulate", "--no-now"])
    assert res.exit_code == 0, res.output
    srv = MCPServer(lambda: ScreenGraph(Config.ephemeral(d)))
    yield srv
    srv.engine.close()


def rpc(srv, method, params=None, mid=1):
    msg = {"jsonrpc": "2.0", "id": mid, "method": method}
    if params is not None:
        msg["params"] = params
    return srv.handle(msg)


def call(srv, name, args):
    r = rpc(srv, "tools/call", {"name": name, "arguments": args})["result"]
    assert r["isError"] is False, r["content"][0]["text"]
    assert json.loads(r["content"][0]["text"]) == json.loads(json.dumps(r["structuredContent"], default=str))
    return r["structuredContent"]


def test_initialize_and_list(server):
    r = rpc(server, "initialize", {"protocolVersion": "2025-03-26", "capabilities": {}})["result"]
    assert r["protocolVersion"] == "2025-03-26" and r["serverInfo"]["name"] == "screengraph-os"
    assert server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    tools = {t["name"]: t for t in rpc(server, "tools/list")["result"]["tools"]}
    for name in ("screengraph_active_workspace", "screengraph_causal_why", "screengraph_visual_search",
                 "screengraph_timeline_replay", "screengraph_ingest_event", "screengraph_privacy_audit"):
        assert name in tools and tools[name]["inputSchema"]["type"] == "object"
    assert rpc(server, "ping")["result"] == {}


def test_tool_active_workspace(server):
    ws = call(server, "screengraph_active_workspace", {})
    assert len(ws["windows"]) == 5 and ws["screenshots_used"] == 0
    assert set(ws["monitors"]) == {0, 1}
    assert {w["app_category"] for w in ws["windows"]} == {"terminal", "browser", "ide", "dashboard", "chat"}


def test_tool_causal_why(server):
    r = call(server, "screengraph_causal_why", {"target": "src/scheduler.rs"})
    assert r["found"] and r["root_causes"][0]["app_category"] == "terminal"
    cats = [c["app_category"] for c in r["chain"]]
    assert "chat" in cats and "browser" in cats and cats[-1] == "ide"


def test_tool_visual_search(server):
    r = call(server, "screengraph_visual_search", {"query": "orange warning modal", "k": 2})
    top = r["results"][0]
    assert top["app"] == "Google Chrome" and "orange" in top["patch"]["visual_concepts"]
    assert {"row", "col", "quadrant", "screen_bbox"} <= set(top["patch"])


def test_tool_timeline_replay(server):
    r = call(server, "screengraph_timeline_replay", {"last_minutes": 15, "causal_only": True})
    assert r["events"] > 0 and r["mermaid"].startswith("graph LR")
    full = call(server, "screengraph_timeline_replay", {})
    assert full["events"] >= r["events"]


def test_tool_ingest_event_links_into_graph(server):
    r = call(server, "screengraph_ingest_event", {
        "app": "zsh", "app_category": "terminal", "window_id": "hook-term", "ts": 1_731_510_000.0,
        "text": "make test\nFAILED tests/test_ledger.py::test_rounding - LedgerRoundingError: off by one cent\n"
                "export STRIPE_KEY=" + "sk" + "_live_" + "abcdefghijklmnop1234"})
    assert r["stored"] and r["redactions"] >= 1
    r2 = call(server, "screengraph_ingest_event", {
        "app": "Code", "app_category": "ide", "window_id": "hook-ide", "ts": 1_731_510_060.0,
        "doc_path": "src/ledger/rounding.py", "text": "# fix LedgerRoundingError: use Decimal quantize"})
    assert any(e["from"] == r["event_id"] for e in r2["edges"])
    w = call(server, "screengraph_causal_why", {"target": "src/ledger/rounding.py"})
    assert w["root_causes"][0]["event_id"] == r["event_id"]


def test_tool_privacy_audit_and_search(server):
    a = call(server, "screengraph_privacy_audit", {})
    assert a["zero_pixel_guarantee_holds"] and a["disk"]["image_magic_hits"] == 0
    assert a["redactions"]["total"] > 0 and a["ram"]["all_wipes_verified_zero"]
    s = call(server, "screengraph_search", {"query": "TokenExpiredError jwt expired", "k": 3})
    assert s["results"] and "jwt" in (s["results"][0]["snippet"] + s["results"][0]["window_title"]).lower()


def test_jsonrpc_errors(server):
    assert rpc(server, "does/not/exist")["error"]["code"] == -32601
    assert server.handle({"id": 1, "method": "x"})["error"]["code"] == -32600
    assert json.loads(server.handle_raw("{not json"))["error"]["code"] == -32700
    r = rpc(server, "tools/call", {"name": "screengraph_visual_search", "arguments": {}})["result"]
    assert r["isError"] is True
    assert rpc(server, "tools/call", {"name": "nope", "arguments": {}})["error"]["code"] == -32602
    batch = json.loads(server.handle_raw(json.dumps([{"jsonrpc": "2.0", "id": 7, "method": "ping"},
                                                     {"jsonrpc": "2.0", "method": "notifications/x"}])))
    assert batch == [{"jsonrpc": "2.0", "id": 7, "result": {}}]
    assert set(TOOLS) >= {"screengraph_ingest_event"}


def test_stdio_transport(server):
    lines = [json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}),
             json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
             json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                         "params": {"name": "screengraph_causal_why", "arguments": {"target": "db_pool.py"}}})]
    out = io.StringIO()
    srv = MCPServer(lambda: server.engine)
    srv._engine = server.engine
    srv.serve_stdio = srv.serve_stdio  # noqa
    for line in lines:
        resp = srv.handle_raw(line)
        if resp:
            out.write(resp + "\n")
    msgs = [json.loads(l) for l in out.getvalue().splitlines()]
    assert [m["id"] for m in msgs] == [1, 2]
    assert msgs[1]["result"]["structuredContent"]["root_causes"]


# ------------------------------------------------------------------ CLI
@pytest.fixture(scope="module")
def cli_home(tmp_path_factory):
    d = tmp_path_factory.mktemp("cli")
    r = CliRunner().invoke(app, ["--home", str(d), "simulate"])
    assert r.exit_code == 0, r.output
    assert "dropped by the idle gate" in r.output
    return str(d)


@pytest.mark.parametrize("args,expect", [
    (["why", "db_pool.py"], "root cause"),
    (["why", "k8s/payments-deployment.yaml", "--json"], "kubectl"),
    (["visual", "--like", "5", "--json"], "maxsim"),
    (["active-workspace"], "screenshots_used"),
    (["privacy-audit"], "zero_pixel_guarantee_holds"),
    (["visual", "red memory spike chart in top-right quadrant"], "top-right"),
    (["search", "OOMKilled payments-api", "-k", "3"], "hits in"),
    (["workspace"], "screenshots_used"),
    (["timeline", "--minutes", "10"], "["),
    (["timeline", "--mermaid", "--causal-only"], "graph LR"),
    (["audit"], "zero_pixel_guarantee_holds"),
    (["status"], "causal chains"),
    (["version"], "screengraph-os"),
])
def test_cli_commands(cli_home, args, expect):
    r = CliRunner().invoke(app, ["--home", cli_home, *args], env={"COLUMNS": "140"})
    assert r.exit_code == 0, r.output
    assert expect in r.output


def test_cli_ingest_from_stdin(cli_home):
    r = CliRunner().invoke(app, ["--home", cli_home, "ingest", "--app", "zsh", "--category", "terminal"],
                           input="curl -H 'Authorization: Bearer abcdefghijklmnopqrstuvwxyz012345' localhost:8080\n")
    assert r.exit_code == 0, r.output
    out = json.loads(r.output)
    assert out["stored"] and out["redactions"] >= 1


def test_cli_eval_runs_benchmark(tmp_path):
    out = tmp_path / "report.json"
    r = CliRunner().invoke(app, ["eval", "--scale", "0", "--sweep", "0", "--out", str(out)], env={"COLUMNS": "140"})
    assert r.exit_code == 0, r.output
    rep = json.loads(out.read_text())
    assert rep["privacy"]["redaction_recall"] == 1.0
    assert rep["gate_and_storage"]["zero_pixel_audit"]["image_files_on_disk"] == 0
    assert rep["gate_and_storage"]["projected_30d_8h_mb"] < 95
    assert rep["causal"]["summary"]["screengraph_chain_precision"] == 1.0
    v = rep["visual"]["summary"]  # int8 SQ must not change recall or localization vs float32
    assert v["patch_maxsim_recall_at_1"] == v["float32_reference_recall_at_1"]
    assert v["patch_quadrant_localization_acc"] == v["float32_reference_localization_acc"]
    assert rep["ablation_ui_concepts"]["causal"]["screengraph_root_recall_at_1"] == 1.0
    # default encoder (morphological concept bank, no simulator-tuned detectors)
    assert v["patch_maxsim_recall_at_1"] == 1.0 and v["patch_quadrant_localization_acc"] == 1.0
    assert rep["causal"]["summary"]["screengraph_root_recall_at_1"] == 1.0



@pytest.mark.skipif(__import__("os").environ.get("SCREENGRAPH_LIVE_TEST") != "1" or
                    __import__("platform").system() != "Darwin", reason="set SCREENGRAPH_LIVE_TEST=1 on macOS")
def test_live_capture_on_this_mac(tmp_path):
    from screengraph.capture.live import run_live_capture

    res = run_live_capture(2, interval=0.5, data_dir=tmp_path)
    assert res["status"] == "ok" and res["frames_seen"] > 0
    assert res["zero_pixel_guarantee_holds"] and res["ingest_under_10ms"]
    assert "title" not in json.dumps(res).lower().replace("title-free", "").replace("window titles", "")
