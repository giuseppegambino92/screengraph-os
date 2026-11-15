"""English/Italian message catalog and the CLI's --lang / SCREENGRAPH_LANG switch."""

from __future__ import annotations

import inspect
import json
import re

import pytest
from typer.main import get_command_name
from typer.models import ArgumentInfo, OptionInfo
from typer.testing import CliRunner

from screengraph import i18n
from screengraph.interfaces import cli
from screengraph.interfaces.cli import app

MARKUP = re.compile(r"\[/?[a-z ]*\]")


@pytest.fixture(autouse=True)
def _reset_language(monkeypatch):
    monkeypatch.delenv(i18n.ENV_VAR, raising=False)
    i18n.set_lang(None)
    yield
    i18n.set_lang(None)
    cli.localize_help()


@pytest.fixture(scope="module")
def home(tmp_path_factory):
    d = tmp_path_factory.mktemp("i18n")
    r = CliRunner().invoke(app, ["--home", str(d), "simulate", "--no-now"], env={"SCREENGRAPH_LANG": None})
    assert r.exit_code == 0, r.output
    return str(d)


def run(*args: str, env: dict | None = None):
    r = CliRunner().invoke(app, list(args), env={"COLUMNS": "140", "SCREENGRAPH_LANG": None, **(env or {})})
    assert r.exit_code == 0, r.output
    return r.output


# ------------------------------------------------------------------ catalog
def test_every_key_exists_in_every_language():
    keys = {lang: set(msgs) for lang, msgs in i18n.CATALOG.items()}
    assert set(keys) == set(i18n.SUPPORTED)
    for lang in i18n.SUPPORTED:
        assert keys[lang] == keys[i18n.DEFAULT_LANG], f"{lang}: {keys[lang] ^ keys[i18n.DEFAULT_LANG]}"


@pytest.mark.parametrize("key", sorted(i18n.CATALOG["en"]))
def test_same_placeholders_and_markup(key):
    en = i18n.CATALOG["en"][key]
    for lang in i18n.SUPPORTED:
        msg = i18n.CATALOG[lang][key]
        assert msg.strip(), f"{lang}:{key} is empty"
        assert i18n.placeholders(msg) == i18n.placeholders(en), f"{lang}:{key}"
        assert MARKUP.findall(msg) == MARKUP.findall(en), f"{lang}:{key}"


def test_lookup_resolution(monkeypatch):
    assert i18n.get_lang() == "en"
    assert i18n.t("why.chain") == "chain" and i18n.t("why.chain", lang="it") == "catena"
    assert i18n.t("search.title", n=3, ms=1.234) == "3 hits in 1.23 ms"
    monkeypatch.setenv(i18n.ENV_VAR, "it_IT.UTF-8")
    assert i18n.get_lang() == "it" and i18n.t("search.title", n=3, ms=1.0) == "3 risultati in 1.00 ms"
    assert i18n.set_lang("en") == "en"  # an explicit choice beats the environment
    monkeypatch.setenv(i18n.ENV_VAR, "klingon")
    i18n.set_lang(None)
    assert i18n.get_lang() == "en"  # unknown env value -> English
    with pytest.raises(ValueError):
        i18n.set_lang("de")
    with pytest.raises(KeyError):
        i18n.t("no.such.key")


def test_every_help_text_comes_from_the_catalog():
    """Each command/option help is a catalog key, so --lang it can translate all of it."""
    cmds = [("app", app.registered_callback.callback)]
    cmds += [(ci.name or get_command_name(ci.callback.__name__), ci.callback) for ci in app.registered_commands]
    for name, cb in cmds:
        assert i18n.has(f"help.{name}"), name
        for pname, p in inspect.signature(cb).parameters.items():
            if isinstance(p.default, (OptionInfo, ArgumentInfo)) and p.default.help and pname != "category":
                assert p.default.help == i18n.t(f"help.{name}.{pname}", lang="en"), (name, pname)


def test_argv_lang():
    assert cli._argv_lang(["--home", "/x", "--lang", "it", "why", "a"]) == "it"
    assert cli._argv_lang(["--lang=it", "status"]) == "it"
    assert cli._argv_lang(["why", "--lang", "it"]) is None  # sub-command arguments are not global options
    assert cli._argv_lang([]) is None


# ------------------------------------------------------------------ CLI output
def test_why_default_is_english(home):
    out = run("--home", home, "why", "db_pool.py")
    assert "why db_pool.py?" in out and "root cause" in out and "chain" in out
    assert "perché" not in out and "catena" not in out


def test_why_in_italian(home):
    out = run("--home", home, "--lang", "it", "why", "db_pool.py")
    assert "perché db_pool.py?" in out and "catena" in out
    assert "SQLiteBusyException" in out and "errors:sqlitebusyexception" in out  # data/evidence untouched


def test_env_var_and_flag_precedence(home):
    assert "risultati in" in run(
        "--home", home, "search", "OOMKilled payments-api", "-k", "3", env={"SCREENGRAPH_LANG": "it"}
    )
    out = run("--home", home, "--lang", "en", "search", "OOMKilled", "-k", "3", env={"SCREENGRAPH_LANG": "it"})
    assert "hits in" in out and "risultati" not in out


def test_visual_and_status_in_italian(home):
    out = run("--home", home, "--lang", "it", "visual", "red memory spike chart in top-right quadrant")
    assert "ricerca visiva:" in out and "quadrante" in out and "top-right" in out
    out = run("--home", home, "--lang", "it", "status")
    assert "catene causali" in out and "finestre attive" in out and "ROOT CAUSE" in out


def test_json_is_never_translated(home):
    en = json.loads(run("--home", home, "why", "db_pool.py", "--json"))
    it = json.loads(run("--home", home, "--lang", "it", "why", "db_pool.py", "--json"))
    en.pop("elapsed_ms"), it.pop("elapsed_ms")
    assert en == it
    assert json.loads(run("--home", home, "--lang", "it", "privacy-audit")) == json.loads(
        run("--home", home, "privacy-audit")
    )


def test_invalid_language_is_rejected():
    r = CliRunner().invoke(app, ["--lang", "de", "version"])
    assert r.exit_code != 0


def test_help_is_localized():
    i18n.set_lang("it")
    cli.localize_help()
    out = run("why", "--help")
    assert "risale il grafo causale" in out
    i18n.set_lang(None)
    cli.localize_help()
    out = run("why", "--help")
    assert "trace the causal graph" in out
