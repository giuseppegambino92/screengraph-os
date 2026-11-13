"""Hawkes-process cross-app causal graph engine.

For a new node V_j at time t_j, every node V_i inside the causal horizon
[t_j - H, t_j) is scored:

    S(i->j) = w_time   * exp(-lambda (t_j - t_i))          (Hawkes exponential kernel)
            + w_sem    * max(0, cos(E_i, E_j))              (global embedding similarity)
            + w_bridge * Bridge(i, j)                       (deterministic entity evidence)
            + w_trans  * P(app_j | app_i)                   (online Dirichlet transition prior)

An edge is kept when S >= tau AND it carries evidence (Bridge > 0, or cosine
>= sem_only_min): elapsed time alone never creates causation. Each kept edge
also stores its Hawkes *branching responsibility*

    p(i->j) = S(i->j) / (mu + sum_k S(k->j))

i.e. the posterior probability that V_i is V_j's parent in the cluster
(branching) representation of a self-exciting process, with mu the background
"immigrant" intensity.

Bridges (max over types):
  clipboard  1.00  text copied in V_i appears in V_j (search box, chat, editor)
  doc_path   1.00  the file V_j edits is mentioned in V_i (stack trace, chat)
  link       0.90  V_j navigated to a URL that was visible in V_i
  secret     0.90  same <REDACTED_SECRET:hash> tag in both (secret moved between apps)
  path       0.85  shared file paths
  error      0.85  shared exception class / error code
  route/host 0.75  shared localhost:port, API route or domain
  resource   0.60  shared kebab-case resource (pod, service, repo)
  symbol     0.45 + 0.15 (n-1), cap 0.80   shared identifiers (n = overlap)

Mention-only discount: from a chat source without error content, path/symbol/
resource/route bridges are multiplied by ``mention_discount`` (0.35).

Within the *same* window, entity bridges are computed only from lines that are
new in V_j (a chat reply, a selection, a fresh log line): the previous state of
a window trivially shares all its persistent text with the next one, which is
continuity, not causation.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Iterable, Optional
from urllib.parse import parse_qs, unquote_plus, urlparse

import numpy as np

from screengraph.config import Config

# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------
_EXT = r"(?:py|pyi|ts|tsx|js|jsx|mjs|rs|go|ya?ml|json|toml|md|java|kt|c|cc|cpp|h|hpp|sh|sql|rb|php|swift|cs|tf|ini|cfg|env)"
_PATH_RE = re.compile(rf"(?<![\w/.-])((?:[\w.-]+/)*[\w-]+(?:\.[\w-]+)*\.{_EXT})(?::\d+)?\b")
_ERROR_RE = re.compile(r"\b([A-Z][A-Za-z0-9]*(?:Exception|Error|Killed|Fault|Panic))\b|\b(E\d{4})\b|\berror\[(E\d{4})\]")
_URL_RE = re.compile(r"\bhttps?://[^\s<>\"'`)\]]+")
_HOST_RE = re.compile(r"\b(localhost:\d{2,5}|127\.0\.0\.1:\d{2,5}|0\.0\.0\.0:\d{2,5})\b")
_ROUTE_RE = re.compile(r"(?<![\w.])(/(?:api|v\d|graphql|auth|internal)(?:/[\w.{}:-]+)+)")
_SNAKE_RE = re.compile(r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b")
_CAMEL_RE = re.compile(r"\b([a-z]+(?:[A-Z][a-z0-9]+)+|[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+)\b")
_CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]{2,})\(")
_KEBAB_RE = re.compile(r"\b([a-z][a-z0-9]+(?:-[a-z0-9]+){1,4})\b")
_TAG_RE = re.compile(r"<REDACTED_[A-Z]+:[0-9a-f]{10}>")
_ERRFLAG_RE = re.compile(
    r"Traceback|Exception\b|Error\b|error\[|\bFAILED\b|\bfailed\b|OOMKilled|\bpanic\b|\bfatal\b|"
    r"\b(?:401|403|500|502|503)\b|CrashLoopBackOff|ALERT|\bfailing\b",
)
MENTION_KINDS = frozenset({"doc_path", "paths", "symbols", "resources", "routes", "hosts"})
ROLE_SEP = "\x1f"  # content lines are "role<US>text" so a new selection of old text still counts as new
CHROME_ROLES = frozenset({"AXRow", "AXOutlineRow", "AXTab", "AXTabGroup", "AXMenuItem", "AXMenuBarItem",
                          "AXToolbar", "AXWindowTitle", "AXButton", "AXTextField"})
_NORM_RE = re.compile(r"[^a-z0-9<>_:/.\-]+")

SYMBOL_STOP = frozenset(
    "self this none true false null return import from def class function const let var async await "
    "print len str int dict list main init test tests setup index value values data result results "
    "node_modules site_packages __init__ __main__ console log".split()
)
DOMAIN_STOP = frozenset({"google.com", "www.google.com", "duckduckgo.com", "bing.com"})


def norm_text(s: str) -> str:
    return " ".join(_NORM_RE.sub(" ", unquote_plus(s or "").lower()).split())


def norm_url(u: str) -> str:
    try:
        p = urlparse(u.rstrip(".,;"))
    except ValueError:
        return u.lower()
    host = (p.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    path = p.path.rstrip("/")
    return f"{host}{path}"


_PROMPT_RE = re.compile(r"(?m)^[\w.-]+@[\w.-]+[: ]\S*\s*[%$#>]\s?")  # "dev@mbp ~/src/acme-api % " shell chrome


def extract_entities(text: str, url: str = "", doc_path: str = "", title: str = "") -> dict[str, list[str]]:
    full = _PROMPT_RE.sub("", f"{title}\n{text}")
    paths = set()
    for m in _PATH_RE.finditer(full):
        p = m.group(1).lower()
        if "://" in p or p.count(".") > 4:
            continue
        paths.add(p.rsplit("/", 1)[-1])
    if doc_path:
        paths.add(doc_path.lower().rsplit("/", 1)[-1])
    errors = set()
    for m in _ERROR_RE.finditer(full):
        e = m.group(1) or m.group(2) or m.group(3)
        if e and e not in ("Error", "Exception"):
            errors.add(e.lower())
    urls, hosts = set(), set()
    raw_urls = _URL_RE.findall(full) + ([url] if url else [])
    search_text = ""
    for u in raw_urls:
        nu = norm_url(u)
        if nu:
            urls.add(nu)
        try:
            p = urlparse(u)
            host = (p.netloc or "").lower().removeprefix("www.")
            if host and host not in DOMAIN_STOP:
                hosts.add(host)
            q = parse_qs(p.query).get("q")
            if q:
                search_text += " " + q[0]
        except ValueError:
            pass
    for m in _HOST_RE.finditer(full):
        hosts.add(m.group(1).replace("127.0.0.1", "localhost").replace("0.0.0.0", "localhost"))
    routes = {m.group(1).lower().rstrip("/") for m in _ROUTE_RE.finditer(full + " " + url)}
    symbols = set()
    for rx in (_SNAKE_RE, _CAMEL_RE, _CALL_RE):
        for m in rx.finditer(full):
            s = m.group(1).lower()
            if len(s) >= 4 and s not in SYMBOL_STOP and not s.endswith(tuple("." + e for e in ("py", "ts", "rs"))):
                symbols.add(s)
    symbols -= {p.rsplit(".", 1)[0] for p in paths}  # don't double count db_pool.py as symbol db_pool
    symbols -= errors
    resources = set()
    for m in _KEBAB_RE.finditer(full):
        r = m.group(1)
        if len(r) >= 6:
            resources.add(r)
            parts = r.split("-")
            if len(parts) > 2:
                resources.add("-".join(parts[:2]))  # payments-api-7d9f8b-x2k -> payments-api
    resources -= {h.split(":")[0] for h in hosts}
    secrets = set(_TAG_RE.findall(full))
    flags = []
    if _ERRFLAG_RE.search(full) or errors:
        flags.append("error")
    ents = {
        "paths": sorted(paths), "errors": sorted(errors), "urls": sorted(urls), "hosts": sorted(hosts),
        "routes": sorted(routes), "symbols": sorted(symbols)[:200], "resources": sorted(resources)[:100],
        "secrets": sorted(secrets), "flags": flags,
    }
    if search_text:
        ents["search"] = [norm_text(search_text)]
    return ents


# ---------------------------------------------------------------------------
# Node + edge containers
# ---------------------------------------------------------------------------
@dataclass
class CausalNode:
    id: int
    ts: float
    window_id: str
    app_category: str
    global_vec: np.ndarray
    entities: dict[str, set]
    text_norm: str
    url_norm: str = ""
    doc_path: str = ""
    copied_norm: str = ""
    has_error: bool = False
    lines: frozenset = frozenset()
    # destination-side evidence: what is *new* in this state relative to the previous state of
    # the same window (filled by CausalEngine.infer). Persistent scrollback is not new evidence.
    novel_entities: Optional[dict] = None
    novel_text: str = ""
    url_changed: bool = True

    @classmethod
    def build(cls, eid: int, ts: float, window_id: str, app_category: str, global_vec: np.ndarray,
              entities: dict, body: str, title: str = "", url: str = "", doc_path: str = "",
              copied: str = "", content_lines: Optional[list[str]] = None) -> "CausalNode":
        ents = {k: set(v) for k, v in entities.items() if k not in ("flags", "visual")}
        text = norm_text(f"{title} {body} {url} {' '.join(entities.get('search', []))}")
        flags = entities.get("flags", [])
        src_lines = content_lines if content_lines is not None else (body or "").splitlines()
        lines = frozenset(l.strip() for l in src_lines if l.strip())
        return cls(eid, ts, window_id, app_category, np.asarray(global_vec, np.float32), ents, text,
                   norm_url(url) if url else "", (doc_path or "").lower().rsplit("/", 1)[-1],
                   norm_text(copied) if copied else "", "error" in flags or "anomaly" in flags, lines)


@dataclass
class Edge:
    src: int
    dst: int
    score: float
    prob: float
    relation: str
    c_time: float
    c_sem: float
    c_bridge: float
    c_trans: float
    evidence: list[str] = field(default_factory=list)

    @property
    def causal(self) -> bool:
        return self.c_bridge > 0

    def row(self) -> tuple:
        return (self.src, self.dst, self.score, self.prob, self.relation, self.c_time, self.c_sem,
                self.c_bridge, self.c_trans, json.dumps(self.evidence[:12]))

    @classmethod
    def from_row(cls, r: dict) -> "Edge":
        ev = r["evidence"] if isinstance(r["evidence"], list) else json.loads(r["evidence"] or "[]")
        return cls(r["src"], r["dst"], r["score"], r["prob"], r["relation"], r["c_time"], r["c_sem"],
                   r["c_bridge"], r["c_trans"], ev)


CATS = ("terminal", "browser", "ide", "dashboard", "chat", "other")
_BASE_PRIOR = {
    "terminal": {"browser": 0.35, "ide": 0.30, "chat": 0.15, "terminal": 0.10, "dashboard": 0.05, "other": 0.05},
    "browser": {"ide": 0.35, "browser": 0.25, "terminal": 0.20, "chat": 0.10, "dashboard": 0.05, "other": 0.05},
    "ide": {"terminal": 0.40, "browser": 0.25, "ide": 0.15, "chat": 0.10, "dashboard": 0.05, "other": 0.05},
    "dashboard": {"terminal": 0.45, "browser": 0.20, "chat": 0.15, "ide": 0.10, "dashboard": 0.05, "other": 0.05},
    "chat": {"browser": 0.35, "ide": 0.20, "terminal": 0.20, "chat": 0.15, "dashboard": 0.05, "other": 0.05},
    "other": {c: 1 / 6 for c in CATS},
}


class TransitionPrior:
    """P(dst_cat | src_cat) = (n(src,dst) + alpha * prior) / (n(src) + alpha), updated online."""

    def __init__(self, alpha: float = 20.0, counts: Optional[dict] = None):
        self.alpha = alpha
        self.counts: Counter = Counter(counts or {})

    def p(self, src: str, dst: str) -> float:
        src = src if src in _BASE_PRIOR else "other"
        prior = _BASE_PRIOR[src].get(dst, 0.05)
        n_src = sum(v for (s, _), v in self.counts.items() if s == src)
        return (self.counts[(src, dst)] + self.alpha * prior) / (n_src + self.alpha)

    def observe(self, src: str, dst: str) -> None:
        self.counts[(src, dst)] += 1


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class CausalEngine:
    def __init__(self, cfg: Config, transitions: Optional[dict] = None):
        self.cfg = cfg
        self.recent: deque[CausalNode] = deque()
        self.prior = TransitionPrior(counts=transitions)
        self._df: Counter = Counter()
        self._df_hist: deque[set] = deque()
        self._last_by_window: dict[str, CausalNode] = {}
        self._last_cat: Optional[str] = None

    # -- document frequency over recent events: ambient entities (project name in every
    #    prompt, a host shown in every log line) appear in a large share of states and
    #    therefore carry no evidence about *which* event caused which.
    def _df_observe(self, node: CausalNode) -> None:
        keys = set()
        for k in ("symbols", "resources", "hosts", "routes", "paths", "errors"):
            keys |= {f"{k}:{e}" for e in node.entities.get(k, set())}
        self._df.update(keys)
        self._df_hist.append(keys)
        while len(self._df_hist) > self.cfg.entity_df_window:
            self._df.subtract(self._df_hist.popleft())

    def _common(self, kind: str, e: str) -> bool:
        n = len(self._df_hist)
        return self._df[f"{kind}:{e}"] >= max(4, self.cfg.entity_df_max * n)

    def bridge(self, i: CausalNode, j: CausalNode) -> tuple[float, str, list[str]]:
        best, kind, ev = 0.0, "", []

        def offer(score: float, k: str, evidence: list[str]):
            nonlocal best, kind, ev
            if score > best:
                best, kind, ev = score, k, evidence

        if i.copied_norm and len(i.copied_norm) >= 8 and i.copied_norm in j.novel_text:
            offer(1.0, "clipboard", [i.copied_norm[:80]])
        if j.url_changed and j.url_norm and j.url_norm != i.url_norm and j.url_norm in i.entities.get("urls", set()):
            offer(0.9, "link", [j.url_norm])
        if i.window_id == j.window_id:
            # Same window: only content that is *new* in V_j relative to V_i can carry
            # evidence (a chat reply, a selection, a fresh log line); persistent text is continuity.
            novel = j.lines - i.lines
            if not novel:
                return best, kind, ev
            text = "\n".join(l.split(ROLE_SEP, 1)[-1] for l in novel)
            jents = {k: set(v) for k, v in extract_entities(text).items()}
        else:
            jents = j.novel_entities if j.novel_entities is not None else j.entities
            if j.doc_path and j.doc_path in i.entities.get("paths", set()):
                offer(1.0, "doc_path", [j.doc_path])
        for k, strength in (("secrets", 0.9), ("paths", 0.85), ("errors", 0.85), ("routes", 0.75),
                            ("hosts", 0.75), ("urls", 0.75), ("resources", 0.6)):
            ov = {e for e in i.entities.get(k, set()) & jents.get(k, set()) if not self._common(k, e)}
            if ov:
                offer(strength, k, sorted(ov))
        sym = {e for e in i.entities.get("symbols", set()) & jents.get("symbols", set())
               if not self._common("symbols", e)}
        if sym:
            offer(min(0.8, 0.45 + 0.15 * (len(sym) - 1)), "symbols", sorted(sym))
        # Mention-only discount: a conversational message that merely *names* a file/symbol/resource
        # (no error text, no clipboard transfer, no link followed) is weak evidence that it caused
        # the next edit -- "I'll touch middleware.ts later" must not compete with a stack trace.
        if i.app_category == "chat" and not i.has_error and kind in MENTION_KINDS:
            best *= self.cfg.mention_discount
        return best, kind, ev

    @staticmethod
    def classify(i: CausalNode, j: CausalNode, bridge: float, kind: str) -> str:
        ci, cj = i.app_category, j.app_category
        if kind == "clipboard":
            return "CLIPBOARD_TRANSFER"
        if kind == "link":
            return "LINK_FOLLOW"
        if bridge <= 0:
            return "SEMANTIC_CONTINUATION"
        if cj == "ide" and ci in ("browser", "chat"):
            return "RESEARCH_TO_CODE_EDIT"
        if cj == "ide" and ci in ("terminal", "dashboard") and i.has_error:
            return "ERROR_TO_CODE_EDIT"
        if ci in ("terminal", "dashboard") and i.has_error and cj in ("browser", "chat"):
            return "ERROR_TO_RESEARCH"
        if {ci, cj} & {"terminal"} and {ci, cj} & {"dashboard", "browser"}:
            return "UI_TO_LOG_CORRELATION"
        return "SEMANTIC_CONTINUATION"

    def _novelty(self, node: CausalNode) -> None:
        prev = self._last_by_window.get(node.window_id)
        if prev is None:
            node.novel_entities = node.entities
            node.novel_text = node.text_norm
            return
        novel = node.lines - prev.lines
        text = "\n".join(l.split(ROLE_SEP, 1)[-1] for l in novel)
        ents = {k: set(v) for k, v in extract_entities(text).items()}
        if node.url_norm and node.url_norm != prev.url_norm:
            ents.setdefault("urls", set()).add(node.url_norm)
            ents.setdefault("hosts", set()).update(
                {h for h in node.entities.get("hosts", set()) if h not in prev.entities.get("hosts", set())})
            node.url_changed = True
        else:
            node.url_changed = False
        node.novel_entities = ents
        node.novel_text = norm_text(text + " " + (" ".join(node.entities.get("search", [])) if node.url_changed else ""))

    def infer(self, node: CausalNode) -> list[Edge]:
        cfg = self.cfg
        self._novelty(node)
        horizon = node.ts - cfg.causal_horizon_s
        while self.recent and self.recent[0].ts < horizon:
            self.recent.popleft()
        cands = [c for c in self.recent if c.ts <= node.ts and c.id != node.id]
        edges: list[Edge] = []
        prev = self._last_by_window.get(node.window_id)
        reobserved = (prev is not None and prev.ts >= horizon and node.lines == prev.lines
                      and not node.url_changed and not node.copied_norm)
        if cands:
            G = np.stack([c.global_vec for c in cands])
            sims = G @ node.global_vec
            for c, sim in zip(cands, sims):
                dt = max(0.0, node.ts - c.ts)
                kt = math.exp(-cfg.decay_lambda * dt)
                b, kind, ev = self.bridge(c, node)
                if reobserved and c is prev:
                    # identical semantic state seen again (e.g. a pixel-only redraw passed the gate):
                    # it inherits its causes through an explicit persistence link
                    b, kind, ev = 1.0, "same_state", [node.window_id]
                sem = max(0.0, float(sim))
                tr = self.prior.p(c.app_category, node.app_category)
                score = cfg.w_time * kt + cfg.w_sem * sem + cfg.w_bridge * b + cfg.w_trans * tr
                if score < cfg.tau_causal or (b <= 0 and sem < cfg.sem_only_min):
                    continue
                edges.append(Edge(c.id, node.id, round(score, 5), 0.0, self.classify(c, node, b, kind),
                                  round(kt, 5), round(sem, 5), round(b, 5), round(tr, 5),
                                  [f"{kind}:{e}" for e in ev] if kind else []))
            # consecutive states of one source window carry the same evidence: keep its best 2,
            # so a burst of terminal redraws cannot crowd out the browser/chat hop
            edges.sort(key=lambda e: -e.score)
            per_win: dict[str, int] = {}
            win_of = {c.id: c.window_id for c in cands}
            kept = []
            for e in edges:
                w = win_of[e.src]
                if per_win.get(w, 0) < 2:
                    kept.append(e)
                    per_win[w] = per_win.get(w, 0) + 1
            edges = kept[: cfg.max_parents]
            z = cfg.hawkes_mu + sum(e.score for e in edges)
            for e in edges:
                e.prob = round(e.score / z, 5)
        self.add(node)
        return edges

    def add(self, node: CausalNode) -> None:
        if self._last_cat is not None:
            self.prior.observe(self._last_cat, node.app_category)
        self._last_cat = node.app_category
        self._df_observe(node)
        self.recent.append(node)
        self._last_by_window[node.window_id] = node


# ---------------------------------------------------------------------------
# In-memory DAG with bidirectional walks and root-cause tracing
# ---------------------------------------------------------------------------
@dataclass
class TraceResult:
    targets: list[int]
    roots: list[tuple[int, float]]  # (event id, root score) best first
    chain: list[int]  # event ids on causal paths root -> targets (chronological)
    edges: list[Edge]
    paths: list[list[int]] = field(default_factory=list)


class CausalGraph:
    def __init__(self):
        self.parents: dict[int, dict[int, Edge]] = {}
        self.children: dict[int, dict[int, Edge]] = {}
        self.ts: dict[int, float] = {}
        self.error: dict[int, bool] = {}

    def add_node(self, eid: int, ts: float, has_error: bool = False) -> None:
        self.ts[eid] = ts
        self.error[eid] = has_error

    def add_edges(self, edges: Iterable[Edge]) -> None:
        for e in edges:
            self.parents.setdefault(e.dst, {})[e.src] = e
            self.children.setdefault(e.src, {})[e.dst] = e

    def causal_parents(self, n: int, min_score: float = 0.0) -> list[Edge]:
        return [e for e in self.parents.get(n, {}).values() if e.causal and e.score >= min_score]

    def neighbors(self, n: int) -> list[tuple[int, Edge]]:
        out = [(s, e) for s, e in self.parents.get(n, {}).items()]
        out += [(d, e) for d, e in self.children.get(n, {}).items()]
        return out

    def trace_back(self, targets: list[int], max_depth: int = 6, min_score: float = 0.0,
                   error_boost: float = 0.3) -> TraceResult:
        """Backward walk over evidence-bearing edges; rank chain sources as root causes.

        best[d][n] = max over length-d backward paths ending at n of sum(log score).
        A source's root score is the geometric mean edge score of its best path
        (length-normalized, so long true chains are not penalized) times an
        error prior (1 + boost) when the source shows an error/alert.
        """
        targets = [t for t in targets if t in self.ts]
        best: dict[int, tuple[float, int, Optional[int]]] = {}  # node -> (mean log, depth, next-hop toward target)
        frontier = {t: 0.0 for t in targets}
        sumlog: dict[tuple[int, int], float] = {}
        ancestors: set[int] = set()
        used: list[Edge] = []
        for depth in range(1, max_depth + 1):
            nxt: dict[int, float] = {}
            for n, sl in frontier.items():
                for e in self.causal_parents(n, min_score):
                    if e.src in targets:
                        continue
                    s = sl + math.log(max(e.score, 1e-9))
                    if s > nxt.get(e.src, -1e18):
                        nxt[e.src] = s
                        sumlog[(e.src, depth)] = s
                    used.append(e)
                    ancestors.add(e.src)
                    mean = s / depth
                    if mean > best.get(e.src, (-1e18, 0, None))[0]:
                        best[e.src] = (mean, depth, n)
            if not nxt:
                break
            frontier = nxt
        if not ancestors:
            return TraceResult(targets, [], [], [])
        # Root-cause candidates: the *earliest failure signals* in the causal ancestry, i.e.
        # error/anomaly nodes with no error/anomaly node among their own causal ancestors.
        # (Upstream non-error context -- the page you were on before the server crashed --
        # is reported in the chain/timeline but is not the root cause.) Fallback: sources.
        err_anc = [a for a in ancestors if self.error.get(a)]
        def has_err_ancestor(a: int) -> bool:
            seen, stack = {a}, [a]
            while stack:
                for e in self.causal_parents(stack.pop(), min_score):
                    if e.src in seen or e.src not in ancestors:
                        continue
                    if self.error.get(e.src):
                        return True
                    seen.add(e.src)
                    stack.append(e.src)
            return False
        first_errors = [a for a in err_anc if not has_err_ancestor(a)]
        sources = [a for a in ancestors if not self.causal_parents(a, min_score)]
        pool = first_errors or sources or list(ancestors)
        scored = []
        for a in pool:
            gm = math.exp(best[a][0])
            boost = 1.0 + (error_boost if self.error.get(a) else 0.0)
            scored.append((a, round(gm * boost, 5)))
        scored.sort(key=lambda x: (-x[1], self.ts.get(x[0], 0)))
        root = scored[0][0]
        # chain = ancestors of targets that are descendants of root (within the walked subgraph)
        sub_children: dict[int, set[int]] = {}
        for e in used:
            sub_children.setdefault(e.src, set()).add(e.dst)
        desc, stack = {root}, [root]
        while stack:
            for c in sub_children.get(stack.pop(), ()):
                if c not in desc:
                    desc.add(c)
                    stack.append(c)
        chain = sorted((desc & (ancestors | set(targets))) | {root}, key=lambda n: self.ts.get(n, 0))
        chain_set = set(chain)
        chain_edges = [e for e in {(e.src, e.dst): e for e in used}.values()
                       if e.src in chain_set and e.dst in chain_set]
        # best single path root -> target
        path, cur = [root], root
        while cur in best and best[cur][2] is not None:
            cur = best[cur][2]
            path.append(cur)
            if cur in targets:
                break
        return TraceResult(targets, scored, chain, chain_edges, [path])

    def walk_forward(self, start: int, max_depth: int = 4) -> list[tuple[int, float, int]]:
        """Effects reachable from ``start``: (node, path product of edge probs, depth)."""
        out, frontier, seen = [], {start: 1.0}, {start}
        for d in range(1, max_depth + 1):
            nxt: dict[int, float] = {}
            for n, w in frontier.items():
                for c, e in self.children.get(n, {}).items():
                    if c in seen or not e.causal:
                        continue
                    nxt[c] = max(nxt.get(c, 0.0), w * e.score)
            for c, w in nxt.items():
                seen.add(c)
                out.append((c, w, d))
            frontier = nxt
            if not frontier:
                break
        return out
