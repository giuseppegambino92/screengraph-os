"""PII zero-trust scrubber: format rules + Luhn + Shannon-entropy secret detection.

Every redaction is replaced by a deterministic, keyed tag::

    <REDACTED_SECRET:3fa9c01b2e>   (credentials, keys, tokens, passwords)
    <REDACTED_CARD:...>  <REDACTED_EMAIL:...>  <REDACTED_IP:...>

The suffix is the first 10 hex chars of HMAC-SHA256(install_key, secret).
The same secret seen in two windows yields the same tag, so the causal engine
can still link "token copied from terminal -> pasted into chat" without the
secret ever being stored. The key lives outside the database, so tags cannot
be brute-forced back into low-entropy values (card numbers, emails) from a
stolen DB.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import re
from collections import Counter
from dataclasses import dataclass, field

TAG_RE = re.compile(r"<REDACTED_[A-Z]+:[0-9a-f]{10}>")


def shannon_entropy(s: str) -> float:
    """H(X) = -sum p(x) log2 p(x) over the character distribution of s (bits/char)."""
    if not s:
        return 0.0
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in Counter(s).values())


def luhn_valid(digits: str) -> bool:
    if not digits.isdigit() or not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = ord(ch) - 48
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def char_classes(s: str) -> int:
    return (
        any(c.islower() for c in s)
        + any(c.isupper() for c in s)
        + any(c.isdigit() for c in s)
        + any(not c.isalnum() for c in s)
    )


# (kind, category, regex, group) -- group 0 = whole match
_FORMAT_RULES: list[tuple[str, str, re.Pattern, int]] = [
    ("pem_private_key", "SECRET",
     re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(?:-----END [A-Z ]*PRIVATE KEY-----|\Z)"), 0),
    ("jwt", "SECRET", re.compile(r"\beyJ[A-Za-z0-9_-]{5,}\.eyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{8,}"), 0),
    ("bearer_token", "SECRET", re.compile(r"(?i)\bbearer\s+([A-Za-z0-9._~+/=-]{16,})"), 1),
    ("openai_key", "SECRET", re.compile(r"\bsk-(?:proj-|ant-(?:api\d\d-)?|svcacct-)?[A-Za-z0-9_-]{20,}"), 0),
    ("github_token", "SECRET", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{40,})"), 0),
    ("aws_access_key", "SECRET", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), 0),
    ("aws_secret_key", "SECRET",
     re.compile(r"(?i)aws_secret_access_key\s*[=:]\s*['\"]?([A-Za-z0-9/+=]{40})"), 1),
    ("stripe_key", "SECRET", re.compile(r"\b(?:sk|rk|pk)_(?:live|test)_[A-Za-z0-9]{16,}"), 0),
    ("slack_token", "SECRET", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"), 0),
    ("google_api_key", "SECRET", re.compile(r"\bAIza[0-9A-Za-z_-]{35}"), 0),
    ("url_credentials", "SECRET", re.compile(r"://[^:/\s@]+:([^@/\s]{3,})@"), 1),
    ("password_assignment", "SECRET",
     re.compile(r"(?i)\b[A-Z0-9_]*(?:PASSWORD|PASSWD|PWD|SECRET|TOKEN|API_?KEY|PRIVATE_KEY|ACCESS_KEY)[A-Z0-9_]*"
                r"\s*[=:]\s*(['\"]?)([^\s'\"]{4,})\1"), 2),
    ("cli_secret_flag", "SECRET",
     re.compile(r"(?i)--(?:password|passwd|token|api-key|apikey|secret|auth-token)(?:=|\s+)(['\"]?)([^\s'\"]{4,})\1"), 2),
    ("credit_card", "CARD", re.compile(r"(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d-])"), 0),
    ("email", "EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), 0),
    ("ipv4", "IP", re.compile(r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\d.])"), 0),
    ("ipv6", "IP", re.compile(r"(?<![0-9A-Fa-f:])(?:(?:[0-9A-Fa-f]{1,4}:){7}[0-9A-Fa-f]{1,4}|"
                              r"(?:[0-9A-Fa-f]{1,4}:){1,6}:(?:[0-9A-Fa-f]{1,4}:?){1,6})(?![0-9A-Fa-f:])"), 0),
]

# Sensitive contexts where Shannon entropy alone decides.
_ENV_ASSIGN = re.compile(r"(?m)^\s*(?:export\s+)?[A-Z][A-Z0-9_]{2,}=(['\"]?)([^\s'\"]{12,})\1")
_AUTH_HEADER = re.compile(r"(?im)\b(?:authorization|x-api-key|api-key|cookie|set-cookie|x-auth-token)\s*:\s*"
                          r"(?:[A-Za-z]+\s+)?([^\s;,'\"]{12,})")
_CLI_FLAG = re.compile(r"(?:^|\s)--?[A-Za-z][\w-]*(?:=|\s+)(['\"]?)([^\s'\"]{12,})\1")
_NL_CRED = re.compile(r"(?i)\b(?:password|passwd|passphrase|passcode|pw|pwd|creds|credentials|secret|token)\b"
                      r"[^\n:=]{0,28}[:=]\s*['\"]?([^\s'\"<>]{8,})")
_GENERIC_TOKEN = re.compile(r"[^\s'\"<>()\[\]{},;`=:]{20,}")
_CODE_REF = re.compile(r"^(?:\$|process\.env|os\.environ|getenv|env\(|<|\{)|^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+[(\[]?")
_PATHLIKE_RX = re.compile(r"(?:://|^[~./]|/[\w.-]+/|\.(?:py|ts|js|rs|go|ya?ml|json|md|html|txt|sh|toml)\b)")


def _pathlike(val: str) -> bool:
    """Looks like a path/URL *and* reads like one: base64 secrets also contain '/', but real
    paths are low-entropy (repeated letters, separators), so H < 4.5 bits/char is required."""
    return bool(_PATHLIKE_RX.search(val)) and shannon_entropy(val) < 4.5


@dataclass
class Redaction:
    kind: str
    category: str
    tag: str
    start: int
    end: int


@dataclass
class ScrubResult:
    text: str
    redactions: list[Redaction] = field(default_factory=list)


class Scrubber:
    def __init__(
        self,
        hmac_key: bytes,
        entropy_threshold: float = 4.2,
        redact_emails: bool = True,
        redact_ips: bool = True,
    ):
        self._key = hmac_key
        self.entropy_threshold = entropy_threshold
        self.redact_emails = redact_emails
        self.redact_ips = redact_ips
        self.stats: Counter[str] = Counter()

    # ------------------------------------------------------------------
    def tag_for(self, category: str, secret: str) -> str:
        digest = hmac.new(self._key, secret.strip().encode("utf-8", "ignore"), hashlib.sha256).hexdigest()
        return f"<REDACTED_{category}:{digest[:10]}>"

    def hash_text(self, text: str) -> str:
        return hmac.new(self._key, text.encode("utf-8", "ignore"), hashlib.sha256).hexdigest()[:16]

    # ------------------------------------------------------------------
    def _candidates(self, text: str) -> list[tuple[int, int, str, str]]:
        spans: list[tuple[int, int, str, str]] = []
        for kind, cat, rx, grp in _FORMAT_RULES:
            if cat == "EMAIL" and not self.redact_emails:
                continue
            if cat == "IP" and not self.redact_ips:
                continue
            for m in rx.finditer(text):
                s, e = m.span(grp)
                if s < 0 or e <= s:
                    continue
                val = text[s:e]
                if val.startswith("<REDACTED"):
                    continue
                if kind == "credit_card" and not luhn_valid(re.sub(r"[ -]", "", val)):
                    continue
                if kind in ("password_assignment", "cli_secret_flag") and _CODE_REF.match(val):
                    continue
                if kind == "email" and val.lower().startswith("git@"):
                    continue
                if kind == "ipv4" and (val.startswith("127.") or val == "0.0.0.0"):
                    continue
                if kind == "ipv6" and (val in ("::1",) or "::" not in val and val.count(":") < 7):
                    continue
                spans.append((s, e, kind, cat))

        # credential headers: the value *is* the secret, whatever its entropy
        for m in _AUTH_HEADER.finditer(text):
            s, e = m.span(1)
            if not text[s:e].startswith("<REDACTED"):
                spans.append((s, e, "auth_header", "SECRET"))
        # Shannon-entropy detector in sensitive contexts (env files, CLI flags)
        for rx, grp in ((_ENV_ASSIGN, 2), (_CLI_FLAG, 2)):
            for m in rx.finditer(text):
                s, e = m.span(grp)
                val = text[s:e]
                if val.startswith("<REDACTED") or _pathlike(val):
                    continue
                if shannon_entropy(val) > self.entropy_threshold:
                    spans.append((s, e, "high_entropy_context", "SECRET"))

        # natural-language credential context: "staging pw: X", "creds rotated, new one: X"
        for m in _NL_CRED.finditer(text):
            s, e = m.span(1)
            val = text[s:e]
            if not val.startswith("<REDACTED") and char_classes(val) >= 3 and not _pathlike(val):
                spans.append((s, e, "credential_context", "SECRET"))
        # Context-free: long multi-class tokens above the entropy threshold
        for m in _GENERIC_TOKEN.finditer(text):
            val = m.group(0)
            core = val.rstrip(".:!?")
            if len(core) >= 20:  # strip sentence punctuation, but never below the length gate
                val = core
            if "<REDACTED" in val or _pathlike(val) or len(val) < 19:
                continue
            if char_classes(val) >= 3 and shannon_entropy(val) > self.entropy_threshold:
                spans.append((m.start(), m.start() + len(val), "high_entropy_token", "SECRET"))
        return spans

    def scrub(self, text: str) -> ScrubResult:
        if not text:
            return ScrubResult(text or "")
        protected = [m.span() for m in TAG_RE.finditer(text)]
        spans = self._candidates(text)
        # format rules (known key shapes, Luhn, PEM...) take priority over entropy heuristics
        formal = [sp for sp in spans if not sp[2].startswith("high_entropy")]
        spans = formal + [sp for sp in spans if sp[2].startswith("high_entropy")
                          and not any(f[0] < sp[1] and sp[0] < f[1] for f in formal)]
        # overlap resolution: credentials beat PII (a token inside "https://TOKEN@github.com" must be
        # tagged SECRET, not EMAIL), then longer spans win; accepted spans never overlap
        prio = {"SECRET": 0, "CARD": 1, "EMAIL": 2, "IP": 3}
        spans.sort(key=lambda t: (prio.get(t[3], 9), -(t[1] - t[0]), t[0]))
        chosen: list[tuple[int, int, str, str]] = []
        for s, e, kind, cat in spans:
            if any(cs < e and s < ce for cs, ce, _, _ in chosen):
                continue
            if any(ps < e and s < pe for ps, pe in protected):
                continue
            chosen.append((s, e, kind, cat))
        chosen.sort()
        if not chosen:
            return ScrubResult(text)
        out, reds, pos = [], [], 0
        for s, e, kind, cat in chosen:
            secret = text[s:e]
            tag = self.tag_for(cat, secret)
            out.append(text[pos:s])
            out.append(tag + "\n" * secret.count("\n"))  # keep line structure for per-node remapping
            reds.append(Redaction(kind, cat, tag, s, e))
            self.stats[kind] += 1
            pos = e
        out.append(text[pos:])
        return ScrubResult("".join(out), reds)

    def scrub_lines(self, lines: list[str]) -> tuple[list[str], list[Redaction]]:
        """Scrub a list of AX text nodes jointly (multi-line secrets such as PEM blocks span nodes)."""
        if not lines:
            return [], []
        joined = "\n".join(l.replace("\n", " ") for l in lines)
        res = self.scrub(joined)
        out = res.text.split("\n")
        if len(out) != len(lines):  # defensive: never fall back to raw text
            out = [self.scrub(l).text for l in lines]
        return out, res.redactions
