"""Secret and PII detection.

This is risk reduction, not a guarantee. The UI must always require a human review
before data is approved for training.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from .schema import CanonicalRecord

# Ordered: earlier patterns win, so specific token formats are checked before the
# generic key=value rule.
PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"), "[REDACTED:PRIVATE_KEY]"),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b"), "[REDACTED:API_KEY]"),
    ("openai_key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{20,}\b"), "[REDACTED:API_KEY]"),
    ("github_pat", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"), "[REDACTED:GITHUB_TOKEN]"),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"), "[REDACTED:GITHUB_TOKEN]"),
    ("aws_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), "[REDACTED:AWS_KEY]"),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b"), "[REDACTED:SLACK_TOKEN]"),
    ("google_key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"), "[REDACTED:GOOGLE_KEY]"),
    ("hf_token", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"), "[REDACTED:HF_TOKEN]"),
    ("vercel_key", re.compile(r"\bvck_[A-Za-z0-9]{20,}\b"), "[REDACTED:VERCEL_KEY]"),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b"), "[REDACTED:JWT]"),
    ("bearer", re.compile(r"(?i)\bbearer\s+([A-Za-z0-9._\-]{20,})"), "Bearer [REDACTED:TOKEN]"),
    ("connection_string", re.compile(r"(?i)\b(?:postgres|mysql|mongodb(?:\+srv)?|redis|amqp)://[^\s\"']+"), "[REDACTED:CONNECTION_STRING]"),
    (
        "credential_assignment",
        re.compile(
            r"(?i)\b(?:password|passwd|pwd|secret|api[_\-]?key|access[_\-]?key|auth[_\-]?token)\b"
            r"\s*[:=]\s*[\"']?([^\s\"',;]{6,})"
        ),
        "[REDACTED:CREDENTIAL]",
    ),
    ("email", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), "[REDACTED:EMAIL]"),
    ("phone", re.compile(r"(?<!\d)(?:\+\d{1,3}[\s\-.]?)?(?:\(\d{2,4}\)[\s\-.]?)?\d{3,4}[\s\-.]\d{3,4}(?:[\s\-.]\d{2,4})?(?!\d)"), "[REDACTED:PHONE]"),
    ("credit_card", re.compile(r"(?<!\d)(?:\d[ \-]?){13,19}(?!\d)"), "[REDACTED:CARD]"),
    ("windows_user_path", re.compile(r"(?i)\b([A-Z]:\\Users\\)([^\\\s\"']+)"), r"\1[USER]"),
    ("unix_home_path", re.compile(r"(?i)(/home/|/Users/)([^/\s\"']+)"), r"\1[USER]"),
]

# High-entropy strings (base64/hex blobs) are suspicious but can also be legitimate
# hashes in code. Only flag them, never auto-redact.
ENTROPY_MIN_LEN = 40
ENTROPY_THRESHOLD = 4.2
ENTROPY_TOKEN = re.compile(r"[A-Za-z0-9+/=_\-]{40,}")


@dataclass
class RedactionResult:
    text: str
    findings: list[dict] = field(default_factory=list)

    @property
    def contains_sensitive(self) -> bool:
        return any(item["count"] > 0 for item in self.findings)


def _shannon(text: str) -> float:
    if not text:
        return 0.0
    counts: dict[str, int] = {}
    for char in text:
        counts[char] = counts.get(char, 0) + 1
    total = len(text)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def redact_text(text: str) -> RedactionResult:
    if not text:
        return RedactionResult(text=text, findings=[])
    findings: dict[str, dict] = {}
    output = text
    for name, pattern, replacement in PATTERNS:
        matches = list(pattern.finditer(output))
        if not matches:
            continue
        output = pattern.sub(replacement, output)
        findings[name] = {
            "type": name,
            "count": len(matches),
            "severity": "high" if name not in {"email", "phone", "windows_user_path", "unix_home_path"} else "medium",
        }

    suspicious = 0
    for match in ENTROPY_TOKEN.finditer(output):
        token = match.group(0)
        if _shannon(token) >= ENTROPY_THRESHOLD:
            suspicious += 1
    if suspicious:
        findings["high_entropy"] = {
            "type": "high_entropy",
            "count": suspicious,
            "severity": "review",
            "note": "Possible secret or hash. Flagged, not redacted.",
        }
    return RedactionResult(text=output, findings=list(findings.values()))


def redact_record(record: CanonicalRecord) -> CanonicalRecord:
    """Return a redacted copy. The original is never mutated."""
    data = record.to_dict()
    totals: dict[str, int] = {}

    def scrub(value: str) -> str:
        result = redact_text(value)
        for finding in result.findings:
            totals[finding["type"]] = totals.get(finding["type"], 0) + finding["count"]
        return result.text

    data["task"]["user_request"] = scrub(data["task"].get("user_request") or "")
    if data["task"].get("context_summary"):
        data["task"]["context_summary"] = scrub(data["task"]["context_summary"])
    for step in data["trace"]:
        step["content"] = scrub(step.get("content") or "")
        step["content_hash"] = None
    for key in ("decision_summary", "method_summary", "negative_example"):
        if data["teaching"].get(key):
            data["teaching"][key] = scrub(data["teaching"][key])

    redacted = CanonicalRecord.from_dict(data)
    for step in redacted.trace:
        from .schema import content_hash

        step.content_hash = content_hash(step.content)

    redacted.privacy["redactions"] = [
        {"type": key, "count": value} for key, value in sorted(totals.items())
    ]
    redacted.privacy["contains_sensitive_data"] = bool(totals)
    return redacted
