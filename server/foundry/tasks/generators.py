"""Verifiable task generators — unlimited data with answers no model can argue with.

Every item is produced from a template, a seed and an index, so the same seed always yields the
same dataset. The verifier is the answer: a candidate answer is accepted only if the family's
check passes, which is what makes a measured improvement meaningful.

Leakage is prevented by construction: a split is assigned per **template**, never per row, so a
test template can never appear in training. ``readiness()`` refuses a split that is too small to
support a claim, and ``decontaminate()`` refuses items whose prompt already exists in an eval set.

No network, no model: pure Python plus the standard library.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable

SPLITS = ("train", "dev", "test")
_JSON_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class TaskItem:
    task_id: str
    family: str
    template_id: str
    split: str
    prompt: str
    answer: str
    meta: dict = field(default_factory=dict)

    def to_public(self) -> dict:
        return {
            "task_id": self.task_id,
            "family": self.family,
            "template_id": self.template_id,
            "split": self.split,
            "prompt": self.prompt,
            "answer": self.answer,
            "meta": self.meta,
        }


@dataclass(frozen=True)
class Family:
    name: str
    templates: tuple[str, ...]
    make: Callable[[str, random.Random], tuple[str, str, dict]]
    verify: Callable[[str, str], bool]


# -- helpers ----------------------------------------------------------------


def _stable_id(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


_SPLIT_MAP: dict[str, str] | None = None


def split_map() -> dict[str, str]:
    """Assign every known template to exactly one split, ranked by a stable hash.

    Ranking (not ``hash % 10``) guarantees that each split receives templates whenever there are
    enough of them; a modulo bucket can silently produce an empty test split, which would make a
    held-out claim impossible.
    """
    global _SPLIT_MAP
    if _SPLIT_MAP is None:
        templates = sorted(
            {template for family in FAMILIES.values() for template in family.templates},
            key=lambda template: hashlib.sha256(template.encode("utf-8")).hexdigest(),
        )
        count = len(templates)
        train_end = max(1, round(count * 0.6))
        dev_end = max(train_end + 1, round(count * 0.8))
        mapping: dict[str, str] = {}
        for index, template in enumerate(templates):
            if index < train_end:
                mapping[template] = "train"
            elif index < dev_end:
                mapping[template] = "dev"
            else:
                mapping[template] = "test"
        _SPLIT_MAP = mapping
    return _SPLIT_MAP


def split_for_template(template_id: str) -> str:
    """Deterministic 60/20/20 split by template — a template never straddles two splits."""
    return split_map()[template_id]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def parse_json_answer(answer: str):
    """Accept a bare object or one wrapped in a markdown fence; refuse anything else."""
    cleaned = _JSON_FENCE.sub("", answer.strip())
    return json.loads(cleaned)


def _close(value: float, expected: float, tolerance: float = 1e-9) -> bool:
    return abs(float(value) - float(expected)) <= tolerance


# -- family: arithmetic -----------------------------------------------------

_NAMES = ("Ada", "Grace", "Alan", "Edsger", "Barbara", "Ken", "Margaret", "Donald", "Radia", "Tim")
_ITEMS = ("cable", "filter", "bearing", "sensor", "adapter", "gasket", "module", "relay")


def _make_arith(template: str, rng: random.Random) -> tuple[str, str, dict]:
    if template == "arith_add":
        a, b = rng.randint(11, 999), rng.randint(11, 999)
        return (
            f"Compute {a} + {b}. Answer with the number only.",
            str(a + b),
            {"expected": a + b, "op": "add"},
        )
    if template == "arith_mul":
        a, b = rng.randint(3, 99), rng.randint(3, 99)
        return (
            f"Compute {a} * {b}. Answer with the number only.",
            str(a * b),
            {"expected": a * b, "op": "mul"},
        )
    if template == "arith_percent":
        whole = rng.randint(2, 40) * 50
        pct = rng.choice((5, 10, 12, 15, 20, 25, 40))
        value = whole * pct // 100
        return (
            f"What is {pct}% of {whole}? Answer with the number only.",
            str(value),
            {"expected": value, "op": "percent"},
        )
    raise KeyError(template)


def _verify_arith(expected_answer: str, answer: str) -> bool:
    cleaned = answer.strip().replace(",", "").replace("_", "")
    try:
        return _close(float(cleaned), float(expected_answer))
    except (TypeError, ValueError):
        return False


# -- family: format conversion ---------------------------------------------


def _make_fmt(template: str, rng: random.Random) -> tuple[str, str, dict]:
    if template == "fmt_csv_to_json":
        headers = ["name", "qty", "price"]
        rows = [
            [_NAMES[rng.randrange(len(_NAMES))], str(rng.randint(1, 9)), f"{rng.randint(100, 9999) / 100:.2f}"]
            for _ in range(3)
        ]
        csv_text = "name,qty,price\n" + "\n".join(",".join(row) for row in rows)
        payload = {
            "rows": [
                {"name": row[0], "qty": int(row[1]), "price": float(row[2])}
                for row in rows
            ]
        }
        return (
            "Convert this CSV into JSON with a single key \"rows\" whose values are objects with "
            f"typed fields (qty integer, price number).\n\n{csv_text}",
            json.dumps(payload, sort_keys=True),
            {"template": "csv_to_json"},
        )
    if template == "fmt_json_to_csv":
        rows = [
            {"id": rng.randint(1, 99), "status": rng.choice(("open", "closed", "blocked"))}
            for _ in range(3)
        ]
        prompt = (
            "Convert this JSON array into CSV with the header \"id,status\" and one line per row, "
            "in the given order. Return only the CSV.\n\n" + json.dumps(rows)
        )
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(["id", "status"])
        for row in rows:
            writer.writerow([row["id"], row["status"]])
        return prompt, buffer.getvalue().strip(), {"template": "json_to_csv"}
    if template == "fmt_sorted_list":
        names = rng.sample(_NAMES, k=5)
        return (
            "Sort these names alphabetically and return them as a comma-separated list, "
            "lowercase, one space after each comma.\n\n" + ", ".join(names),
            ", ".join(sorted(name.lower() for name in names)),
            {"template": "sorted_list"},
        )
    raise KeyError(template)


def _verify_fmt(expected_answer: str, answer: str) -> bool:
    cleaned = _JSON_FENCE.sub("", answer.strip())
    try:
        if cleaned.startswith("{"):
            return json.loads(cleaned) == json.loads(expected_answer)
        if "," in expected_answer or expected_answer.count("\n") <= 1:
            return [part.strip() for part in cleaned.split(",")] == [
                part.strip() for part in expected_answer.split(",")
            ]
    except json.JSONDecodeError:
        return False
    return "\n".join(line.strip() for line in cleaned.splitlines() if line.strip()) == "\n".join(
        line.strip() for line in expected_answer.splitlines() if line.strip()
    )


# -- family: field extraction ----------------------------------------------


def _make_extract(template: str, rng: random.Random) -> tuple[str, str, dict]:
    if template == "extract_invoice":
        invoice = f"INV-{rng.randint(1000, 9999)}"
        total = rng.randint(10, 999)
        currency = rng.choice(("EUR", "USD", "CHF"))
        text = (
            f"Invoice {invoice} was issued on 2026-03-{rng.randint(1, 28):02d}. "
            f"The amount due is {total},00 {currency}, payable within 14 days."
        )
        payload = {"invoice_id": invoice, "total": total, "currency": currency}
        return (
            "Extract the invoice id, the total as an integer and the currency. Return only JSON "
            "with the keys invoice_id, total, currency.\n\n" + text,
            json.dumps(payload, sort_keys=True),
            {"template": "invoice"},
        )
    if template == "extract_log":
        level = rng.choice(("INFO", "WARN", "ERROR"))
        service = rng.choice(("api", "worker", "scheduler", "gateway"))
        request_id = f"req-{rng.randint(10000, 99999)}"
        text = f"2026-03-0{rng.randint(1, 9)}T12:00:00Z {level} service={service} request_id={request_id} handled"
        payload = {"level": level, "service": service, "request_id": request_id}
        return (
            "Extract the log level, the service and the request id. Return only JSON with the keys "
            "level, service, request_id.\n\n" + text,
            json.dumps(payload, sort_keys=True),
            {"template": "log"},
        )
    raise KeyError(template)


def _verify_extract(expected_answer: str, answer: str) -> bool:
    try:
        return parse_json_answer(answer) == json.loads(expected_answer)
    except (json.JSONDecodeError, TypeError):
        return False


# -- family: tool calls -----------------------------------------------------


def _make_toolcall(template: str, rng: random.Random) -> tuple[str, str, dict]:
    if template == "tool_weather":
        city = rng.choice(("Berlin", "Paris", "Zurich", "Vienna", "Lisbon"))
        unit = rng.choice(("celsius", "fahrenheit"))
        payload = {"tool": "weather.get", "arguments": {"city": city, "unit": unit}}
        return (
            "Emit one JSON tool call for the weather tool. Required shape: "
            "{\"tool\": \"weather.get\", \"arguments\": {\"city\": string, \"unit\": "
            "\"celsius\"|\"fahrenheit\"}}.\n\n"
            f"What is the weather in {city} in {unit}?",
            json.dumps(payload, sort_keys=True),
            {"template": "weather"},
        )
    if template == "tool_calendar":
        title = rng.choice(("standup", "review", "demo", "retro", "planning"))
        start = f"2026-04-{rng.randint(1, 28):02d}T09:00:00Z"
        payload = {"tool": "calendar.create", "arguments": {"title": title, "start": start}}
        return (
            "Emit one JSON tool call for the calendar tool. Required shape: "
            "{\"tool\": \"calendar.create\", \"arguments\": {\"title\": string, \"start\": ISO-8601}}.\n\n"
            f"Schedule a {title} at {start}.",
            json.dumps(payload, sort_keys=True),
            {"template": "calendar"},
        )
    raise KeyError(template)


def _verify_toolcall(expected_answer: str, answer: str) -> bool:
    try:
        parsed = parse_json_answer(answer)
    except (json.JSONDecodeError, TypeError):
        return False
    if not isinstance(parsed, dict):
        return False
    if set(parsed) != {"tool", "arguments"}:
        return False
    expected = json.loads(expected_answer)
    if parsed["tool"] != expected["tool"]:
        return False
    if not isinstance(parsed["arguments"], dict) or set(parsed["arguments"]) != set(expected["arguments"]):
        return False
    return all(parsed["arguments"][key] == value for key, value in expected["arguments"].items())


FAMILIES: dict[str, Family] = {
    "arith": Family(
        name="arith",
        templates=("arith_add", "arith_mul", "arith_percent"),
        make=_make_arith,
        verify=_verify_arith,
    ),
    "fmt": Family(
        name="fmt",
        templates=("fmt_csv_to_json", "fmt_json_to_csv", "fmt_sorted_list"),
        make=_make_fmt,
        verify=_verify_fmt,
    ),
    "extract": Family(
        name="extract",
        templates=("extract_invoice", "extract_log"),
        make=_make_extract,
        verify=_verify_extract,
    ),
    "toolcall": Family(
        name="toolcall",
        templates=("tool_weather", "tool_calendar"),
        make=_make_toolcall,
        verify=_verify_toolcall,
    ),
}


# -- generation -------------------------------------------------------------


def generate(family: str, *, seed: int = 0, per_template: int = 10) -> list[TaskItem]:
    """Deterministic items for one family, distributed evenly over its templates."""
    spec = FAMILIES[family]
    if per_template < 1:
        raise ValueError("per_template must be >= 1")
    items: list[TaskItem] = []
    for template in spec.templates:
        for index in range(per_template):
            rng = random.Random(f"{seed}:{template}:{index}")
            prompt, answer, meta = spec.make(template, rng)
            items.append(
                TaskItem(
                    task_id=_stable_id(family, template, str(seed), str(index)),
                    family=family,
                    template_id=template,
                    split=split_for_template(template),
                    prompt=prompt,
                    answer=answer,
                    meta=meta,
                )
            )
    return items


def generate_all(*, seed: int = 0, per_template: int = 10) -> list[TaskItem]:
    items: list[TaskItem] = []
    for family in FAMILIES:
        items.extend(generate(family, seed=seed, per_template=per_template))
    return items


def verify(item: TaskItem, answer: str) -> bool:
    """The family verifier is the ground truth; there is no model judge here."""
    return FAMILIES[item.family].verify(item.answer, answer)


def by_split(items: Iterable[TaskItem], split: str) -> list[TaskItem]:
    return [item for item in items if item.split == split]


def template_split_map(items: Iterable[TaskItem]) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = {}
    for item in items:
        mapping.setdefault(item.template_id, set()).add(item.split)
    return mapping


def check_template_isolation(items: Iterable[TaskItem]) -> list[str]:
    """Templates that ended up in more than one split — a leak, by definition."""
    return sorted(template for template, splits in template_split_map(items).items() if len(splits) > 1)


def decontaminate(items: Iterable[TaskItem], forbidden_prompts: Iterable[str]) -> tuple[list[TaskItem], int]:
    """Drop items whose normalized prompt already exists in the eval set; return (kept, dropped)."""
    forbidden = {normalize(prompt) for prompt in forbidden_prompts}
    kept: list[TaskItem] = []
    dropped = 0
    for item in items:
        if normalize(item.prompt) in forbidden:
            dropped += 1
            continue
        kept.append(item)
    return kept, dropped


@dataclass
class Readiness:
    ok: bool
    counts: dict[str, int]
    blockers: list[str] = field(default_factory=list)

    def to_public(self) -> dict:
        return {"ok": self.ok, "counts": dict(self.counts), "blockers": list(self.blockers)}


def readiness(
    items: Iterable[TaskItem],
    *,
    min_train: int = 60,
    min_dev: int = 20,
    min_test: int = 20,
) -> Readiness:
    """Refuse a dataset that cannot support a paired claim, before any training happens."""
    items = list(items)
    counts = {split: len(by_split(items, split)) for split in SPLITS}
    blockers: list[str] = []
    for split, minimum in (("train", min_train), ("dev", min_dev), ("test", min_test)):
        if counts[split] < minimum:
            blockers.append(f"{split} has {counts[split]} item(s), needs >= {minimum}")
    leaks = check_template_isolation(items)
    if leaks:
        blockers.append(f"templates appear in more than one split: {', '.join(leaks)}")
    return Readiness(ok=not blockers, counts=counts, blockers=blockers)


def to_jsonl(path, items: Iterable[TaskItem]) -> int:
    """Write items as JSONL; returns the number of rows written."""
    from pathlib import Path

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for item in items:
            handle.write(json.dumps(item.to_public(), ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count
