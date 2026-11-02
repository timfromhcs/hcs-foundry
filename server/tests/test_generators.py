"""Task generators: every item passes its own verifier, and a wrong answer cannot slip through."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from foundry.tasks import generators as gen


def test_every_generated_item_passes_its_own_verifier():
    for family in gen.FAMILIES:
        items = gen.generate(family, seed=7, per_template=8)
        assert items
        for item in items:
            assert gen.verify(item, item.answer), (family, item.template_id, item.answer)


def test_generation_is_deterministic_for_a_seed():
    first = gen.generate_all(seed=11, per_template=3)
    second = gen.generate_all(seed=11, per_template=3)
    assert [item.to_public() for item in first] == [item.to_public() for item in second]
    other = gen.generate("arith", seed=12, per_template=3)
    assert [item.prompt for item in other] != [item.prompt for item in gen.generate("arith", seed=11, per_template=3)]


def test_a_wrong_answer_fails_even_when_it_looks_plausible():
    item = next(item for item in gen.generate("arith", seed=3, per_template=1))
    wrong = str(int(item.answer) + 1)
    assert gen.verify(item, wrong) is False
    assert gen.verify(item, item.answer + " apples") is False  # no silent unit tolerance
    assert gen.verify(item, "I cannot compute that") is False
    assert gen.verify(item, " " + item.answer + "\n") is True


def test_toolcall_rejects_a_missing_or_extra_argument():
    item = next(item for item in gen.generate("toolcall", seed=5, per_template=1))
    payload = json.loads(item.answer)
    missing = json.loads(item.answer)
    missing["arguments"].pop(next(iter(missing["arguments"])))
    extra = json.loads(item.answer)
    extra["arguments"]["surprise"] = True
    assert gen.verify(item, json.dumps(missing)) is False
    assert gen.verify(item, json.dumps(extra)) is False
    assert gen.verify(item, "```json\n" + item.answer + "\n```") is True


def test_extract_rejects_a_field_with_the_wrong_type():
    item = next(item for item in gen.generate("extract", seed=2, per_template=1))
    payload = json.loads(item.answer)
    for key, value in payload.items():
        if isinstance(value, int):
            payload[key] = str(value)
    assert gen.verify(item, json.dumps(payload)) is False
    assert gen.verify(item, item.answer) is True


def test_splits_never_share_a_template():
    items = gen.generate_all(seed=1, per_template=12)
    assert gen.check_template_isolation(items) == []
    assert gen.readiness(items).ok is True
    counts = gen.readiness(items).counts
    assert all(counts[split] > 0 for split in gen.SPLITS), counts
    # assigning the same template to two splits is detected, not tolerated
    broken = [item for item in items]
    broken.append(
        gen.TaskItem(
            task_id="x",
            family=broken[0].family,
            template_id=broken[0].template_id,
            split="test" if broken[0].split != "test" else "train",
            prompt="p",
            answer="a",
        )
    )
    assert gen.check_template_isolation(broken) == [broken[0].template_id]
    assert gen.readiness(broken).ok is False


def test_readiness_refuses_a_split_that_is_too_small():
    items = gen.generate("arith", seed=1, per_template=2)
    report = gen.readiness(items, min_train=100, min_dev=100, min_test=100)
    assert report.ok is False
    assert any("needs >=" in blocker for blocker in report.blockers)


def test_decontamination_drops_exact_prompt_matches():
    items = gen.generate("fmt", seed=9, per_template=3)
    kept, dropped = gen.decontaminate(items, [items[0].prompt.upper() + "  "])
    assert dropped == 1
    assert len(kept) == len(items) - 1
    assert all(gen.normalize(item.prompt) != gen.normalize(items[0].prompt) for item in kept)


def test_jsonl_round_trip_keeps_the_verifier(tmp_path: Path):
    items = gen.generate("arith", seed=4, per_template=2)
    path = tmp_path / "tasks.jsonl"
    assert gen.to_jsonl(path, items) == len(items)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == len(items)
    rebuilt = gen.TaskItem(**rows[0])
    assert gen.verify(rebuilt, rebuilt.answer) is True


def test_a_metric_claim_needs_a_real_verifier_not_a_judge():
    """The verifier is deterministic: the same answer gets the same verdict every time."""
    item = next(item for item in gen.generate("fmt", seed=8, per_template=1))
    verdicts = {gen.verify(item, item.answer) for _ in range(5)}
    assert verdicts == {True}
    with pytest.raises(json.JSONDecodeError):
        gen.parse_json_answer("{not json")
