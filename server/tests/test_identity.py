from __future__ import annotations

import pytest

from foundry.identity import (
    approve_profile,
    build_dataset_records,
    create_dataset,
    create_profile,
    draft_identity,
    evaluate_identity,
    get_profile,
    update_card,
)

PROMPT = (
    "You are Mara, a calm senior engineer. I am a solo developer building local AI tools. "
    "Always answer in short paragraphs. Never invent facts or personal history. You value honesty and precision."
)


def test_draft_separates_facts_from_proposals():
    card = draft_identity(PROMPT)
    assert card["name"] == "Mara"
    assert any("solo developer" in fact for fact in card["explicit_facts"])
    assert card["generated_proposals"]
    assert card["boundaries"]
    assert card["provenance"]["source"] == "user_prompt"
    assert card["approved"] is False


def test_short_prompt_rejected():
    with pytest.raises(ValueError):
        draft_identity("hi")


def test_dataset_requires_approval(db):
    profile = create_profile(db, PROMPT, "Mara", ["en"])
    with pytest.raises(ValueError):
        create_dataset(db, profile["id"])
    approve_profile(db, profile["id"])
    result = create_dataset(db, profile["id"])
    assert result["record_count"] >= 7
    assert result["manifest"]["synthetic"] is True


def test_dataset_rows_are_labelled_synthetic():
    card = draft_identity(PROMPT)
    records = build_dataset_records(card)
    assert all(record.is_synthetic for record in records)
    assert all(record.teaching["annotation_origin"] == "synthetic" for record in records)
    assert all(record.privacy["approved_for_training"] is False for record in records)
    assert {record.metadata["identity_case"] for record in records} >= {"privacy_boundary", "unknown_case"}


def test_approved_card_is_immutable(db):
    profile = create_profile(db, PROMPT, "Mara", ["en"])
    approve_profile(db, profile["id"])
    with pytest.raises(ValueError):
        update_card(db, profile["id"], {"purpose": "changed"})


def test_evaluation_scores_deterministically():
    card = draft_identity(PROMPT)

    def complete(messages):
        prompt = messages[-1]["content"].lower()
        if "who are you" in prompt:
            return "I am Mara."
        if "yesterday" in prompt:
            return "I do not know."
        if "api key" in prompt:
            return "I will not repeat credentials."
        return "READY"

    result = evaluate_identity("profile", card, complete)
    assert result["score"] == 1.0
    assert result["total"] == 4


def test_profile_roundtrip(db):
    profile = create_profile(db, PROMPT, "Mara", ["en", "de"])
    loaded = get_profile(db, profile["id"])
    assert loaded["card"]["languages"] == ["en", "de"]
