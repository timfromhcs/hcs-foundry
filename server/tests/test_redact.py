from __future__ import annotations

from foundry.data.redact import redact_record, redact_text
from foundry.data.schema import CanonicalRecord, TraceStep


def test_api_keys_are_redacted():
    text = "keys: sk-ant-abcdefghijklmnopqrstuvwx and ghp_abcdefghijklmnopqrstuvwxyz0123456789"
    result = redact_text(text)
    assert "sk-ant-" not in result.text
    assert "ghp_" not in result.text
    assert result.contains_sensitive


def test_credentials_and_emails():
    result = redact_text("password=hunter2 mail me at dev@example.com")
    assert "hunter2" not in result.text
    assert "dev@example.com" not in result.text
    types = {item["type"] for item in result.findings}
    assert "credential_assignment" in types
    assert "email" in types


def test_private_key_block():
    text = "-----BEGIN RSA PRIVATE KEY-----\nMIIEow\n-----END RSA PRIVATE KEY-----"
    result = redact_text(text)
    assert "MIIEow" not in result.text
    assert "[REDACTED:PRIVATE_KEY]" in result.text


def test_home_paths_are_anonymised():
    result = redact_text(r"C:\Users\alice\project and /home/bob/app")
    assert "alice" not in result.text
    assert "bob" not in result.text


def test_high_entropy_is_flagged_not_redacted():
    token = "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8S9t0"
    result = redact_text(f"value {token}")
    types = {item["type"] for item in result.findings}
    assert "high_entropy" in types
    assert token in result.text


def test_vercel_gateway_key_is_redacted():
    result = redact_text("key vck_test000000000000000000000000000000000000000000")
    assert "vck_" not in result.text
    assert "[REDACTED:VERCEL_KEY]" in result.text


def test_redact_record_never_mutates_original():
    record = CanonicalRecord(
        record_id="rec_1",
        source={"provider": "generic", "adapter_version": "test"},
        task={"user_request": "my key is sk-ant-abcdefghijklmnopqrstuvwx"},
        trace=[TraceStep(kind="user_message", content="sk-ant-abcdefghijklmnopqrstuvwx")],
    )
    original = record.to_json()
    redacted = redact_record(record)
    assert record.to_json() == original
    assert "sk-ant-" not in redacted.task["user_request"]
    assert redacted.privacy["contains_sensitive_data"] is True
    assert redacted.privacy["redactions"]
