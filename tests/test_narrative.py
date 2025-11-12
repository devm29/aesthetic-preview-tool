"""Tests for the Claude-backed consultation notes.

No test in this file makes a network call. The Anthropic SDK is replaced by a
fake module, so the suite passes with or without ``anthropic`` installed and
never spends money.
"""

from __future__ import annotations

import sys
import types

import pytest

from src import narrative
from src.config import DISCLAIMER, Settings

SUMMARY = {
    "disclaimer": DISCLAIMER,
    "detector": "haar",
    "landmarks": False,
    "quality_checks": [{"name": "Lighting", "status": "pass", "value": 130.0, "detail": "ok"}],
    "treatments": [
        {
            "key": "wrinkle",
            "label": "Wrinkle smoothing",
            "intensity": 40,
            "engine": "parametric",
            "elapsed_ms": 12.0,
        }
    ],
    "identity": {
        "ssim": 0.94,
        "threshold": 0.8,
        "within_threshold": True,
        "verdict": "Visible change, structure retained",
    },
    "elapsed_ms": 42.0,
}
EMPTY_SUMMARY = {**SUMMARY, "treatments": [], "identity": None}


class FakeResponse:
    def __init__(self, text="", stop_reason="end_turn"):
        self.stop_reason = stop_reason
        self.content = [types.SimpleNamespace(type="text", text=text)] if text else []


class FakeMessages:
    def __init__(self, response=None, error=None):
        self._response = response
        self._error = error
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self._error is not None:
            raise self._error
        return self._response


def install_fake_anthropic(monkeypatch, response=None, error=None) -> FakeMessages:
    """Put a fake ``anthropic`` module on sys.modules and set a key."""
    messages = FakeMessages(response, error)

    class FakeClient:
        def __init__(self, *_, **__):
            self.beta = types.SimpleNamespace(messages=messages)

    module = types.ModuleType("anthropic")
    module.Anthropic = FakeClient
    monkeypatch.setitem(sys.modules, "anthropic", module)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    return messages


@pytest.fixture
def no_credentials(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)


# ------------------------------------------------------------------ the gate


def test_narrator_is_unavailable_without_credentials(no_credentials):
    assert narrative.narrator_available(Settings(narrator_enabled=True)) is False


def test_narrator_is_unavailable_when_switched_off(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    assert narrative.narrator_available(Settings(narrator_enabled=False)) is False


def test_narrator_is_available_with_a_key_and_the_sdk(monkeypatch):
    install_fake_anthropic(monkeypatch, FakeResponse("hello"))
    assert narrative.narrator_available(Settings(narrator_enabled=True)) is True


def test_narrator_is_unavailable_without_the_sdk(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    monkeypatch.setitem(sys.modules, "anthropic", None)
    assert narrative.narrator_available(Settings(narrator_enabled=True)) is False


# --------------------------------------------------------------- the template


def test_template_is_used_when_no_key_is_set(no_credentials):
    result = narrative.summarise(SUMMARY, Settings(narrator_enabled=True))
    assert result.source == "template"
    assert result.generated is False
    assert "wrinkle smoothing at 40%" in result.text
    assert "0.940" in result.text


def test_template_says_nothing_was_applied_for_an_untouched_preview(no_credentials):
    text = narrative.summarise(EMPTY_SUMMARY, Settings(narrator_enabled=True)).text
    assert "No treatment was applied" in text
    assert "no measure of" in text


def test_template_refers_the_reader_to_a_practitioner(no_credentials):
    """Honesty requirement: the notes must never read as a treatment plan."""
    text = narrative.summarise(SUMMARY, Settings(narrator_enabled=True)).text.lower()
    assert "qualified practitioner" in text
    assert "not a plan" in text
    assert "image edits" in text


def test_template_explains_that_ssim_is_not_an_identity_match(no_credentials):
    text = narrative.summarise(SUMMARY, Settings(narrator_enabled=True)).text.lower()
    assert "not an identity" in text


# ------------------------------------------------------------- the model path


def test_a_generated_summary_is_returned_with_the_disclaimer_appended(monkeypatch):
    install_fake_anthropic(monkeypatch, FakeResponse("Generated notes."))
    result = narrative.summarise(SUMMARY, Settings(narrator_enabled=True))
    assert result.source == "claude"
    assert result.generated is True
    assert result.text.startswith("Generated notes.")
    assert result.text.endswith(DISCLAIMER)


def test_the_request_pins_the_model_and_a_safety_system_prompt(monkeypatch):
    messages = install_fake_anthropic(monkeypatch, FakeResponse("Notes."))
    settings = Settings(narrator_enabled=True, narrator_model="claude-opus-5")
    narrative.summarise(SUMMARY, settings)

    assert messages.kwargs["model"] == "claude-opus-5"
    assert messages.kwargs["max_tokens"] == narrative.MAX_TOKENS
    system = messages.kwargs["system"]
    assert "not a medical device" in system
    assert "Never state or imply a clinical outcome" in system
    # The summary travels as data, and the identity caveat travels with it.
    user_text = messages.kwargs["messages"][0]["content"]
    assert "wrinkle" in user_text
    assert "not a biometric match score" in user_text


def test_a_refusal_falls_back_to_the_template(monkeypatch):
    install_fake_anthropic(monkeypatch, FakeResponse("", stop_reason="refusal"))
    result = narrative.summarise(SUMMARY, Settings(narrator_enabled=True))
    assert result.source == "template"
    assert "declined" in result.detail


def test_a_truncated_response_falls_back_to_the_template(monkeypatch):
    install_fake_anthropic(monkeypatch, FakeResponse("half a th", stop_reason="max_tokens"))
    result = narrative.summarise(SUMMARY, Settings(narrator_enabled=True))
    assert result.source == "template"
    assert "truncated" in result.detail


def test_an_empty_response_falls_back_to_the_template(monkeypatch):
    install_fake_anthropic(monkeypatch, FakeResponse(""))
    result = narrative.summarise(SUMMARY, Settings(narrator_enabled=True))
    assert result.source == "template"
    assert "Empty" in result.detail


def test_an_api_failure_falls_back_to_the_template_without_raising(monkeypatch, caplog):
    install_fake_anthropic(monkeypatch, error=RuntimeError("network is down"))
    with caplog.at_level("WARNING"):
        result = narrative.summarise(SUMMARY, Settings(narrator_enabled=True))
    assert result.source == "template"
    assert "network is down" in result.detail
    assert "network is down" in caplog.text


def test_non_text_blocks_are_ignored_when_reading_the_response(monkeypatch):
    response = FakeResponse("Visible notes.")
    response.content.insert(0, types.SimpleNamespace(type="thinking", thinking="hidden"))
    install_fake_anthropic(monkeypatch, response)
    result = narrative.summarise(SUMMARY, Settings(narrator_enabled=True))
    assert "hidden" not in result.text
    assert "Visible notes." in result.text
