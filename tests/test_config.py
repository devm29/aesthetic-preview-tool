"""Tests for environment-driven settings."""

from __future__ import annotations

import pytest

from src.config import DISCLAIMER, PROJECT_ROOT, QCThresholds, Settings, get_settings


@pytest.fixture(autouse=True)
def clear_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_defaults_are_usable_without_any_environment(monkeypatch):
    for name in list(dict(__import__("os").environ)):
        if name.startswith("DERMA_"):
            monkeypatch.delenv(name, raising=False)
    settings = get_settings()
    assert settings.preview_max_side == 1024
    assert settings.enable_generative is False
    assert settings.detector_preference == "auto"
    assert isinstance(settings.qc, QCThresholds)


@pytest.mark.parametrize(
    "value,expected",
    [("true", True), ("1", True), ("YES", True), ("on", True), ("false", False), ("0", False)],
)
def test_boolean_env_vars_accept_the_usual_spellings(monkeypatch, value, expected):
    monkeypatch.setenv("DERMA_ENABLE_GENERATIVE", value)
    assert get_settings().enable_generative is expected


def test_numeric_env_vars_are_parsed(monkeypatch):
    monkeypatch.setenv("DERMA_PREVIEW_MAX_SIDE", "640")
    monkeypatch.setenv("DERMA_IDENTITY_MIN_SSIM", "0.65")
    settings = get_settings()
    assert settings.preview_max_side == 640
    assert settings.identity_min_ssim == pytest.approx(0.65)


def test_a_malformed_numeric_env_var_falls_back_to_the_default(monkeypatch):
    """A typo in the environment must not stop the app from booting."""
    monkeypatch.setenv("DERMA_PREVIEW_MAX_SIDE", "not-a-number")
    assert get_settings().preview_max_side == 1024


def test_an_empty_env_var_is_treated_as_unset(monkeypatch):
    monkeypatch.setenv("DERMA_FACE_DETECTOR", "   ")
    assert get_settings().detector_preference == "auto"


def test_case_is_normalised_for_enumerated_values(monkeypatch):
    monkeypatch.setenv("DERMA_FACE_DETECTOR", "MediaPipe")
    monkeypatch.setenv("DERMA_LOG_LEVEL", "debug")
    settings = get_settings()
    assert settings.detector_preference == "mediapipe"
    assert settings.log_level == "DEBUG"


def test_settings_are_frozen():
    import dataclasses

    with pytest.raises(dataclasses.FrozenInstanceError):
        Settings().preview_max_side = 1  # type: ignore[misc]


def test_settings_are_cached_per_process(monkeypatch):
    monkeypatch.setenv("DERMA_PREVIEW_MAX_SIDE", "800")
    first = get_settings()
    monkeypatch.setenv("DERMA_PREVIEW_MAX_SIDE", "900")
    assert get_settings() is first


def test_sample_image_path_points_inside_the_repo():
    assert Settings().sample_image_path.parent == PROJECT_ROOT / "assets"


def test_the_disclaimer_says_it_is_not_a_medical_device():
    """Product promise: this string is rendered in the UI, the CLI and exports."""
    lowered = DISCLAIMER.lower()
    assert "proof of concept" in lowered
    assert "not a medical device" in lowered
    assert "not a prediction or guarantee" in lowered
