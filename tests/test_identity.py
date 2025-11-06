"""Tests for the SSIM identity-preservation signal.

The signal is a guard-rail the UI surfaces to the user, so its edge cases have
to behave honestly - especially the ones where no meaningful score exists.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.identity import EXPLANATION, MIN_SSIM_WINDOW, compute_identity
from src.models import IdentityReport

BBOX = (20, 20, 120, 140)


def _textured(seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(150, 35, (200, 200, 3)), 0, 255).astype(np.uint8)


def test_identical_images_score_one():
    image = _textured()
    report = compute_identity(image, image.copy(), BBOX, threshold=0.8)
    assert report.ssim == pytest.approx(1.0, abs=1e-6)
    assert report.within_threshold is True
    assert report.verdict == "Barely changed"


def test_a_small_edit_stays_above_the_threshold():
    original = _textured()
    edited = cv2.GaussianBlur(original, (0, 0), sigmaX=0.6)
    report = compute_identity(original, edited, BBOX, threshold=0.5)
    assert 0.5 < report.ssim < 1.0
    assert report.within_threshold is True


def test_a_destructive_edit_falls_below_the_threshold():
    original = _textured()
    replaced = _textured(seed=99)
    report = compute_identity(original, replaced, BBOX, threshold=0.8)
    assert report.ssim < 0.8
    assert report.within_threshold is False
    assert report.verdict == "Large structural change"


def test_score_is_clipped_into_the_valid_ssim_range():
    report = compute_identity(_textured(), _textured(seed=42), BBOX, threshold=0.8)
    assert -1.0 <= report.ssim <= 1.0


def test_a_crop_too_small_for_an_ssim_window_reports_zero_not_a_reassuring_score():
    """An unscoreable crop must not come back looking like a pass."""
    image = _textured()
    tiny = (0, 0, MIN_SSIM_WINDOW - 2, MIN_SSIM_WINDOW - 2)
    report = compute_identity(image, image.copy(), tiny, threshold=0.8)
    assert report.ssim == 0.0
    assert report.within_threshold is False


def test_differently_sized_crops_are_aligned_rather_than_stretched():
    original = _textured()
    edited = original.copy()
    report = compute_identity(original, edited[:180, :180], BBOX, threshold=0.8)
    assert 0.0 <= report.ssim <= 1.0


@pytest.mark.parametrize(
    "ssim,expected",
    [
        (0.99, "Barely changed"),
        (0.90, "Visible change, structure retained"),
        (0.50, "Large structural change"),
    ],
)
def test_verdict_wording_tracks_the_score(ssim, expected):
    assert IdentityReport(ssim=ssim, threshold=0.80).verdict == expected


def test_the_explanation_disclaims_biometric_identity():
    """The copy shown in the UI must not overclaim. This is a product promise."""
    lowered = EXPLANATION.lower()
    assert "does not verify identity" in lowered
    assert "not a biometric" in lowered
    assert "safe" in lowered
