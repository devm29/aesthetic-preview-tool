"""Tests for the upload quality checks."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src import qc
from src.config import QCThresholds, Settings
from src.models import FaceGeometry

from .conftest import make_geometry


def _noisy(shape=(256, 256, 3), mean=140.0, sigma=40.0, seed=5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(mean, sigma, shape), 0, 255).astype(np.uint8)


def test_check_blur_passes_on_a_detailed_image():
    result = qc.check_blur(_noisy())
    assert result.status == "pass"
    assert result.value > QCThresholds().blur_variance_min
    assert result.unit == "Laplacian variance"


def test_check_blur_fails_on_a_flat_image():
    result = qc.check_blur(np.full((256, 256, 3), 128, np.uint8))
    assert result.status == "fail"
    assert result.value == pytest.approx(0.0)
    assert "soft" in result.detail.lower()


def test_check_blur_fails_after_heavy_blurring():
    sharp = _noisy()
    blurred = cv2.GaussianBlur(sharp, (0, 0), sigmaX=9)
    assert qc.check_blur(sharp).status == "pass"
    assert qc.check_blur(blurred).status == "fail"


def test_check_blur_normalises_width_before_scoring(sample_image):
    """The score is taken at a fixed width so the threshold means the same
    thing for a phone photo and a webcam frame."""
    thresholds = QCThresholds()
    wide = cv2.resize(sample_image, (1600, 1600), interpolation=cv2.INTER_CUBIC)
    pre_scaled = cv2.resize(
        wide,
        (thresholds.blur_target_width, thresholds.blur_target_width),
        interpolation=cv2.INTER_AREA,
    )
    assert qc.check_blur(wide, thresholds).value == pytest.approx(
        qc.check_blur(pre_scaled, thresholds).value, rel=0.01
    )


def test_check_blur_leaves_images_narrower_than_the_target_alone():
    small = _noisy((200, 200, 3))
    assert qc.check_blur(small).value == pytest.approx(
        float(cv2.Laplacian(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())
    )


def test_check_lighting_passes_on_an_evenly_lit_image():
    result = qc.check_lighting(_noisy(mean=140.0, sigma=45.0))
    assert result.status == "pass"
    assert result.unit == "mean brightness"


@pytest.mark.parametrize(
    "image,expected",
    [
        (np.full((64, 64, 3), 10, np.uint8), "too dark"),
        (np.full((64, 64, 3), 250, np.uint8), "too bright"),
        (np.full((64, 64, 3), 140, np.uint8), "very flat"),
    ],
)
def test_check_lighting_names_the_problem(image, expected):
    result = qc.check_lighting(image)
    assert result.status == "fail"
    assert expected in result.detail


def test_check_lighting_flags_blown_highlights():
    image = _noisy(mean=140.0, sigma=40.0)
    image[:, :40] = 255  # ~16% of the frame clipped
    result = qc.check_lighting(image)
    assert result.status == "fail"
    assert "blown highlights" in result.detail


def test_check_orientation_is_skipped_when_roll_cannot_be_measured():
    """Regression: the original code reported roll 0.0 for the Haar fallback.

    A bbox-only detector cannot measure head tilt. Reporting a pass is a
    fabricated signal; "skipped" is the honest answer.
    """
    geometry = FaceGeometry(bbox=(0, 0, 10, 10), detector="haar", roll_deg=None)
    result = qc.check_orientation(geometry)
    assert result.status == "skipped"
    assert result.passed is False
    assert result.value is None
    assert "haar" in result.detail


@pytest.mark.parametrize("roll,status", [(0.0, "pass"), (15.0, "pass"), (35.0, "fail")])
def test_check_orientation_uses_the_threshold(roll, status):
    geometry = make_geometry(roll_deg=roll)
    assert qc.check_orientation(geometry).status == status


def test_check_face_size_fails_for_a_distant_face():
    image = np.zeros((600, 600, 3), np.uint8)
    small = FaceGeometry(bbox=(0, 0, 40, 40), detector="fake")
    large = FaceGeometry(bbox=(100, 100, 400, 400), detector="fake")
    assert qc.check_face_size(image, small).status == "fail"
    assert qc.check_face_size(image, large).status == "pass"
    assert qc.check_face_size(image, large).unit == "% of frame"


def test_run_qc_reports_a_single_failing_check_when_no_face_is_found(settings):
    report = qc.run_qc(np.zeros((256, 256, 3), np.uint8), settings)
    assert report.face_present is False
    assert report.passed is False
    assert len(report.checks) == 1
    assert report.checks[0].name == "Face detected"
    assert "No face found" in report.checks[0].detail


def test_run_qc_on_the_sample_passes_every_measurable_check(sample_report):
    assert sample_report.face_present is True
    assert sample_report.failures == ()
    assert sample_report.get("Sharpness").status == "pass"
    assert sample_report.get("Lighting").status == "pass"


def test_qc_report_passed_ignores_skipped_checks(sample_report):
    """A skipped check must not fail the report, and must not silently pass it."""
    assert sample_report.passed is True
    for check in sample_report.skipped:
        assert check.status == "skipped"
        assert check.passed is False


def test_qc_report_get_returns_none_for_an_unknown_check(sample_report):
    assert sample_report.get("Nonexistent") is None


def test_run_qc_uses_the_thresholds_from_settings(sample_image):
    """Settings are threaded through rather than read from module globals."""
    impossible = Settings(
        detector_preference="haar",
        qc=QCThresholds(blur_variance_min=1e9),
    )
    report = qc.run_qc(sample_image, impossible)
    if report.face is None:  # pragma: no cover - slim OpenCV builds only
        pytest.skip("No face detector available.")
    assert report.get("Sharpness").status == "fail"
