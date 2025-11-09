"""Upload quality control.

Each check returns a :class:`CheckResult` so the UI can render a consistent
row per check, including checks that had to be skipped.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from .config import QCThresholds, Settings, get_settings
from .detection import detect_face
from .models import CheckResult, FaceGeometry, QCReport

logger = logging.getLogger(__name__)


def _to_gray(image_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)


def check_blur(image_rgb: np.ndarray, thresholds: QCThresholds | None = None) -> CheckResult:
    """Variance of the Laplacian at a fixed width, so the score is size-stable."""
    thresholds = thresholds or QCThresholds()
    height, width = image_rgb.shape[:2]
    target = thresholds.blur_target_width
    if width > target:
        scaled = cv2.resize(
            image_rgb,
            (target, max(1, int(height * target / float(width)))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        scaled = image_rgb
    variance = float(cv2.Laplacian(_to_gray(scaled), cv2.CV_64F).var())
    passed = variance >= thresholds.blur_variance_min
    return CheckResult(
        name="Sharpness",
        status="pass" if passed else "fail",
        value=variance,
        unit="Laplacian variance",
        detail=(
            "Detail looks sufficient."
            if passed
            else "Image looks soft. Hold the camera steady and refocus."
        ),
    )


def check_lighting(image_rgb: np.ndarray, thresholds: QCThresholds | None = None) -> CheckResult:
    """Brightness, contrast and clipping, measured on the HSV value channel."""
    thresholds = thresholds or QCThresholds()
    value = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)[:, :, 2]
    mean, std = float(np.mean(value)), float(np.std(value))
    over, under = float(np.mean(value > 240)), float(np.mean(value < 15))
    reasons: list[str] = []
    if not thresholds.lighting_mean_min <= mean <= thresholds.lighting_mean_max:
        reasons.append("too dark" if mean < thresholds.lighting_mean_min else "too bright")
    if std < thresholds.lighting_std_min:
        reasons.append("very flat")
    if over >= thresholds.overexposed_max_frac:
        reasons.append("blown highlights")
    if under >= thresholds.underexposed_max_frac:
        reasons.append("crushed shadows")
    passed = not reasons
    return CheckResult(
        name="Lighting",
        status="pass" if passed else "fail",
        value=mean,
        unit="mean brightness",
        detail=(
            f"Even light (contrast {std:.0f})."
            if passed
            else "Lighting is " + ", ".join(reasons) + ". Use even, indirect light."
        ),
    )


def check_orientation(
    geometry: FaceGeometry, thresholds: QCThresholds | None = None
) -> CheckResult:
    """Head roll. Skipped - not passed - when the detector cannot measure it."""
    thresholds = thresholds or QCThresholds()
    if geometry.roll_deg is None:
        return CheckResult(
            name="Head tilt",
            status="skipped",
            detail=(
                f"The {geometry.detector} detector does not provide landmarks, "
                "so head tilt could not be measured."
            ),
        )
    roll = float(geometry.roll_deg)
    passed = abs(roll) <= thresholds.roll_max_abs_deg
    return CheckResult(
        name="Head tilt",
        status="pass" if passed else "fail",
        value=roll,
        unit="degrees",
        detail=("Head is upright." if passed else "Straighten your head towards level."),
    )


def check_face_size(image_rgb: np.ndarray, geometry: FaceGeometry) -> CheckResult:
    """The face has to fill enough of the frame for the edits to be meaningful."""
    height, width = image_rgb.shape[:2]
    _, _, w, h = geometry.bbox
    coverage = float(w * h) / float(max(1, width * height))
    passed = coverage >= 0.04
    return CheckResult(
        name="Face size",
        status="pass" if passed else "fail",
        value=coverage * 100.0,
        unit="% of frame",
        detail=("Face fills enough of the frame." if passed else "Move closer to the camera."),
    )


def run_qc(image_rgb: np.ndarray, settings: Settings | None = None) -> QCReport:
    """Detect a face and run every check that applies to it."""
    settings = settings or get_settings()
    thresholds = settings.qc
    geometry = detect_face(image_rgb, settings.detector_preference)
    if geometry is None:
        logger.info("QC: no face detected.")
        return QCReport(
            checks=(
                CheckResult(
                    name="Face detected",
                    status="fail",
                    detail="No face found. Centre your face, look at the camera and retry.",
                ),
            ),
            face=None,
        )
    checks = (
        CheckResult(
            name="Face detected",
            status="pass",
            detail=f"One face located by the {geometry.detector} detector.",
        ),
        check_face_size(image_rgb, geometry),
        check_blur(image_rgb, thresholds),
        check_lighting(image_rgb, thresholds),
        check_orientation(geometry, thresholds),
    )
    return QCReport(checks=checks, face=geometry)
