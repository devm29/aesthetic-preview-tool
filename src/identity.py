"""Identity-preservation signal.

SSIM over the face crop measures how much image *structure* changed. It is a
guard-rail against an edit quietly turning someone into a different-looking
person - not a biometric identity check, and not a safety guarantee.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from .imaging import crop
from .models import BBox, IdentityReport

MIN_SSIM_WINDOW = 7

EXPLANATION = (
    "Structural similarity (SSIM) is computed on the greyscale face crop, before "
    "versus after. It tells you how much of the original image structure survived "
    "the edit. It does not verify identity, is not a biometric match score, and a "
    "high value does not make an edit accurate or safe."
)


def compute_identity(
    original_rgb: np.ndarray,
    edited_rgb: np.ndarray,
    bbox: BBox,
    threshold: float,
    margin: float = 0.1,
) -> IdentityReport:
    """SSIM between the original and edited face crops."""
    original_crop = cv2.cvtColor(crop(original_rgb, bbox, margin), cv2.COLOR_RGB2GRAY)
    edited_crop = cv2.cvtColor(crop(edited_rgb, bbox, margin), cv2.COLOR_RGB2GRAY)

    height = min(original_crop.shape[0], edited_crop.shape[0])
    width = min(original_crop.shape[1], edited_crop.shape[1])
    if height < MIN_SSIM_WINDOW or width < MIN_SSIM_WINDOW:
        # Too small for an SSIM window; report a neutral, honest 0.0 rather than
        # inventing a reassuring number.
        return IdentityReport(ssim=0.0, threshold=threshold)
    original_crop = original_crop[:height, :width]
    edited_crop = edited_crop[:height, :width]

    score = structural_similarity(original_crop, edited_crop, data_range=255)
    return IdentityReport(ssim=float(np.clip(score, -1.0, 1.0)), threshold=threshold)
