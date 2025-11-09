"""Non-generative, skin-mask-aware adjustments.

All three run classical OpenCV filters inside the skin mask's bounding box and
blend the result back with the intensity as the blend weight.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..imaging import masked_filter
from ..models import TreatmentContext
from .base import ADJUSTMENT, Treatment, TreatmentSpec, register


def _alpha(intensity: int) -> float:
    return float(np.clip(intensity, 0, 100)) / 100.0


@register
class WrinkleSmoothing(Treatment):
    spec = TreatmentSpec(
        key="wrinkle",
        label="Wrinkle smoothing",
        summary="Edge-preserving bilateral smoothing across the skin region.",
        group=ADJUSTMENT,
        order=10,
        default=40,
    )

    def apply(self, image_rgb: np.ndarray, context: TreatmentContext, intensity: int) -> np.ndarray:
        alpha = _alpha(intensity)
        if alpha <= 0:
            return image_rgb

        def smooth(roi: np.ndarray) -> np.ndarray:
            bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
            filtered = cv2.bilateralFilter(bgr, d=9, sigmaColor=75, sigmaSpace=75)
            return cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)

        return masked_filter(image_rgb, context.skin_mask, smooth, alpha)


@register
class ToneEvening(Treatment):
    spec = TreatmentSpec(
        key="tone",
        label="Tone evening",
        summary="CLAHE on the L channel to even out local luminance.",
        group=ADJUSTMENT,
        order=20,
        default=25,
    )

    def apply(self, image_rgb: np.ndarray, context: TreatmentContext, intensity: int) -> np.ndarray:
        alpha = _alpha(intensity)
        if alpha <= 0:
            return image_rgb

        def equalise(roi: np.ndarray) -> np.ndarray:
            lab = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB)
            lightness, a_channel, b_channel = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            merged = cv2.merge([clahe.apply(lightness), a_channel, b_channel])
            return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

        return masked_filter(image_rgb, context.skin_mask, equalise, alpha)


@register
class PigmentationReduction(Treatment):
    spec = TreatmentSpec(
        key="pigment",
        label="Pigmentation evening",
        summary="Lifts skin areas darker than their local neighbourhood.",
        group=ADJUSTMENT,
        order=30,
        default=20,
    )

    def apply(self, image_rgb: np.ndarray, context: TreatmentContext, intensity: int) -> np.ndarray:
        alpha = _alpha(intensity)
        if alpha <= 0:
            return image_rgb

        def lift(roi: np.ndarray) -> np.ndarray:
            local = cv2.GaussianBlur(roi, (0, 0), sigmaX=7, sigmaY=7).astype(np.float32)
            darker_than_local = np.maximum(local - roi.astype(np.float32), 0.0)
            return np.clip(roi.astype(np.float32) + 0.6 * darker_than_local, 0, 255).astype(
                np.uint8
            )

        return masked_filter(image_rgb, context.skin_mask, lift, alpha)
