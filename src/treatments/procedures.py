"""Landmark-driven procedure previews: lips and under-eye.

Each procedure has two engines. The parametric engine is a geometric warp or a
local tone edit and always works. The generative engine repaints the same mask
with SDXL when the optional ML stack is installed; if it fails for any reason
the parametric engine takes over and the fallback is recorded as a note.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from ..imaging import blend, feather, masked_filter, piecewise_warp
from ..masks import lip_points, lip_ring_mask, under_eye_union
from ..models import TreatmentContext
from .base import PROCEDURE, Treatment, TreatmentSpec, register

logger = logging.getLogger(__name__)

LIP_PROMPT = (
    "natural lip augmentation, fuller lips, realistic shading and skin texture, "
    "neutral tone, photographic, high detail"
)
LIP_NEGATIVE = "over-smooth, plastic skin, lipstick, heavy makeup, cartoon, deformed"
EYE_PROMPT = (
    "reduce under-eye shadowing, smooth under-eye skin, even tone, keep pore detail, "
    "realistic portrait, photographic"
)
EYE_NEGATIVE = "over-smooth, plastic, blur, cartoon, airbrushed"


def _scale(intensity: int) -> float:
    return float(np.clip(intensity, 0, 100)) / 100.0


class _GenerativeCapable(Treatment):
    """Shared generative-then-parametric dispatch."""

    def _mask(self, image_rgb: np.ndarray, context: TreatmentContext):
        raise NotImplementedError

    def _parametric(self, image_rgb, context, intensity, mask):
        raise NotImplementedError

    def _dilate_fraction(self) -> float:
        return 0.015

    def apply(self, image_rgb: np.ndarray, context: TreatmentContext, intensity: int) -> np.ndarray:
        strength = _scale(intensity)
        if strength <= 0:
            return image_rgb
        mask = self._mask(context.original, context)
        if mask is None:
            context.note(f"{self.spec.label}: region could not be located on this face.")
            return image_rgb

        editor = context.editor
        if self.spec.supports_generative and editor.available():
            try:
                face_width = max(1, context.geometry.bbox[2])
                radius = int(np.clip(face_width * self._dilate_fraction() * strength, 1, 24))
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
                )
                return editor.inpaint(
                    image_rgb,
                    cv2.dilate(mask, kernel, iterations=1),
                    prompt=self.spec.prompt,
                    negative_prompt=self.spec.negative_prompt,
                    strength=float(np.clip(0.30 + 0.5 * strength, 0.1, 0.9)),
                    steps=14,
                    seed=42,
                )
            except Exception as exc:  # pragma: no cover - needs the ML stack
                logger.warning("Generative %s failed: %s", self.spec.key, exc)
                context.note(
                    f"{self.spec.label}: generative pass failed, used the parametric engine."
                )
        return self._parametric(image_rgb, context, intensity, mask)


@register
class LipVolume(_GenerativeCapable):
    spec = TreatmentSpec(
        key="lip",
        label="Lip volume",
        summary=(
            "Expands the lip ring along its short axis with a piecewise-affine warp, "
            "attenuated at the corners."
        ),
        group=PROCEDURE,
        order=40,
        requires_landmarks=True,
        supports_generative=True,
        prompt=LIP_PROMPT,
        negative_prompt=LIP_NEGATIVE,
    )

    def _mask(self, image_rgb, context):
        return lip_ring_mask(image_rgb, context.geometry.landmarks)

    def _parametric(self, image_rgb, context, intensity, mask):
        points = lip_points(context.geometry.landmarks)
        if points is None or len(points) < 3:
            return image_rgb
        strength = _scale(intensity)

        centre = points.mean(axis=0, keepdims=True)
        centred = points - centre
        # Principal axes: the long axis runs across the mouth, the short axis is
        # the one that should grow when lips get fuller.
        eigenvalues, eigenvectors = np.linalg.eigh(np.cov(centred.T))
        basis = eigenvectors[:, np.argsort(eigenvalues)[::-1]]
        in_basis = centred @ basis

        left = points[np.argmin(points[:, 0])]
        right = points[np.argmax(points[:, 0])]
        lip_width = max(8.0, float(right[0] - left[0]))
        corner_distance = np.minimum(
            np.linalg.norm(points - left, axis=1), np.linalg.norm(points - right, axis=1)
        )
        # Corners barely move, otherwise the mouth takes on a "fish" shape.
        corner_weight = 0.25 + 0.75 * np.clip(corner_distance / (0.45 * lip_width), 0.0, 1.0)

        scaled = in_basis.copy()
        scaled[:, 0] *= 1.0 + 0.05 * strength * 0.5 * corner_weight
        scaled[:, 1] *= 1.0 + 0.45 * strength * corner_weight
        destination = scaled @ basis.T + centre

        min_xy, max_xy = points.min(axis=0), points.max(axis=0)
        pad = np.maximum((max_xy - min_xy) * 0.25, 6.0)
        x0, y0 = min_xy - pad
        x1, y1 = max_xy + pad
        mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
        # Anchors pin the surrounding skin so the warp stays local.
        # fmt: off  (laid out as the 3x3 grid it represents)
        anchors = np.array(
            [
                [x0, y0],
                [mid_x, y0],
                [x1, y0],
                [x0, mid_y],
                [x1, mid_y],
                [x0, y1],
                [mid_x, y1],
                [x1, y1],
            ],
            np.float32,
        )
        # fmt: on
        warped = piecewise_warp(
            image_rgb,
            np.vstack([points, anchors]),
            np.vstack([destination, anchors]),
        )
        if warped is None:
            return image_rgb

        hsv = cv2.cvtColor(warped, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + 0.25 * strength), 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1.0 + 0.06 * strength), 0, 255)
        tinted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        grown = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1)
        return blend(image_rgb, tinted, feather(grown, 9), alpha=1.0)


@register
class UnderEyeSoftening(_GenerativeCapable):
    spec = TreatmentSpec(
        key="under_eye",
        label="Under-eye softening",
        summary="Lifts and de-saturates the under-eye area, then suppresses fine detail.",
        group=PROCEDURE,
        order=50,
        requires_landmarks=True,
        supports_generative=True,
        prompt=EYE_PROMPT,
        negative_prompt=EYE_NEGATIVE,
    )

    def _dilate_fraction(self) -> float:
        return 0.010

    def _mask(self, image_rgb, context):
        return under_eye_union(image_rgb, context.geometry.landmarks)

    @staticmethod
    def _reference_chroma(image_rgb: np.ndarray, context: TreatmentContext, mask: np.ndarray):
        """Median a/b of the face skin *outside* the under-eye region.

        Shifting the chroma by a fixed amount (the original "+18 on b") pushes
        pale or cool skin straight past neutral into a green-yellow cast. Pulling
        towards the surrounding skin's own chroma is both what the treatment
        claims to do and impossible to overshoot.
        """
        grown = cv2.dilate(
            mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=1
        )
        reference = cv2.subtract(context.skin_mask, grown)
        if not np.any(reference):
            return None
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        selected = reference > 0
        return (
            float(np.median(lab[:, :, 1][selected])),
            float(np.median(lab[:, :, 2][selected])),
        )

    def _parametric(self, image_rgb, context, intensity, mask):
        strength = _scale(intensity)
        kernel_size = max(3, int(5 + 6 * strength)) | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        soft_mask = feather(cv2.erode(mask, kernel, iterations=1), 11)
        chroma = self._reference_chroma(image_rgb, context, mask)
        if chroma is None:
            context.note(
                f"{self.spec.label}: no surrounding skin to sample, so only the "
                "brightness was lifted."
            )

        def lighten(roi: np.ndarray) -> np.ndarray:
            lab = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB)
            lightness, a_channel, b_channel = cv2.split(lab)
            lightness_f = lightness.astype(np.float32)
            lifted = np.clip(lightness_f + (255.0 - lightness_f) * (0.35 * strength), 0, 255)
            if chroma is None:
                a_new, b_new = a_channel.astype(np.float32), b_channel.astype(np.float32)
            else:
                # Interpolate towards the surrounding skin; weight < 1 never overshoots.
                weight = 0.65 * strength
                a_new = a_channel.astype(np.float32) * (1 - weight) + chroma[0] * weight
                b_new = b_channel.astype(np.float32) * (1 - weight) + chroma[1] * weight
            brightened = cv2.cvtColor(
                cv2.merge(
                    [
                        lifted.astype(np.uint8),
                        np.clip(a_new, 0, 255).astype(np.uint8),
                        np.clip(b_new, 0, 255).astype(np.uint8),
                    ]
                ),
                cv2.COLOR_LAB2RGB,
            )
            base = cv2.GaussianBlur(brightened, (0, 0), sigmaX=2, sigmaY=2)
            detail = brightened.astype(np.float32) - base.astype(np.float32)
            return np.clip(
                base.astype(np.float32) + (1.0 - 0.6 * strength) * detail, 0, 255
            ).astype(np.uint8)

        return masked_filter(image_rgb, soft_mask, lighten, alpha=1.0)
