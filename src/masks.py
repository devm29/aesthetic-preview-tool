"""Region masks derived from face geometry.

Masks are uint8, 0 or 255, and always the full size of the source image. Where
landmarks exist the masks are polygonal; where only a bounding box exists the
skin mask degrades to an inset ellipse and the finer regions are unavailable.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from .models import BBox, FaceGeometry

# FaceMesh ring indices. Using the outer-minus-inner ring keeps teeth and the
# mouth interior out of lip edits.
LIP_OUTER_IDX: Sequence[int] = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291)
LIP_INNER_IDX: Sequence[int] = (78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308)
LEFT_EYE_IDX: Sequence[int] = (33, 160, 158, 133, 153, 144)
RIGHT_EYE_IDX: Sequence[int] = (362, 385, 387, 263, 373, 380)
MIN_MESH_POINTS = 468


def _blank(image_rgb: np.ndarray) -> np.ndarray:
    return np.zeros(image_rgb.shape[:2], dtype=np.uint8)


def _has_mesh(landmarks: np.ndarray | None) -> bool:
    return landmarks is not None and len(landmarks) >= MIN_MESH_POINTS


def skin_mask_from_bbox(image_rgb: np.ndarray, bbox: BBox) -> np.ndarray:
    """Inset ellipse inside the face box - a crude but safe skin approximation."""
    mask = _blank(image_rgb)
    x, y, w, h = bbox
    centre = (x + w // 2, y + h // 2)
    axes = (max(1, int(w * 0.45)), max(1, int(h * 0.55)))
    cv2.ellipse(mask, centre, axes, 0, 0, 360, 255, thickness=-1)
    return mask


def skin_mask_from_landmarks(image_rgb: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Convex hull over all face landmarks."""
    mask = _blank(image_rgb)
    hull = cv2.convexHull(np.asarray(landmarks, np.int32))
    cv2.fillConvexPoly(mask, hull, 255)
    return mask


def skin_mask(image_rgb: np.ndarray, geometry: FaceGeometry) -> np.ndarray:
    if geometry.has_landmarks:
        return skin_mask_from_landmarks(image_rgb, geometry.landmarks)
    return skin_mask_from_bbox(image_rgb, geometry.bbox)


def lip_points(landmarks: np.ndarray | None) -> np.ndarray | None:
    if not _has_mesh(landmarks):
        return None
    return np.asarray(landmarks, np.float32)[list(LIP_OUTER_IDX)]


def lip_ring_mask(image_rgb: np.ndarray, landmarks: np.ndarray | None) -> np.ndarray | None:
    """Lip tissue only: outer lip hull minus inner lip hull."""
    if not _has_mesh(landmarks):
        return None
    points = np.asarray(landmarks, np.int32)
    outer, inner = _blank(image_rgb), _blank(image_rgb)
    cv2.fillConvexPoly(outer, cv2.convexHull(points[list(LIP_OUTER_IDX)]), 255)
    cv2.fillConvexPoly(inner, cv2.convexHull(points[list(LIP_INNER_IDX)]), 255)
    ring = cv2.subtract(outer, inner)
    return ring if np.any(ring) else None


def under_eye_mask(
    image_rgb: np.ndarray, landmarks: np.ndarray | None, side: str
) -> np.ndarray | None:
    """Ellipse sitting just below one eye."""
    if not _has_mesh(landmarks):
        return None
    idx = LEFT_EYE_IDX if side.lower().startswith("l") else RIGHT_EYE_IDX
    polygon = np.asarray(landmarks, np.float32)[list(idx)]
    min_x, min_y = polygon.min(axis=0)
    max_x, max_y = polygon.max(axis=0)
    width = max(8.0, float(max_x - min_x))
    height = max(6.0, float(max_y - min_y))
    centre = (int((min_x + max_x) / 2), int((min_y + max_y) / 2 + height * 0.85))
    axes = (max(1, int(width * 0.55)), max(1, int(height * 0.85)))
    mask = _blank(image_rgb)
    cv2.ellipse(mask, centre, axes, 0, 0, 360, 255, thickness=-1)
    return mask


def under_eye_union(image_rgb: np.ndarray, landmarks: np.ndarray | None) -> np.ndarray | None:
    left = under_eye_mask(image_rgb, landmarks, "left")
    right = under_eye_mask(image_rgb, landmarks, "right")
    if left is None and right is None:
        return None
    mask = _blank(image_rgb)
    for part in (left, right):
        if part is not None:
            mask = cv2.bitwise_or(mask, part)
    return mask


def region_masks(image_rgb: np.ndarray, geometry: FaceGeometry) -> dict:
    """All landmark-derived region masks that are computable for this face."""
    masks = {"skin": skin_mask(image_rgb, geometry)}
    if not geometry.has_landmarks:
        return masks
    lips = lip_ring_mask(image_rgb, geometry.landmarks)
    if lips is not None:
        masks["lip_ring"] = lips
    under_eye = under_eye_union(image_rgb, geometry.landmarks)
    if under_eye is not None:
        masks["under_eye"] = under_eye
    return masks
