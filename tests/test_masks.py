"""Tests for the landmark- and bbox-derived region masks."""

from __future__ import annotations

import numpy as np
import pytest

from src import masks
from src.models import FaceGeometry

from .conftest import make_geometry, make_mesh


@pytest.fixture
def image() -> np.ndarray:
    return np.zeros((256, 256, 3), np.uint8)


def test_skin_mask_from_bbox_covers_the_bbox_centre(image):
    bbox = (50, 50, 100, 100)
    mask = masks.skin_mask_from_bbox(image, bbox)
    assert mask.shape == image.shape[:2]
    assert mask.dtype == np.uint8
    assert mask[100, 100] == 255
    # The ellipse is inset, so the bbox corners must stay outside it.
    assert mask[50, 50] == 0


def test_skin_mask_from_bbox_handles_a_one_pixel_box(image):
    mask = masks.skin_mask_from_bbox(image, (10, 10, 1, 1))
    assert mask.shape == image.shape[:2]


def test_skin_mask_uses_landmarks_when_they_exist(image):
    geometry = make_geometry(256, 256, with_landmarks=True)
    hull = masks.skin_mask(image, geometry)
    bbox_based = masks.skin_mask_from_bbox(image, geometry.bbox)
    assert np.any(hull)
    # The convex hull of the mesh is a different shape from the inset ellipse.
    assert not np.array_equal(hull, bbox_based)


def test_skin_mask_falls_back_to_the_bbox_ellipse_without_landmarks(image):
    geometry = make_geometry(256, 256, with_landmarks=False)
    assert np.array_equal(
        masks.skin_mask(image, geometry),
        masks.skin_mask_from_bbox(image, geometry.bbox),
    )


def test_lip_ring_mask_is_none_without_a_full_mesh(image):
    assert masks.lip_ring_mask(image, None) is None
    assert masks.lip_ring_mask(image, np.zeros((10, 2), np.float32)) is None


def test_lip_ring_mask_is_none_for_degenerate_landmarks(image):
    """Regression: the shipped suite asserted a non-empty ring here.

    Collapsing every landmark to one point produces an empty outer-minus-inner
    ring. Returning ``None`` lets the caller skip the treatment and say why,
    which is what the pipeline now does.
    """
    collapsed = np.full((masks.MIN_MESH_POINTS, 2), 100.0, np.float32)
    assert masks.lip_ring_mask(image, collapsed) is None


def test_lip_ring_mask_excludes_the_mouth_interior(image):
    mesh = make_mesh(256, 256)
    ring = masks.lip_ring_mask(image, mesh)
    assert ring is not None
    assert np.any(ring)
    inner_centre = np.asarray(mesh, np.int32)[list(masks.LIP_INNER_IDX)].mean(axis=0)
    assert ring[int(inner_centre[1]), int(inner_centre[0])] == 0


def test_lip_points_returns_the_outer_ring(image):
    mesh = make_mesh(256, 256)
    points = masks.lip_points(mesh)
    assert points is not None
    assert points.shape == (len(masks.LIP_OUTER_IDX), 2)
    assert masks.lip_points(None) is None


@pytest.mark.parametrize("side", ["left", "right", "L", "R"])
def test_under_eye_mask_sits_below_the_eye(image, side):
    mesh = make_mesh(256, 256)
    mask = masks.under_eye_mask(image, mesh, side)
    assert mask is not None
    assert np.any(mask)
    idx = masks.LEFT_EYE_IDX if side.lower().startswith("l") else masks.RIGHT_EYE_IDX
    eye_y = np.asarray(mesh)[list(idx)][:, 1].mean()
    mask_y = np.nonzero(mask)[0].mean()
    assert mask_y > eye_y


def test_under_eye_mask_is_none_without_a_full_mesh(image):
    assert masks.under_eye_mask(image, np.zeros((10, 2), np.float32), "left") is None


def test_under_eye_union_covers_both_sides(image):
    mesh = make_mesh(256, 256)
    union = masks.under_eye_union(image, mesh)
    left = masks.under_eye_mask(image, mesh, "left")
    right = masks.under_eye_mask(image, mesh, "right")
    assert union is not None
    assert int(union.sum()) >= max(int(left.sum()), int(right.sum()))
    assert masks.under_eye_union(image, None) is None


def test_region_masks_only_offers_skin_without_landmarks(image):
    geometry = FaceGeometry(bbox=(40, 40, 80, 100), detector="fake")
    assert set(masks.region_masks(image, geometry)) == {"skin"}


def test_region_masks_offers_every_region_with_a_mesh(image):
    geometry = make_geometry(256, 256, with_landmarks=True)
    assert set(masks.region_masks(image, geometry)) == {"skin", "lip_ring", "under_eye"}


def test_every_mask_is_binary_uint8_and_full_size(image):
    geometry = make_geometry(256, 256, with_landmarks=True)
    for name, mask in masks.region_masks(image, geometry).items():
        assert mask.dtype == np.uint8, name
        assert mask.shape == image.shape[:2], name
        assert set(np.unique(mask)).issubset({0, 255}), name
