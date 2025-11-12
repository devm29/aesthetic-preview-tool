"""Tests for the pure NumPy image helpers."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.imaging import (
    ImageTooLargeError,
    blend,
    crop,
    decode_image,
    downscale,
    encode_png,
    feather,
    image_fingerprint,
    mask_bbox,
    masked_filter,
    piecewise_warp,
    side_by_side,
    stamp_disclaimer,
    wipe_composite,
)


def test_decode_encode_roundtrip_preserves_pixels():
    original = np.random.default_rng(1).integers(0, 255, (40, 60, 3), dtype=np.uint8)
    decoded = decode_image(encode_png(original))
    assert decoded.shape == original.shape
    assert np.array_equal(decoded, original)


def test_decode_image_rejects_oversized_payload():
    payload = encode_png(np.zeros((64, 64, 3), np.uint8))
    with pytest.raises(ImageTooLargeError):
        decode_image(payload, max_bytes=8)


def test_decode_image_converts_greyscale_to_rgb():
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(np.full((16, 16), 120, np.uint8)).convert("L").save(buffer, format="PNG")
    decoded = decode_image(buffer.getvalue())
    assert decoded.shape == (16, 16, 3)


def test_downscale_shrinks_to_max_side_and_keeps_aspect():
    image = np.zeros((400, 800, 3), np.uint8)
    out = downscale(image, 200)
    assert max(out.shape[:2]) == 200
    assert out.shape[1] == 200 and out.shape[0] == 100


def test_downscale_never_upscales():
    image = np.zeros((50, 30, 3), np.uint8)
    assert downscale(image, 500) is image


def test_image_fingerprint_is_stable_and_content_sensitive():
    a = np.zeros((8, 8, 3), np.uint8)
    b = a.copy()
    b[0, 0] = 1
    assert image_fingerprint(a) == image_fingerprint(a.copy())
    assert image_fingerprint(a) != image_fingerprint(b)


def test_image_fingerprint_distinguishes_shapes_with_equal_bytes():
    flat = np.zeros((4, 12, 3), np.uint8)
    tall = np.zeros((12, 4, 3), np.uint8)
    assert image_fingerprint(flat) != image_fingerprint(tall)


def test_mask_bbox_returns_none_for_empty_mask():
    assert mask_bbox(np.zeros((20, 20), np.uint8)) is None


def test_mask_bbox_is_inclusive_of_the_last_set_pixel():
    mask = np.zeros((20, 20), np.uint8)
    mask[5:10, 4:8] = 255
    assert mask_bbox(mask) == (4, 5, 4, 5)


def test_mask_bbox_pad_is_clipped_to_the_image():
    mask = np.zeros((20, 20), np.uint8)
    mask[0:2, 0:2] = 255
    x, y, w, h = mask_bbox(mask, pad=50)
    assert (x, y) == (0, 0)
    assert (w, h) == (20, 20)


def test_crop_with_margin_stays_inside_bounds():
    image = np.zeros((50, 50, 3), np.uint8)
    out = crop(image, (0, 0, 10, 10), margin=1.0)
    assert out.shape[0] <= 50 and out.shape[1] <= 50
    assert out.size > 0


def test_feather_forces_an_odd_kernel():
    mask = np.zeros((30, 30), np.uint8)
    mask[10:20, 10:20] = 255
    # An even ksize would make cv2.GaussianBlur raise.
    assert feather(mask, ksize=10).shape == mask.shape


def test_blend_respects_mask_and_alpha():
    base = np.zeros((10, 10, 3), np.uint8)
    overlay = np.full((10, 10, 3), 200, np.uint8)
    mask = np.zeros((10, 10), np.uint8)
    mask[5:, :] = 255
    out = blend(base, overlay, mask, alpha=0.5)
    assert np.all(out[:5] == 0)
    assert np.all(out[5:] == 100)


def test_masked_filter_is_a_noop_at_zero_alpha():
    image = np.full((20, 20, 3), 40, np.uint8)
    mask = np.full((20, 20), 255, np.uint8)
    out = masked_filter(image, mask, lambda roi: np.zeros_like(roi), alpha=0.0)
    assert out is image


def test_masked_filter_is_a_noop_for_an_empty_mask():
    image = np.full((20, 20, 3), 40, np.uint8)
    out = masked_filter(image, np.zeros((20, 20), np.uint8), lambda roi: roi * 0, alpha=1.0)
    assert np.array_equal(out, image)


def test_masked_filter_only_touches_pixels_inside_the_mask():
    image = np.full((60, 60, 3), 40, np.uint8)
    mask = np.zeros((60, 60), np.uint8)
    mask[20:40, 20:40] = 255
    out = masked_filter(image, mask, lambda roi: np.full_like(roi, 255), alpha=1.0)
    assert np.all(out[20:40, 20:40] == 255)
    assert np.all(out[:20] == 40)
    assert np.all(out[45:] == 40)


def test_masked_filter_matches_a_full_frame_pass_for_a_shift_invariant_filter():
    """The ROI optimisation must not change the result.

    ``masked_filter`` exists purely to avoid filtering pixels the mask will
    discard. For a per-pixel filter the cropped and full-frame results have to
    agree exactly, otherwise the optimisation is silently changing output.
    """
    rng = np.random.default_rng(3)
    image = rng.integers(0, 255, (80, 80, 3), dtype=np.uint8)
    mask = np.zeros((80, 80), np.uint8)
    mask[30:50, 25:55] = 255

    def invert(roi: np.ndarray) -> np.ndarray:
        return (255 - roi.astype(np.int16)).astype(np.uint8)

    roi_result = masked_filter(image, mask, invert, alpha=1.0)
    full_result = blend(image, invert(image), mask, 1.0)
    assert np.array_equal(roi_result, full_result)


def test_piecewise_warp_moves_content_towards_the_destination_points():
    """Regression: ``skimage.warp`` takes an inverse map.

    Passing the forward transform runs the warp backwards, so a transform built
    to expand a region visibly shrinks it. This asserts the direction.
    """
    image = np.zeros((160, 160, 3), np.uint8)
    cv2.circle(image, (80, 80), 20, (255, 255, 255), -1)

    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    ring = np.stack([80 + 20 * np.cos(angles), 80 + 20 * np.sin(angles)], axis=1)
    grown = np.stack([80 + 34 * np.cos(angles), 80 + 34 * np.sin(angles)], axis=1)
    anchors = np.array(
        [[10, 10], [80, 10], [150, 10], [10, 80], [150, 80], [10, 150], [80, 150], [150, 150]],
        np.float32,
    )

    warped = piecewise_warp(
        image,
        np.vstack([ring, anchors]).astype(np.float32),
        np.vstack([grown, anchors]).astype(np.float32),
    )
    assert warped is not None
    before = int((image[..., 0] > 128).sum())
    after = int((warped[..., 0] > 128).sum())
    assert after > before * 1.3, "expansion warp shrank the region - inverse map is wrong"


def test_piecewise_warp_returns_none_for_a_degenerate_transform():
    image = np.zeros((40, 40, 3), np.uint8)
    points = np.zeros((6, 2), np.float32)
    assert piecewise_warp(image, points, points) is None


def test_side_by_side_matches_heights_and_inserts_a_gap():
    left = np.zeros((60, 40, 3), np.uint8)
    right = np.zeros((30, 40, 3), np.uint8)
    out = side_by_side(left, right, gap=6)
    assert out.shape[0] == 60
    assert out.shape[1] == 40 + 6 + 80  # right is scaled 2x to match the height


def test_wipe_composite_takes_the_before_image_on_the_left():
    before = np.zeros((20, 40, 3), np.uint8)
    after = np.full((20, 40, 3), 200, np.uint8)
    out = wipe_composite(before, after, 0.5)
    assert np.all(out[:, :18] == 0)
    assert np.all(out[:, 22:] == 200)


@pytest.mark.parametrize("position", [0.0, 1.0])
def test_wipe_composite_endpoints_are_the_pure_images(position):
    before = np.zeros((10, 20, 3), np.uint8)
    after = np.full((10, 20, 3), 200, np.uint8)
    out = wipe_composite(before, after, position)
    expected = after if position == 0.0 else before
    assert np.array_equal(out, expected)


def test_stamp_disclaimer_darkens_a_band_and_leaves_the_rest_alone():
    image = np.full((120, 300, 3), 180, np.uint8)
    out = stamp_disclaimer(image, "NOT A CLINICAL OUTCOME")
    assert out.shape == image.shape
    assert out[:40].mean() == pytest.approx(180, abs=1)
    assert out[-6:].mean() < 150
