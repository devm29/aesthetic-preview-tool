"""Pure image helpers.

No Streamlit, no ML frameworks: everything in here is a NumPy in / NumPy out
function, which is what makes the pipeline testable headlessly.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import logging
from collections.abc import Callable

import cv2
import numpy as np
from PIL import Image, ImageOps
from skimage.transform import PiecewiseAffineTransform, warp

logger = logging.getLogger(__name__)

BBox = tuple[int, int, int, int]


class ImageTooLargeError(ValueError):
    """Raised when an upload exceeds the configured size budget."""


def decode_image(data: bytes, max_bytes: int | None = None) -> np.ndarray:
    """Decode image bytes to an RGB array, honouring EXIF orientation."""
    if max_bytes is not None and len(data) > max_bytes:
        raise ImageTooLargeError(
            f"Image is {len(data) / 1e6:.1f} MB; the limit is {max_bytes / 1e6:.1f} MB."
        )
    with Image.open(io.BytesIO(data)) as img:
        with contextlib.suppress(Exception):  # malformed EXIF is not fatal
            img = ImageOps.exif_transpose(img)
        return np.asarray(img.convert("RGB"))


def encode_png(image_rgb: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(image_rgb).save(buffer, format="PNG")
    return buffer.getvalue()


def image_fingerprint(image_rgb: np.ndarray) -> str:
    """Stable short hash of pixel content, used as a cache key."""
    digest = hashlib.sha256()
    digest.update(str(image_rgb.shape).encode())
    digest.update(np.ascontiguousarray(image_rgb).tobytes())
    return digest.hexdigest()[:16]


def downscale(image_rgb: np.ndarray, max_side: int) -> np.ndarray:
    """Shrink so the longest side is at most ``max_side``. Never upscales."""
    height, width = image_rgb.shape[:2]
    longest = max(height, width)
    if longest <= max_side or longest == 0:
        return image_rgb
    scale = max_side / float(longest)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image_rgb, size, interpolation=cv2.INTER_AREA)


def mask_bbox(mask: np.ndarray, pad: int = 0) -> BBox | None:
    """Bounding box (x, y, w, h) of the non-zero mask area, padded and clipped."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    height, width = mask.shape[:2]
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(width, int(xs.max()) + pad + 1)
    y1 = min(height, int(ys.max()) + pad + 1)
    return x0, y0, x1 - x0, y1 - y0


def crop(image_rgb: np.ndarray, bbox: BBox, margin: float = 0.0) -> np.ndarray:
    x, y, w, h = bbox
    height, width = image_rgb.shape[:2]
    mx, my = int(w * margin), int(h * margin)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(width, x + w + mx), min(height, y + h + my)
    return image_rgb[y0:y1, x0:x1]


def feather(mask: np.ndarray, ksize: int = 11) -> np.ndarray:
    ksize = max(1, ksize | 1)
    return cv2.GaussianBlur(mask, (ksize, ksize), 0)


def blend(base: np.ndarray, overlay: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    """Blend ``overlay`` into ``base`` where ``mask`` is set, scaled by ``alpha``."""
    weight = (mask.astype(np.float32) / 255.0 * float(alpha))[..., None]
    out = base.astype(np.float32) * (1.0 - weight) + overlay.astype(np.float32) * weight
    return np.clip(out, 0, 255).astype(np.uint8)


def masked_filter(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    filter_fn: Callable[[np.ndarray], np.ndarray],
    alpha: float,
    pad: int = 12,
) -> np.ndarray:
    """Run an expensive filter only inside the mask's bounding box.

    Full-frame bilateral/CLAHE passes dominate per-image latency, yet the result
    is discarded everywhere the mask is zero. Cropping to the region of interest
    first gives the same output for a fraction of the pixels.
    """
    if alpha <= 0:
        return image_rgb
    box = mask_bbox(mask, pad=pad)
    if box is None:
        return image_rgb
    x, y, w, h = box
    roi = np.ascontiguousarray(image_rgb[y : y + h, x : x + w])
    filtered = filter_fn(roi)
    out = image_rgb.copy()
    out[y : y + h, x : x + w] = blend(roi, filtered, mask[y : y + h, x : x + w], alpha)
    return out


def piecewise_warp(image_rgb: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray | None:
    """Warp so that the pixels at ``src`` end up at ``dst``.

    ``skimage.transform.warp`` takes an *inverse* map, so a transform estimated
    from src to dst has to be inverted before it is applied - otherwise the warp
    runs backwards and a "make bigger" edit visibly shrinks the region.

    Returns ``None`` when the point set cannot be triangulated (collinear or
    coincident points). ``estimate`` raises ``QhullError`` from SciPy in that
    case rather than returning ``False``, so both outcomes are handled here and
    callers only ever have to check for ``None``.
    """
    transform = PiecewiseAffineTransform()
    try:
        estimated = transform.estimate(np.asarray(src, np.float32), np.asarray(dst, np.float32))
    except Exception:
        logger.debug("Piecewise-affine estimation failed on a degenerate point set.")
        return None
    if not estimated:
        return None
    warped = warp(
        image_rgb,
        transform.inverse,
        output_shape=image_rgb.shape,
        mode="edge",
        preserve_range=True,
        order=1,
    )
    return np.clip(warped, 0, 255).astype(np.uint8)


def side_by_side(
    left: np.ndarray, right: np.ndarray, gap: int = 12, gap_value: int = 245
) -> np.ndarray:
    """Stitch two images horizontally, matching heights."""
    height = max(left.shape[0], right.shape[0])

    def _pad(img: np.ndarray) -> np.ndarray:
        if img.shape[0] == height:
            return img
        scale = height / img.shape[0]
        return cv2.resize(img, (round(img.shape[1] * scale), height))

    left_r, right_r = _pad(left), _pad(right)
    spacer = np.full((height, gap, 3), gap_value, np.uint8)
    return np.hstack([left_r, spacer, right_r])


def wipe_composite(before: np.ndarray, after: np.ndarray, position: float) -> np.ndarray:
    """Left/right wipe between two same-sized images at ``position`` in [0, 1]."""
    if before.shape != after.shape:
        after = cv2.resize(after, (before.shape[1], before.shape[0]))
    split = int(np.clip(position, 0.0, 1.0) * before.shape[1])
    out = after.copy()
    out[:, :split] = before[:, :split]
    if 0 < split < before.shape[1]:
        out[:, max(0, split - 1) : split + 1] = 255
    return out


def stamp_disclaimer(image_rgb: np.ndarray, text: str) -> np.ndarray:
    """Burn a visible disclaimer banner onto an exported image."""
    out = image_rgb.copy()
    height, width = out.shape[:2]
    scale = max(0.45, min(1.2, width / 1400.0))
    thickness = max(1, round(scale * 2))
    (_, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    band = int(text_h * 2.4)
    overlay = out.copy()
    cv2.rectangle(overlay, (0, height - band), (width, height), (20, 20, 20), -1)
    out = cv2.addWeighted(overlay, 0.72, out, 0.28, 0)
    cv2.putText(
        out,
        text,
        (int(text_h * 0.6), height - int(band * 0.32)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return out
