"""Render the synthetic sample portrait shipped in ``assets/sample_face.jpg``.

The sample is generated from this script - a few hundred OpenCV primitives and
a fixed RNG seed. It is deliberately **not** a photograph of a real person, and
nobody's likeness is committed to this repository. Re-run to regenerate:

    python tools/make_sample_face.py

It is tuned to pass the app's quality checks: enough grain to clear the
sharpness threshold, even lighting, and a frontal layout the Haar cascade
detects without the optional landmark model.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

SEED = 20260923
SIZE = 768
OUTPUT = Path(__file__).resolve().parent.parent / "assets" / "sample_face.jpg"
JPEG_QUALITY = 92

RNG = np.random.default_rng(SEED)


def _radial(cy: float, cx: float, ry: float, rx: float) -> np.ndarray:
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    return np.sqrt(((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2)


def build() -> np.ndarray:
    """Return the RGB sample image."""
    img = np.zeros((SIZE, SIZE, 3), np.float32)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)

    backdrop = 150 - 40 * ((yy / SIZE) ** 1.3) - 18 * np.abs(xx / SIZE - 0.5)
    img[:] = np.dstack([backdrop * 0.92, backdrop * 0.95, backdrop])

    skin = np.array([214.0, 176.0, 152.0], np.float32)
    neck = _radial(SIZE * 0.93, SIZE * 0.5, SIZE * 0.30, SIZE * 0.17) < 1.0
    shoulders = _radial(SIZE * 1.22, SIZE * 0.5, SIZE * 0.42, SIZE * 0.62) < 1.0
    for region, shade in ((shoulders, 0.72), (neck, 0.82)):
        img[region] = skin * shade

    face_cy, face_cx = SIZE * 0.47, SIZE * 0.5
    face_ry, face_rx = SIZE * 0.30, SIZE * 0.215
    radial_face = _radial(face_cy, face_cx, face_ry, face_rx)
    face = radial_face < 1.0
    sphere = np.clip(1.10 - 0.42 * radial_face**2, 0.55, 1.12)
    key_light = (
        1.0 + 0.10 * ((SIZE * 0.5 - xx) / (SIZE * 0.5)) + 0.08 * ((SIZE * 0.5 - yy) / (SIZE * 0.5))
    )
    tone = np.clip(sphere * key_light, 0.5, 1.25)[..., None]
    img[face] = (skin * tone)[face]

    hair = _radial(face_cy - face_ry * 0.78, face_cx, face_ry * 0.72, face_rx * 1.34) < 1.0
    hair &= yy < face_cy - face_ry * 0.52
    hair |= (
        (_radial(face_cy - face_ry * 0.12, face_cx, face_ry * 1.02, face_rx * 1.30) < 1.0)
        & (radial_face > 1.00)
        & (yy < face_cy + face_ry * 0.30)
    )
    img[hair] = (np.array([70.0, 52.0, 44.0], np.float32) * np.clip(tone, 0.8, 1.15))[hair]

    def ellipse(cy, cx, ry, rx, colour, alpha=1.0, angle=0):
        layer = np.zeros((SIZE, SIZE), np.uint8)
        cv2.ellipse(layer, (int(cx), int(cy)), (int(rx), int(ry)), angle, 0, 360, 255, -1)
        weight = (layer.astype(np.float32) / 255.0 * alpha)[..., None]
        img[:] = img * (1 - weight) + np.asarray(colour, np.float32) * weight

    eye_y = face_cy - face_ry * 0.10
    eye_dx = face_rx * 0.42
    eye_rx, eye_ry = face_rx * 0.235, face_ry * 0.082

    for side in (-1, 1):
        ellipse(
            eye_y - face_ry * 0.165,
            face_cx + side * eye_dx,
            face_ry * 0.034,
            face_rx * 0.30,
            (72, 54, 46),
            0.95,
            angle=-7 * side,
        )
    for side in (-1, 1):
        cx = face_cx + side * eye_dx
        ellipse(eye_y, cx, eye_ry * 1.9, eye_rx * 1.35, (150, 118, 103), 0.30)
        ellipse(eye_y, cx, eye_ry, eye_rx, (238, 236, 233))
        ellipse(eye_y, cx, eye_ry * 0.94, eye_ry * 0.94, (96, 112, 120))
        ellipse(eye_y, cx, eye_ry * 0.42, eye_ry * 0.42, (24, 22, 24))
        ellipse(eye_y - eye_ry * 0.85, cx, eye_ry * 0.30, eye_rx, (55, 42, 38), 0.85)
        # Under-eye shadow, so the under-eye treatment has something to act on.
        ellipse(eye_y + eye_ry * 2.5, cx, eye_ry * 1.5, eye_rx * 1.05, (150, 112, 112), 0.42)

    ellipse(
        face_cy + face_ry * 0.10,
        face_cx,
        face_ry * 0.20,
        face_rx * 0.075,
        (236, 200, 175),
        0.35,
    )
    ellipse(
        face_cy + face_ry * 0.30,
        face_cx,
        face_ry * 0.055,
        face_rx * 0.155,
        (168, 130, 112),
        0.55,
    )
    for side in (-1, 1):
        ellipse(
            face_cy + face_ry * 0.305,
            face_cx + side * face_rx * 0.105,
            face_ry * 0.024,
            face_rx * 0.036,
            (96, 68, 60),
            0.75,
        )

    lip_y = face_cy + face_ry * 0.545
    ellipse(
        lip_y - face_ry * 0.020,
        face_cx,
        face_ry * 0.052,
        face_rx * 0.30,
        (176, 106, 104),
        0.92,
    )
    ellipse(
        lip_y + face_ry * 0.038,
        face_cx,
        face_ry * 0.058,
        face_rx * 0.275,
        (188, 116, 112),
        0.92,
    )
    ellipse(lip_y + face_ry * 0.008, face_cx, face_ry * 0.008, face_rx * 0.27, (120, 70, 70), 0.85)
    ellipse(
        face_cy + face_ry * 0.76,
        face_cx,
        face_ry * 0.045,
        face_rx * 0.16,
        (198, 158, 138),
        0.28,
    )

    for side in (-1, 1):
        ellipse(
            face_cy + face_ry * 0.22,
            face_cx + side * face_rx * 0.58,
            face_ry * 0.16,
            face_rx * 0.22,
            (208, 150, 140),
            0.26,
        )

    lines = img.copy()
    for i in range(3):
        y = int(face_cy - face_ry * (0.46 - 0.085 * i))
        cv2.ellipse(
            lines,
            (int(face_cx), y + 40),
            (int(face_rx * 0.52), 34),
            0,
            200,
            340,
            (176, 138, 122),
            2,
            cv2.LINE_AA,
        )
    for side in (-1, 1):
        for i in range(3):
            x = int(face_cx + side * (eye_dx + eye_rx * 1.15))
            y = int(eye_y + (i - 1) * 7)
            cv2.line(
                lines,
                (x, y),
                (int(x + side * 20), int(y + (i - 1) * 6)),
                (170, 130, 118),
                2,
                cv2.LINE_AA,
            )
    img[:] = cv2.GaussianBlur(lines, (0, 0), 1.2)

    spots = np.zeros((SIZE, SIZE), np.float32)
    for _ in range(90):
        angle = RNG.uniform(0, 2 * np.pi)
        radius = np.sqrt(RNG.uniform(0, 1)) * 0.92
        cy = face_cy + np.sin(angle) * radius * face_ry
        cx = face_cx + np.cos(angle) * radius * face_rx
        cv2.circle(
            spots, (int(cx), int(cy)), int(RNG.integers(2, 6)), float(RNG.uniform(0.10, 0.34)), -1
        )
    spots = cv2.GaussianBlur(spots, (0, 0), 2.4) * face
    weight = (spots * 0.55)[..., None]
    img[:] = img * (1 - weight) + np.array([150.0, 110.0, 96.0]) * weight

    fine = RNG.normal(0, 9.0, (SIZE, SIZE, 1)).astype(np.float32)
    coarse = cv2.GaussianBlur(RNG.normal(0, 11.0, (SIZE, SIZE)).astype(np.float32), (0, 0), 1.6)
    img += (fine + coarse[..., None]) * (0.30 + 0.70 * face[..., None])
    return np.clip(img, 0, 255).astype(np.uint8)


def main() -> None:
    image = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(OUTPUT),
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
    )
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
