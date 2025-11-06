"""Shared fixtures.

Everything here is synthetic or generated. No test downloads a model, needs a
GPU, reaches the network, or touches a photograph of a real person.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.config import QCThresholds, Settings
from src.generative import NullEditor
from src.models import FaceGeometry, QCReport, TreatmentContext
from src.qc import run_qc

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PATH = PROJECT_ROOT / "assets" / "sample_face.jpg"

# FaceMesh ships 468 points (478 with refine_landmarks). Anything the mask code
# treats as "a real mesh" has to clear MIN_MESH_POINTS.
MESH_POINTS = 478


@pytest.fixture(scope="session")
def settings() -> Settings:
    """Deterministic settings that never enable an optional dependency."""
    return Settings(
        preview_max_side=512,
        export_max_side=1024,
        max_upload_mb=4,
        identity_min_ssim=0.80,
        detector_preference="haar",
        enable_generative=False,
        narrator_enabled=False,
        qc=QCThresholds(),
    )


@pytest.fixture(scope="session")
def sample_bytes() -> bytes:
    """Raw bytes of the synthetic sample portrait shipped in the repo."""
    if not SAMPLE_PATH.is_file():  # pragma: no cover - only if the asset is missing
        pytest.skip(f"{SAMPLE_PATH} is missing; run python tools/make_sample_face.py")
    return SAMPLE_PATH.read_bytes()


@pytest.fixture(scope="session")
def sample_image(sample_bytes: bytes) -> np.ndarray:
    from src.imaging import decode_image, downscale

    return downscale(decode_image(sample_bytes), 512)


@pytest.fixture(scope="session")
def sample_report(sample_image: np.ndarray, settings: Settings) -> QCReport:
    report = run_qc(sample_image, settings)
    if report.face is None:  # pragma: no cover - would mean the Haar path broke
        pytest.skip("No face detector is available in this environment.")
    return report


def make_mesh(
    width: int = 256,
    height: int = 256,
    roll_deg: float = 0.0,
    points: int = MESH_POINTS,
) -> np.ndarray:
    """A synthetic FaceMesh-shaped landmark array.

    Real index semantics are honoured for the handful of indices the code uses
    (outer eye corners, the lip rings, the eye rings) so the masks come out
    non-degenerate; every other point sits on a face-sized ellipse so the
    convex hull is a plausible face outline.
    """
    from src.masks import LEFT_EYE_IDX, LIP_INNER_IDX, LIP_OUTER_IDX, RIGHT_EYE_IDX

    cx, cy = width / 2.0, height / 2.0
    rx, ry = width * 0.28, height * 0.38

    angles = np.linspace(0.0, 2.0 * np.pi, points, endpoint=False)
    mesh = np.stack([cx + rx * np.cos(angles), cy + ry * np.sin(angles)], axis=1)

    def place(indices, centre_x, centre_y, radius_x, radius_y):
        count = len(indices)
        theta = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
        mesh[list(indices), 0] = centre_x + radius_x * np.cos(theta)
        mesh[list(indices), 1] = centre_y + radius_y * np.sin(theta)

    place(LIP_OUTER_IDX, cx, cy + ry * 0.55, rx * 0.40, ry * 0.15)
    place(LIP_INNER_IDX, cx, cy + ry * 0.55, rx * 0.24, ry * 0.07)
    place(LEFT_EYE_IDX, cx - rx * 0.42, cy - ry * 0.18, rx * 0.16, ry * 0.07)
    place(RIGHT_EYE_IDX, cx + rx * 0.42, cy - ry * 0.18, rx * 0.16, ry * 0.07)

    # Outer eye corners drive the roll estimate (indices 33 and 263).
    angle = np.radians(roll_deg)
    span = rx * 0.55
    eye_y = cy - ry * 0.18
    mesh[33] = [cx - span * np.cos(angle), eye_y - span * np.sin(angle)]
    mesh[263] = [cx + span * np.cos(angle), eye_y + span * np.sin(angle)]
    return mesh.astype(np.float32)


def make_geometry(
    width: int = 256,
    height: int = 256,
    with_landmarks: bool = True,
    roll_deg: float = 0.0,
) -> FaceGeometry:
    if not with_landmarks:
        return FaceGeometry(
            bbox=(int(width * 0.22), int(height * 0.12), int(width * 0.56), int(height * 0.76)),
            detector="fake-bbox",
            roll_deg=None,
        )
    mesh = make_mesh(width, height, roll_deg)
    x0, y0 = mesh.min(axis=0)
    x1, y1 = mesh.max(axis=0)
    return FaceGeometry(
        bbox=(int(x0), int(y0), max(1, int(x1 - x0)), max(1, int(y1 - y0))),
        detector="fake-mesh",
        landmarks=mesh,
        roll_deg=roll_deg,
    )


def make_face_image(width: int = 256, height: int = 256, seed: int = 7) -> np.ndarray:
    """A textured, face-toned image. Texture matters: a flat image makes every
    smoothing filter a no-op and the assertions vacuous."""
    rng = np.random.default_rng(seed)
    base = np.zeros((height, width, 3), np.float32)
    base[:, :] = (205.0, 168.0, 148.0)
    base += rng.normal(0.0, 22.0, (height, width, 3))
    return np.clip(base, 0, 255).astype(np.uint8)


@pytest.fixture
def face_image() -> np.ndarray:
    return make_face_image()


@pytest.fixture
def mesh_geometry() -> FaceGeometry:
    return make_geometry(with_landmarks=True)


@pytest.fixture
def bbox_geometry() -> FaceGeometry:
    return make_geometry(with_landmarks=False)


def make_context(
    image: np.ndarray,
    geometry: FaceGeometry,
    settings: Settings,
    editor: object | None = None,
) -> TreatmentContext:
    from src.masks import skin_mask

    return TreatmentContext(
        original=image,
        geometry=geometry,
        skin_mask=skin_mask(image, geometry),
        settings=settings,
        editor=editor if editor is not None else NullEditor(),
    )


@pytest.fixture
def mesh_context(face_image, mesh_geometry, settings) -> TreatmentContext:
    return make_context(face_image, mesh_geometry, settings)


@pytest.fixture
def bbox_context(face_image, bbox_geometry, settings) -> TreatmentContext:
    return make_context(face_image, bbox_geometry, settings)


class RecordingEditor:
    """Generative editor stand-in. Paints the mask solid magenta.

    It is deliberately not a real model: the tests assert on dispatch and
    compositing, which is everything the pipeline is responsible for.
    """

    name = "recording"

    def __init__(self, available: bool = True, fail: bool = False) -> None:
        self._available = available
        self._fail = fail
        self.calls = []

    def available(self) -> bool:
        return self._available

    def unavailable_reason(self) -> str:
        return "Recording editor is switched off."

    def inpaint(self, image_rgb, mask, prompt, negative_prompt="", **kwargs):
        self.calls.append({"prompt": prompt, "negative_prompt": negative_prompt, **kwargs})
        if self._fail:
            raise RuntimeError("simulated generative failure")
        out = image_rgb.copy()
        out[mask > 0] = (255, 0, 255)
        return out


@pytest.fixture
def recording_editor() -> RecordingEditor:
    return RecordingEditor()
