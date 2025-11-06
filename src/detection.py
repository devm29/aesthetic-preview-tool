"""Face detection providers.

A detector is anything that turns an RGB array into a :class:`FaceGeometry`.
The app ships two: MediaPipe FaceMesh (468 landmarks, optional install) and an
OpenCV Haar cascade (bounding box only, bundled with opencv-python). Adding a
third - YuNet, RetinaFace, a hosted API - means implementing
``available``/``unavailable_reason``/``detect`` and calling
:func:`register_detector`.
"""

from __future__ import annotations

import logging
import threading
from functools import lru_cache
from typing import Protocol

import cv2
import numpy as np

from .models import BBox, FaceGeometry

logger = logging.getLogger(__name__)

# FaceMesh indices for the outer eye corners, used for the roll estimate.
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
HAAR_CASCADE = "haarcascade_frontalface_default.xml"


class FaceDetector(Protocol):
    """Provider interface for face detection."""

    name: str
    provides_landmarks: bool

    def available(self) -> bool:
        """Whether this detector's dependencies are usable right now."""

    def unavailable_reason(self) -> str:
        """Human-readable explanation shown when ``available`` is false."""

    def detect(self, image_rgb: np.ndarray) -> FaceGeometry | None:
        """Return geometry for the most prominent face, or ``None``."""


def compute_roll_degrees(landmarks: np.ndarray) -> float:
    """Head tilt in degrees from the outer eye corners (0 = level)."""
    left, right = landmarks[LEFT_EYE_OUTER], landmarks[RIGHT_EYE_OUTER]
    return float(np.degrees(np.arctan2(right[1] - left[1], right[0] - left[0])))


def _bbox_from_points(points: np.ndarray, shape: tuple[int, int]) -> BBox:
    height, width = shape
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    x0 = int(np.clip(x_min, 0, width - 1))
    y0 = int(np.clip(y_min, 0, height - 1))
    x1 = int(np.clip(x_max, 0, width - 1))
    y1 = int(np.clip(y_max, 0, height - 1))
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


class MediaPipeDetector:
    """FaceMesh landmarks. Optional dependency; see requirements-landmarks.txt."""

    name = "mediapipe"
    provides_landmarks = True

    def __init__(self) -> None:
        self._mesh = None
        self._lock = threading.Lock()

    @staticmethod
    def _face_mesh_module():
        """Import the legacy ``solutions.face_mesh`` API, or return ``None``.

        ``import mediapipe`` succeeding is not enough. Recent MediaPipe wheels
        on some platforms ship only the ``mediapipe.tasks`` API and drop
        ``mediapipe.solutions`` entirely, so probing the attribute is the only
        reliable check - otherwise the app advertises landmark support and then
        raises ``AttributeError`` on the first upload.
        """
        try:
            import mediapipe as mp

            return mp.solutions.face_mesh
        except Exception:  # pragma: no cover - depends on optional install
            return None

    def available(self) -> bool:
        return self._face_mesh_module() is not None

    def unavailable_reason(self) -> str:
        return (
            "MediaPipe FaceMesh is not importable. Install it with "
            "'pip install -r requirements-landmarks.txt'; a build that only ships "
            "the mediapipe.tasks API will not work."
        )

    def _get_mesh(self):
        """Build the FaceMesh graph once and reuse it.

        Constructing FaceMesh per call costs far more than inference, and the
        UI re-runs detection on every rerun, so the graph is cached.
        """
        if self._mesh is None:
            face_mesh = self._face_mesh_module()
            if face_mesh is None:
                return None
            self._mesh = face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
            )
        return self._mesh

    def detect(self, image_rgb: np.ndarray) -> FaceGeometry | None:
        try:
            mesh = self._get_mesh()
        except Exception:  # pragma: no cover - depends on optional install
            logger.exception("MediaPipe FaceMesh failed to initialise.")
            return None
        if mesh is None:
            return None
        height, width = image_rgb.shape[:2]
        with self._lock:
            results = mesh.process(np.ascontiguousarray(image_rgb))
        if not getattr(results, "multi_face_landmarks", None):
            return None
        mesh_points = results.multi_face_landmarks[0].landmark
        points = np.array([[lm.x * width, lm.y * height] for lm in mesh_points], dtype=np.float32)
        return FaceGeometry(
            bbox=_bbox_from_points(points, (height, width)),
            detector=self.name,
            landmarks=points,
            roll_deg=compute_roll_degrees(points),
        )


class HaarDetector:
    """OpenCV Haar cascade fallback: a bounding box, and nothing else.

    Slim OpenCV builds ship without ``cv2.data`` (and therefore without the
    bundled cascade XML), so ``available`` probes for a cascade that actually
    loads rather than blowing up at detection time.
    """

    name = "haar"
    provides_landmarks = False

    def __init__(self) -> None:
        self._cascade = None

    def available(self) -> bool:
        return self._load() is not None

    def unavailable_reason(self) -> str:
        return (
            "This OpenCV build does not ship the bundled Haar cascade data "
            "(cv2.data). Install opencv-python-headless."
        )

    def _load(self):
        if self._cascade is None:
            if not hasattr(cv2, "CascadeClassifier") or not hasattr(cv2, "data"):
                return None
            cascade = cv2.CascadeClassifier(cv2.data.haarcascades + HAAR_CASCADE)
            if cascade.empty():
                logger.warning("Haar cascade %s could not be loaded.", HAAR_CASCADE)
                return None
            self._cascade = cascade
        return self._cascade

    def detect(self, image_rgb: np.ndarray) -> FaceGeometry | None:
        cascade = self._load()
        if cascade is None:
            return None
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        min_side = max(48, int(min(gray.shape[:2]) * 0.15))
        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_side, min_side)
        )
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda b: int(b[2]) * int(b[3]))
        # No landmarks means no roll estimate; reporting 0.0 would be a lie.
        return FaceGeometry(
            bbox=(int(x), int(y), int(w), int(h)), detector=self.name, roll_deg=None
        )


DETECTOR_ORDER: list[str] = ["mediapipe", "haar"]
_DETECTORS: dict[str, FaceDetector] = {
    "mediapipe": MediaPipeDetector(),
    "haar": HaarDetector(),
}


def register_detector(detector: FaceDetector, first: bool = True) -> None:
    """Add a detector provider at runtime (used by tests and integrations)."""
    _DETECTORS[detector.name] = detector
    if detector.name in DETECTOR_ORDER:
        DETECTOR_ORDER.remove(detector.name)
    DETECTOR_ORDER.insert(0 if first else len(DETECTOR_ORDER), detector.name)
    resolve_detector.cache_clear()


def available_detectors() -> list[str]:
    return [name for name in DETECTOR_ORDER if _DETECTORS[name].available()]


@lru_cache(maxsize=4)
def resolve_detector(preference: str = "auto") -> FaceDetector | None:
    """Pick the best available detector, honouring an explicit preference."""
    if preference not in ("auto", ""):
        detector = _DETECTORS.get(preference)
        if detector is not None and detector.available():
            return detector
        logger.warning("Requested detector %r unavailable; falling back.", preference)
    for name in DETECTOR_ORDER:
        detector = _DETECTORS[name]
        if detector.available():
            return detector
    return None


def detect_face(image_rgb: np.ndarray, preference: str = "auto") -> FaceGeometry | None:
    detector = resolve_detector(preference)
    if detector is None:
        logger.error("No face detector is available in this environment.")
        return None
    return detector.detect(image_rgb)
