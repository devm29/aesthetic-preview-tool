"""Tests for the face-detector provider layer.

No test here requires MediaPipe. The provider seam is exercised with a fake
detector so the behaviour is asserted on every machine.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import detection
from src.models import FaceGeometry

from .conftest import make_mesh


class FakeDetector:
    """A detector provider that returns a fixed geometry."""

    provides_landmarks = True

    def __init__(self, name: str, available: bool = True, geometry=None) -> None:
        self.name = name
        self._available = available
        self._geometry = geometry
        self.calls = 0

    def available(self) -> bool:
        return self._available

    def unavailable_reason(self) -> str:
        return f"{self.name} is switched off."

    def detect(self, image_rgb: np.ndarray) -> FaceGeometry | None:
        self.calls += 1
        return self._geometry


@pytest.fixture
def clean_registry():
    """Snapshot and restore the module-level detector registry."""
    order = list(detection.DETECTOR_ORDER)
    registry = dict(detection._DETECTORS)
    detection.resolve_detector.cache_clear()
    yield
    detection.DETECTOR_ORDER[:] = order
    detection._DETECTORS.clear()
    detection._DETECTORS.update(registry)
    detection.resolve_detector.cache_clear()


def test_compute_roll_degrees_is_zero_for_level_eyes():
    mesh = make_mesh(roll_deg=0.0)
    assert detection.compute_roll_degrees(mesh) == pytest.approx(0.0, abs=0.01)


@pytest.mark.parametrize("roll", [-25.0, -8.0, 12.0, 30.0])
def test_compute_roll_degrees_recovers_the_synthesised_tilt(roll):
    mesh = make_mesh(roll_deg=roll)
    assert detection.compute_roll_degrees(mesh) == pytest.approx(roll, abs=0.5)


def test_bbox_from_points_is_clipped_to_the_image():
    points = np.array([[-40.0, -40.0], [500.0, 500.0]], np.float32)
    x, y, w, h = detection._bbox_from_points(points, (100, 120))
    assert (x, y) == (0, 0)
    assert x + w <= 120 and y + h <= 100


def test_register_detector_puts_the_new_provider_first(clean_registry):
    geometry = FaceGeometry(bbox=(1, 2, 3, 4), detector="fake")
    fake = FakeDetector("fake", geometry=geometry)
    detection.register_detector(fake)
    assert detection.DETECTOR_ORDER[0] == "fake"
    assert detection.detect_face(np.zeros((32, 32, 3), np.uint8)) is geometry
    assert fake.calls == 1


def test_register_detector_can_append_instead(clean_registry):
    detection.register_detector(FakeDetector("last"), first=False)
    assert detection.DETECTOR_ORDER[-1] == "last"


def test_registering_the_same_name_twice_does_not_duplicate_it(clean_registry):
    detection.register_detector(FakeDetector("dup"))
    detection.register_detector(FakeDetector("dup"))
    assert detection.DETECTOR_ORDER.count("dup") == 1


def test_resolve_detector_skips_unavailable_providers(clean_registry):
    detection.register_detector(FakeDetector("usable"), first=True)
    detection.register_detector(FakeDetector("broken", available=False), first=True)
    assert detection.resolve_detector("auto").name == "usable"


def test_resolve_detector_honours_an_explicit_preference(clean_registry):
    detection.register_detector(FakeDetector("first"), first=True)
    detection.register_detector(FakeDetector("wanted"), first=False)
    assert detection.resolve_detector("wanted").name == "wanted"


def test_resolve_detector_falls_back_when_the_preference_is_unavailable(clean_registry, caplog):
    detection.register_detector(FakeDetector("usable"), first=True)
    detection.register_detector(FakeDetector("gone", available=False), first=False)
    with caplog.at_level("WARNING"):
        assert detection.resolve_detector("gone").name == "usable"
    assert "unavailable" in caplog.text


def test_available_detectors_lists_only_usable_providers(clean_registry):
    detection.register_detector(FakeDetector("yes"), first=True)
    detection.register_detector(FakeDetector("no", available=False), first=True)
    listed = detection.available_detectors()
    assert "yes" in listed
    assert "no" not in listed


def test_detect_face_returns_none_when_nothing_is_available(clean_registry, caplog):
    detection._DETECTORS.clear()
    detection.DETECTOR_ORDER[:] = []
    detection.resolve_detector.cache_clear()
    with caplog.at_level("ERROR"):
        assert detection.detect_face(np.zeros((32, 32, 3), np.uint8)) is None
    assert "No face detector" in caplog.text


def test_mediapipe_availability_probes_the_face_mesh_api(monkeypatch):
    """Regression: importing ``mediapipe`` is not enough.

    Builds that ship only ``mediapipe.tasks`` import fine but have no
    ``solutions`` attribute. Reporting those as available made the app raise
    AttributeError on the first upload instead of falling back to Haar.
    """
    detector = detection.MediaPipeDetector()
    monkeypatch.setattr(detector, "_face_mesh_module", staticmethod(lambda: None))
    assert detector.available() is False
    assert detector.detect(np.zeros((32, 32, 3), np.uint8)) is None
    assert "mediapipe.tasks" in detector.unavailable_reason()


def test_haar_detector_reports_a_reason_when_opencv_has_no_cascade_data(monkeypatch):
    detector = detection.HaarDetector()
    monkeypatch.delattr(detection.cv2, "data", raising=False)
    assert detector.available() is False
    assert detector.detect(np.zeros((32, 32, 3), np.uint8)) is None
    assert "cv2.data" in detector.unavailable_reason()


def test_haar_detector_finds_the_sample_face(sample_image):
    """The README's headline claim: the app works with no ML deps installed."""
    detector = detection.HaarDetector()
    if not detector.available():  # pragma: no cover - slim OpenCV builds only
        pytest.skip("This OpenCV build ships no Haar cascade data.")
    geometry = detector.detect(sample_image)
    assert geometry is not None
    assert geometry.detector == "haar"
    assert geometry.has_landmarks is False
    # No landmarks means no roll estimate. Reporting 0.0 would be a fabrication.
    assert geometry.roll_deg is None


def test_haar_detector_returns_none_on_a_blank_image():
    detector = detection.HaarDetector()
    if not detector.available():  # pragma: no cover - slim OpenCV builds only
        pytest.skip("This OpenCV build ships no Haar cascade data.")
    assert detector.detect(np.zeros((256, 256, 3), np.uint8)) is None
