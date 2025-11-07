"""Domain types shared by the imaging pipeline and the UI.

These are plain dataclasses with no OpenCV, Streamlit or ML imports, so the UI
and the tests can talk about results without touching the pipeline internals.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - import cycle guard, types only
    from .config import Settings
    from .generative import GenerativeEditor

BBox = tuple[int, int, int, int]
CheckStatus = Literal["pass", "fail", "skipped"]


@dataclass(frozen=True)
class FaceGeometry:
    """What a detector managed to work out about the face in an image."""

    bbox: BBox
    detector: str
    landmarks: np.ndarray | None = None
    roll_deg: float | None = None

    @property
    def has_landmarks(self) -> bool:
        return self.landmarks is not None and len(self.landmarks) > 0


@dataclass(frozen=True)
class CheckResult:
    """One quality-control check."""

    name: str
    status: CheckStatus
    detail: str
    value: float | None = None
    unit: str = ""

    @property
    def passed(self) -> bool:
        return self.status == "pass"


@dataclass(frozen=True)
class QCReport:
    """Outcome of the upload quality checks."""

    checks: tuple[CheckResult, ...]
    face: FaceGeometry | None = None

    @property
    def face_present(self) -> bool:
        return self.face is not None

    @property
    def failures(self) -> tuple[CheckResult, ...]:
        return tuple(c for c in self.checks if c.status == "fail")

    @property
    def skipped(self) -> tuple[CheckResult, ...]:
        return tuple(c for c in self.checks if c.status == "skipped")

    @property
    def passed(self) -> bool:
        return self.face_present and not self.failures

    def get(self, name: str) -> CheckResult | None:
        for check in self.checks:
            if check.name == name:
                return check
        return None


@dataclass(frozen=True)
class AppliedTreatment:
    """Record of one treatment actually contributing to a preview."""

    key: str
    label: str
    intensity: int
    engine: str
    elapsed_ms: float


@dataclass(frozen=True)
class IdentityReport:
    """Structural-similarity signal over the face crop.

    SSIM compares image structure. It is a change-magnitude signal, not a
    biometric identity match, and is reported as such everywhere it surfaces.
    """

    ssim: float
    threshold: float

    @property
    def within_threshold(self) -> bool:
        return self.ssim >= self.threshold

    @property
    def verdict(self) -> str:
        if self.ssim >= 0.95:
            return "Barely changed"
        if self.within_threshold:
            return "Visible change, structure retained"
        return "Large structural change"


@dataclass(frozen=True)
class PreviewResult:
    """Everything a caller needs to render and explain one preview."""

    original: np.ndarray
    edited: np.ndarray
    applied: tuple[AppliedTreatment, ...] = ()
    notes: tuple[str, ...] = ()
    identity: IdentityReport | None = None
    elapsed_ms: float = 0.0

    @property
    def changed(self) -> bool:
        return bool(self.applied)


@dataclass
class TreatmentContext:
    """Per-image state handed to every treatment."""

    original: np.ndarray
    geometry: FaceGeometry
    skin_mask: np.ndarray
    settings: Settings
    editor: GenerativeEditor
    notes: list[str] = field(default_factory=list)

    def note(self, message: str) -> None:
        if message not in self.notes:
            self.notes.append(message)


def summarise_checks(checks: Sequence[CheckResult]) -> str:
    """Single-line human summary of a check sequence (used by the CLI)."""
    return ", ".join(f"{c.name}={c.status}" for c in checks)
