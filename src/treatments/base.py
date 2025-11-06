"""Treatment interface and registry.

A treatment is one adjustable edit. Adding one means writing a class with a
:class:`TreatmentSpec` and an ``apply`` method, decorating it with
``@register``, and importing it in ``src/treatments/__init__.py``. The UI, the
CLI and the AI narrator all read the registry, so no other file changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..models import TreatmentContext

ADJUSTMENT = "adjustment"
PROCEDURE = "procedure"


@dataclass(frozen=True)
class TreatmentSpec:
    """Everything the UI needs to render a control for a treatment."""

    key: str
    label: str
    summary: str
    group: str = ADJUSTMENT
    order: int = 100
    default: int = 0
    step: int = 5
    max_intensity: int = 100
    requires_landmarks: bool = False
    supports_generative: bool = False
    prompt: str = ""
    negative_prompt: str = ""


class Treatment(ABC):
    """Base class for every edit the app can apply."""

    spec: TreatmentSpec

    @property
    def key(self) -> str:
        return self.spec.key

    def availability(self, context: TreatmentContext) -> tuple[bool, str]:
        """Whether this treatment can run against the given face."""
        if self.spec.requires_landmarks and not context.geometry.has_landmarks:
            return False, (
                "Needs facial landmarks. Install the optional landmark model "
                "(requirements-landmarks.txt) for this treatment."
            )
        return True, ""

    def engine(self, context: TreatmentContext) -> str:
        """Which implementation path a call would take, for reporting."""
        if self.spec.supports_generative and context.editor.available():
            return "generative"
        return "parametric"

    @abstractmethod
    def apply(self, image_rgb: np.ndarray, context: TreatmentContext, intensity: int) -> np.ndarray:
        """Return an edited copy. Must be a no-op when ``intensity`` is 0."""


_REGISTRY: dict[str, Treatment] = {}


def register(cls: type[Treatment]) -> type[Treatment]:
    """Class decorator that instantiates and registers a treatment."""
    instance = cls()
    if instance.spec.key in _REGISTRY:
        raise ValueError(f"Duplicate treatment key: {instance.spec.key}")
    _REGISTRY[instance.spec.key] = instance
    return cls


def unregister(key: str) -> None:
    """Remove a treatment (used by tests)."""
    _REGISTRY.pop(key, None)


def get_treatment(key: str) -> Treatment | None:
    return _REGISTRY.get(key)


def all_treatments(group: str | None = None) -> list[Treatment]:
    """Registered treatments in deterministic application order."""
    items = [t for t in _REGISTRY.values() if group is None or t.spec.group == group]
    return sorted(items, key=lambda t: (t.spec.order, t.spec.key))


def default_intensities() -> dict[str, int]:
    return {t.spec.key: t.spec.default for t in all_treatments()}
