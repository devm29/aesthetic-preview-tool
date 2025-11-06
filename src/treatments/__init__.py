"""Treatment registry and the built-in treatments.

Importing this package registers every built-in treatment. Third-party
treatments only need to import ``register`` and decorate a class.
"""

from . import parametric as _parametric  # noqa: F401  (registers adjustments)
from . import procedures as _procedures  # noqa: F401  (registers procedures)
from .base import (
    ADJUSTMENT,
    PROCEDURE,
    Treatment,
    TreatmentSpec,
    all_treatments,
    default_intensities,
    get_treatment,
    register,
    unregister,
)

__all__ = [
    "ADJUSTMENT",
    "PROCEDURE",
    "Treatment",
    "TreatmentSpec",
    "all_treatments",
    "default_intensities",
    "get_treatment",
    "register",
    "unregister",
]
