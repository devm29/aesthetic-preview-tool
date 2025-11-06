"""Runtime configuration.

Every tunable lives here and is resolved from the environment once, so the rest
of the code never reads ``os.environ`` directly and tests can build a
``Settings`` instance by hand.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

_TRUTHY = {"1", "true", "yes", "on"}


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value.strip() == "" else value.strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env_str(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env_str(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    return _env_str(name, "true" if default else "false").lower() in _TRUTHY


@dataclass(frozen=True)
class QCThresholds:
    """Acceptance thresholds for the upload quality checks."""

    blur_target_width: int = 512
    blur_variance_min: float = 100.0
    lighting_mean_min: float = 90.0
    lighting_mean_max: float = 200.0
    lighting_std_min: float = 25.0
    overexposed_max_frac: float = 0.15
    underexposed_max_frac: float = 0.15
    roll_max_abs_deg: float = 20.0


@dataclass(frozen=True)
class Settings:
    """Immutable application settings."""

    preview_max_side: int = 1024
    export_max_side: int = 2048
    max_upload_mb: int = 12
    identity_min_ssim: float = 0.80
    detector_preference: str = "auto"
    enable_generative: bool = False
    sdxl_model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    sdxl_max_side: int = 768
    narrator_model: str = "claude-opus-5"
    narrator_enabled: bool = True
    log_level: str = "INFO"
    qc: QCThresholds = field(default_factory=QCThresholds)

    @property
    def sample_image_path(self) -> Path:
        return PROJECT_ROOT / "assets" / "sample_face.jpg"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Resolve settings from the environment (cached for the process lifetime)."""
    return Settings(
        preview_max_side=_env_int("DERMA_PREVIEW_MAX_SIDE", 1024),
        export_max_side=_env_int("DERMA_EXPORT_MAX_SIDE", 2048),
        max_upload_mb=_env_int("DERMA_MAX_UPLOAD_MB", 12),
        identity_min_ssim=_env_float("DERMA_IDENTITY_MIN_SSIM", 0.80),
        detector_preference=_env_str("DERMA_FACE_DETECTOR", "auto").lower(),
        enable_generative=_env_bool("DERMA_ENABLE_GENERATIVE", False),
        sdxl_model_id=_env_str(
            "DERMA_SDXL_MODEL_ID", "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
        ),
        sdxl_max_side=_env_int("DERMA_SDXL_MAX_SIDE", 768),
        narrator_model=_env_str("DERMA_NARRATOR_MODEL", "claude-opus-5"),
        narrator_enabled=_env_bool("DERMA_NARRATOR_ENABLED", True),
        log_level=_env_str("DERMA_LOG_LEVEL", "INFO").upper(),
    )


DISCLAIMER = (
    "This is a proof of concept, not a medical device. Previews are illustrative "
    "image edits, not a prediction or guarantee of any clinical outcome."
)
