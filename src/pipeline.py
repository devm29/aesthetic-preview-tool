"""Orchestration: image in, preview and explanation out.

This module is the only place that knows the order of operations. It has no UI
dependency, so the Streamlit app, the CLI and the tests all drive the exact
same code path.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence

import numpy as np

from .config import DISCLAIMER, Settings, get_settings
from .generative import GenerativeEditor, NullEditor, build_editor
from .identity import compute_identity
from .imaging import decode_image, downscale, side_by_side, stamp_disclaimer
from .masks import skin_mask
from .models import (
    AppliedTreatment,
    PreviewResult,
    QCReport,
    TreatmentContext,
)
from .qc import run_qc
from .treatments import Treatment, all_treatments

logger = logging.getLogger(__name__)

EXPORT_BANNER = "AI PREVIEW - ILLUSTRATIVE EDIT, NOT A CLINICAL OUTCOME"


def load_preview_image(data: bytes, settings: Settings | None = None) -> np.ndarray:
    """Decode an upload and shrink it to the preview working size."""
    settings = settings or get_settings()
    image = decode_image(data, max_bytes=settings.max_upload_mb * 1024 * 1024)
    return downscale(image, settings.preview_max_side)


def analyse(image_rgb: np.ndarray, settings: Settings | None = None) -> QCReport:
    return run_qc(image_rgb, settings or get_settings())


def build_context(
    image_rgb: np.ndarray,
    report: QCReport,
    settings: Settings | None = None,
    editor: GenerativeEditor | None = None,
) -> TreatmentContext:
    """Assemble the per-image state every treatment needs."""
    settings = settings or get_settings()
    if report.face is None:
        raise ValueError("Cannot build a treatment context without a detected face.")
    return TreatmentContext(
        original=image_rgb,
        geometry=report.face,
        skin_mask=skin_mask(image_rgb, report.face),
        settings=settings,
        editor=editor if editor is not None else NullEditor(),
    )


def treatment_availability(
    context: TreatmentContext, group: str | None = None
) -> list[tuple[Treatment, bool, str]]:
    """(treatment, usable, reason) for each registered treatment."""
    rows = []
    for treatment in all_treatments(group):
        usable, reason = treatment.availability(context)
        rows.append((treatment, usable, reason))
    return rows


def render_preview(
    image_rgb: np.ndarray,
    report: QCReport,
    intensities: Mapping[str, int] | None = None,
    settings: Settings | None = None,
    editor: GenerativeEditor | None = None,
    context: TreatmentContext | None = None,
) -> PreviewResult:
    """Apply every requested treatment and measure the identity signal.

    Only the treatments named in ``intensities`` run; anything omitted is
    treated as 0. Defaults belong to the caller (the UI's slider positions, the
    CLI's flag defaults) - applying them here would silently edit an image a
    caller asked to leave alone.
    """
    settings = settings or get_settings()
    values: dict[str, int] = {k: int(v) for k, v in (intensities or {}).items()}

    if context is None:
        context = build_context(image_rgb, report, settings, editor)

    started = time.perf_counter()
    edited = image_rgb
    applied: list[AppliedTreatment] = []

    for treatment in all_treatments():
        intensity = values.get(treatment.key, 0)
        if intensity <= 0:
            continue
        usable, reason = treatment.availability(context)
        if not usable:
            context.note(f"{treatment.spec.label}: {reason}")
            continue
        engine = treatment.engine(context)
        step_started = time.perf_counter()
        try:
            edited = treatment.apply(edited, context, intensity)
        except Exception:
            logger.exception("Treatment %s failed; skipping it.", treatment.key)
            context.note(f"{treatment.spec.label}: failed and was skipped.")
            continue
        applied.append(
            AppliedTreatment(
                key=treatment.key,
                label=treatment.spec.label,
                intensity=intensity,
                engine=engine,
                elapsed_ms=(time.perf_counter() - step_started) * 1000.0,
            )
        )

    identity = None
    if report.face is not None:
        identity = compute_identity(image_rgb, edited, report.face.bbox, settings.identity_min_ssim)

    return PreviewResult(
        original=image_rgb,
        edited=edited,
        applied=tuple(applied),
        notes=tuple(context.notes),
        identity=identity,
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
    )


def build_export(
    result: PreviewResult, settings: Settings | None = None, banner: str = EXPORT_BANNER
) -> np.ndarray:
    """Export image: original beside the edit, with a burnt-in disclaimer.

    The original always travels with the edit. An exported "after" on its own
    is exactly the artefact this tool should not produce.
    """
    settings = settings or get_settings()
    pair = side_by_side(result.original, result.edited)
    pair = downscale(pair, settings.export_max_side)
    return stamp_disclaimer(pair, banner)


def describe_result(result: PreviewResult, report: QCReport) -> dict[str, object]:
    """JSON-serialisable summary used by the CLI, the export and the narrator."""
    return {
        "disclaimer": DISCLAIMER,
        "detector": report.face.detector if report.face else None,
        "landmarks": bool(report.face and report.face.has_landmarks),
        "quality_checks": [
            {"name": c.name, "status": c.status, "value": c.value, "detail": c.detail}
            for c in report.checks
        ],
        "treatments": [
            {
                "key": a.key,
                "label": a.label,
                "intensity": a.intensity,
                "engine": a.engine,
                "elapsed_ms": round(a.elapsed_ms, 1),
            }
            for a in result.applied
        ],
        "identity": (
            {
                "ssim": round(result.identity.ssim, 4),
                "threshold": result.identity.threshold,
                "within_threshold": result.identity.within_threshold,
                "verdict": result.identity.verdict,
            }
            if result.identity
            else None
        ),
        "elapsed_ms": round(result.elapsed_ms, 1),
    }


def run(
    data: bytes,
    intensities: Mapping[str, int] | None = None,
    settings: Settings | None = None,
) -> tuple[QCReport, PreviewResult | None]:
    """Convenience end-to-end run over raw image bytes."""
    settings = settings or get_settings()
    image = load_preview_image(data, settings)
    report = analyse(image, settings)
    if report.face is None:
        return report, None
    editor = build_editor(settings)
    result = render_preview(image, report, intensities, settings, editor)
    return report, result


__all__: Sequence[str] = (
    "analyse",
    "build_context",
    "build_export",
    "describe_result",
    "load_preview_image",
    "render_preview",
    "run",
    "treatment_availability",
)
