"""Tests for the orchestration layer.

The pipeline is the seam that makes the app testable without a browser: these
run the exact code path the Streamlit UI and the CLI use.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import pipeline
from src.config import DISCLAIMER, Settings
from src.imaging import ImageTooLargeError, encode_png
from src.models import PreviewResult, QCReport
from src.treatments import PROCEDURE, Treatment, TreatmentSpec, all_treatments, register, unregister

from .conftest import RecordingEditor, make_context, make_geometry

# ------------------------------------------------------------------- ingestion


def test_load_preview_image_downscales_to_the_working_size(settings):
    payload = encode_png(np.zeros((2000, 2000, 3), np.uint8))
    image = pipeline.load_preview_image(payload, settings)
    assert max(image.shape[:2]) == settings.preview_max_side


def test_load_preview_image_enforces_the_upload_budget():
    tiny_budget = Settings(max_upload_mb=0)
    with pytest.raises(ImageTooLargeError):
        pipeline.load_preview_image(encode_png(np.zeros((64, 64, 3), np.uint8)), tiny_budget)


# ----------------------------------------------------------------- the context


def test_build_context_refuses_to_run_without_a_detected_face(face_image, settings):
    empty = QCReport(checks=(), face=None)
    with pytest.raises(ValueError, match="without a detected face"):
        pipeline.build_context(face_image, empty, settings)


def test_build_context_defaults_to_a_null_editor(face_image, settings):
    report = QCReport(checks=(), face=make_geometry())
    context = pipeline.build_context(face_image, report, settings)
    assert context.editor.available() is False
    assert context.skin_mask.shape == face_image.shape[:2]


def test_treatment_availability_explains_every_unusable_treatment(bbox_context):
    rows = pipeline.treatment_availability(bbox_context)
    assert len(rows) == len(all_treatments())
    for treatment, usable, reason in rows:
        assert usable == (not treatment.spec.requires_landmarks)
        assert (reason == "") == usable


def test_treatment_availability_can_be_filtered_by_group(mesh_context):
    rows = pipeline.treatment_availability(mesh_context, PROCEDURE)
    assert {t.key for t, _, _ in rows} == {"lip", "under_eye"}


# ------------------------------------------------------------------- rendering


def test_render_preview_with_no_intensities_leaves_the_image_untouched(
    face_image, settings, mesh_geometry
):
    report = QCReport(checks=(), face=mesh_geometry)
    result = pipeline.render_preview(
        face_image, report, {k.key: 0 for k in all_treatments()}, settings
    )
    assert result.applied == ()
    assert result.changed is False
    assert np.array_equal(result.edited, face_image)
    assert result.identity.ssim == pytest.approx(1.0, abs=1e-6)


def test_render_preview_records_what_it_applied(face_image, settings, mesh_geometry):
    report = QCReport(checks=(), face=mesh_geometry)
    result = pipeline.render_preview(face_image, report, {"wrinkle": 60, "tone": 30}, settings)

    keys = [a.key for a in result.applied]
    assert keys == ["wrinkle", "tone"]
    assert all(a.engine == "parametric" for a in result.applied)
    assert all(a.elapsed_ms >= 0 for a in result.applied)
    assert result.elapsed_ms >= 0
    assert result.changed is True


def test_render_preview_keeps_the_original_alongside_the_edit(face_image, settings, mesh_geometry):
    """The original must always travel with the edit."""
    report = QCReport(checks=(), face=mesh_geometry)
    result = pipeline.render_preview(face_image, report, {"wrinkle": 90}, settings)
    assert np.array_equal(result.original, face_image)
    assert not np.array_equal(result.edited, face_image)


def test_render_preview_applies_treatments_in_registry_order(face_image, settings, mesh_geometry):
    report = QCReport(checks=(), face=mesh_geometry)
    result = pipeline.render_preview(
        face_image, report, {"pigment": 40, "wrinkle": 40, "tone": 40}, settings
    )
    expected = [t.key for t in all_treatments() if t.key in {"pigment", "wrinkle", "tone"}]
    assert [a.key for a in result.applied] == expected


def test_render_preview_skips_an_unavailable_treatment_and_notes_why(
    face_image, settings, bbox_geometry
):
    report = QCReport(checks=(), face=bbox_geometry)
    result = pipeline.render_preview(face_image, report, {"lip": 80}, settings)
    assert result.applied == ()
    assert any("Lip volume" in note for note in result.notes)


def test_a_treatment_that_raises_is_skipped_rather_than_crashing_the_app(
    face_image, settings, mesh_geometry, caplog
):
    @register
    class Exploding(Treatment):
        spec = TreatmentSpec(key="boom", label="Exploding", summary="Always fails.", order=1)

        def apply(self, image_rgb, context, intensity):
            raise RuntimeError("kaboom")

    try:
        report = QCReport(checks=(), face=mesh_geometry)
        with caplog.at_level("ERROR"):
            result = pipeline.render_preview(
                face_image, report, {"boom": 50, "wrinkle": 40}, settings
            )
        assert [a.key for a in result.applied] == ["wrinkle"]
        assert any("failed and was skipped" in note for note in result.notes)
        assert "kaboom" in caplog.text
    finally:
        unregister("boom")


def test_render_preview_reuses_a_supplied_context(face_image, settings, mesh_geometry):
    context = make_context(face_image, mesh_geometry, settings)
    report = QCReport(checks=(), face=mesh_geometry)
    result = pipeline.render_preview(face_image, report, {"wrinkle": 30}, settings, context=context)
    assert result.applied[0].key == "wrinkle"
    assert context.skin_mask.shape == face_image.shape[:2]


def test_render_preview_reports_the_generative_engine_when_one_is_used(
    face_image, settings, mesh_geometry
):
    editor = RecordingEditor()
    context = make_context(face_image, mesh_geometry, settings, editor)
    report = QCReport(checks=(), face=mesh_geometry)
    result = pipeline.render_preview(
        face_image, report, {"lip": 50}, settings, editor, context=context
    )
    assert result.applied[0].engine == "generative"
    assert len(editor.calls) == 1


def test_identity_is_measured_against_the_original(face_image, settings, mesh_geometry):
    report = QCReport(checks=(), face=mesh_geometry)
    gentle = pipeline.render_preview(face_image, report, {"wrinkle": 10}, settings)
    strong = pipeline.render_preview(face_image, report, {"wrinkle": 100}, settings)
    assert gentle.identity.ssim > strong.identity.ssim
    assert gentle.identity.threshold == settings.identity_min_ssim


# ---------------------------------------------------------------------- export


def test_build_export_puts_the_original_next_to_the_edit(face_image, settings, mesh_geometry):
    report = QCReport(checks=(), face=mesh_geometry)
    result = pipeline.render_preview(face_image, report, {"wrinkle": 80}, settings)
    export = pipeline.build_export(result, settings)

    assert export.shape[1] > face_image.shape[1] * 1.9, "export is not a before/after pair"
    assert max(export.shape[:2]) <= settings.export_max_side
    # The disclaimer band is burnt into the bottom of the export.
    assert export[-4:].mean() < export[: export.shape[0] // 2].mean()


def test_build_export_banner_text_is_configurable(face_image, settings, mesh_geometry):
    report = QCReport(checks=(), face=mesh_geometry)
    result = pipeline.render_preview(face_image, report, {}, settings)
    assert pipeline.build_export(result, settings, banner="X").shape[2] == 3


def test_export_banner_names_the_preview_as_illustrative():
    assert "NOT A CLINICAL OUTCOME" in pipeline.EXPORT_BANNER


# --------------------------------------------------------------------- summary


def test_describe_result_is_json_serialisable_and_carries_the_disclaimer(
    face_image, settings, mesh_geometry
):
    import json

    report = QCReport(checks=(), face=mesh_geometry)
    result = pipeline.render_preview(face_image, report, {"wrinkle": 40}, settings)
    summary = pipeline.describe_result(result, report)

    json.dumps(summary)  # must not raise
    assert summary["disclaimer"] == DISCLAIMER
    assert summary["detector"] == mesh_geometry.detector
    assert summary["landmarks"] is True
    assert summary["treatments"][0]["key"] == "wrinkle"
    assert summary["identity"]["threshold"] == settings.identity_min_ssim
    assert "verdict" in summary["identity"]


def test_describe_result_handles_a_faceless_report(face_image):
    empty = QCReport(checks=(), face=None)
    summary = pipeline.describe_result(PreviewResult(face_image, face_image), empty)
    assert summary["detector"] is None
    assert summary["landmarks"] is False
    assert summary["identity"] is None


# ------------------------------------------------------------------ end to end


def test_run_end_to_end_on_the_sample(sample_bytes, settings):
    report, result = pipeline.run(sample_bytes, {"wrinkle": 50, "tone": 20}, settings)
    if report.face is None:  # pragma: no cover - slim OpenCV builds only
        pytest.skip("No face detector available.")
    assert result is not None
    assert [a.key for a in result.applied] == ["wrinkle", "tone"]
    assert result.identity.within_threshold is True
    assert result.edited.shape == result.original.shape


def test_run_returns_no_preview_when_no_face_is_found(settings):
    payload = encode_png(np.zeros((300, 300, 3), np.uint8))
    report, result = pipeline.run(payload, {"wrinkle": 50}, settings)
    assert report.face is None
    assert result is None


def test_the_pipeline_never_imports_streamlit():
    """Architecture guard: the image layer must not depend on the UI."""
    import sys

    for module in ("src.pipeline", "src.treatments", "src.qc", "src.imaging"):
        assert module in sys.modules
    source_modules = [sys.modules[m] for m in ("src.pipeline", "src.qc", "src.imaging")]
    for module in source_modules:
        assert "streamlit" not in getattr(module, "__dict__", {})
