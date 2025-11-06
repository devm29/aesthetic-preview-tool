"""Tests for the treatment registry and the built-in treatments.

The registry is the extensibility seam: adding a treatment should be one class
and nothing else. The "custom treatment" tests below are the executable proof
of that claim.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.imaging import mask_bbox
from src.masks import lip_ring_mask, under_eye_union
from src.models import TreatmentContext
from src.treatments import (
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

BUILT_IN_KEYS = {"wrinkle", "tone", "pigment", "lip", "under_eye"}


@pytest.fixture
def temp_key():
    """Register-and-clean-up helper so the global registry stays pristine."""
    created = []
    yield created.append
    for key in created:
        unregister(key)


# --------------------------------------------------------------------- registry


def test_every_built_in_treatment_is_registered():
    assert {t.key for t in all_treatments()} == BUILT_IN_KEYS


def test_treatments_are_returned_in_deterministic_application_order():
    orders = [t.spec.order for t in all_treatments()]
    assert orders == sorted(orders)
    assert [t.key for t in all_treatments()] == [t.key for t in all_treatments()]


def test_groups_partition_the_registry():
    adjustments = {t.key for t in all_treatments(ADJUSTMENT)}
    procedures = {t.key for t in all_treatments(PROCEDURE)}
    assert adjustments == {"wrinkle", "tone", "pigment"}
    assert procedures == {"lip", "under_eye"}
    assert adjustments | procedures == BUILT_IN_KEYS


def test_get_treatment_returns_none_for_an_unknown_key():
    assert get_treatment("does-not-exist") is None


def test_default_intensities_covers_every_treatment():
    defaults = default_intensities()
    assert set(defaults) == BUILT_IN_KEYS
    assert all(0 <= v <= 100 for v in defaults.values())


def test_procedures_default_to_off():
    """A procedure preview should never be applied without the user asking."""
    for treatment in all_treatments(PROCEDURE):
        assert treatment.spec.default == 0


def test_registering_a_duplicate_key_is_rejected(temp_key):
    class First(Treatment):
        spec = TreatmentSpec(key="dupe", label="Dupe", summary="x")

        def apply(self, image_rgb, context, intensity):
            return image_rgb

    register(First)
    temp_key("dupe")

    class Second(Treatment):
        spec = TreatmentSpec(key="dupe", label="Dupe again", summary="x")

        def apply(self, image_rgb, context, intensity):
            return image_rgb

    with pytest.raises(ValueError, match="Duplicate treatment key"):
        register(Second)


def test_a_treatment_cannot_be_instantiated_without_apply():
    class Incomplete(Treatment):
        spec = TreatmentSpec(key="incomplete", label="Incomplete", summary="x")

    with pytest.raises(TypeError):
        Incomplete()


# ------------------------------------------------------------- the seam itself


def test_a_third_party_treatment_needs_only_a_class(temp_key, mesh_context):
    """One class, one decorator - the UI, CLI and pipeline pick it up."""

    @register
    class Sepia(Treatment):
        spec = TreatmentSpec(
            key="sepia",
            label="Sepia",
            summary="Third-party treatment registered in a test.",
            group=ADJUSTMENT,
            order=5,
            default=0,
        )

        def apply(self, image_rgb, context, intensity):
            if intensity <= 0:
                return image_rgb
            warm = image_rgb.astype(np.float32) * (1.0, 0.9, 0.7)
            return np.clip(warm, 0, 255).astype(np.uint8)

    temp_key("sepia")

    assert get_treatment("sepia") is not None
    assert "sepia" in default_intensities()
    # order=5 is the lowest, so it must run first.
    assert all_treatments()[0].key == "sepia"

    edited = get_treatment("sepia").apply(mesh_context.original, mesh_context, 50)
    assert not np.array_equal(edited, mesh_context.original)


def test_availability_blocks_a_landmark_treatment_on_a_bbox_only_face(bbox_context):
    for treatment in all_treatments(PROCEDURE):
        usable, reason = treatment.availability(bbox_context)
        assert usable is False
        assert "landmark" in reason.lower()
        assert "requirements-landmarks.txt" in reason


def test_availability_allows_landmark_treatments_with_a_mesh(mesh_context):
    for treatment in all_treatments(PROCEDURE):
        usable, reason = treatment.availability(mesh_context)
        assert usable is True
        assert reason == ""


def test_engine_reports_parametric_without_a_generative_editor(mesh_context):
    assert all(t.engine(mesh_context) == "parametric" for t in all_treatments())


def test_engine_reports_generative_only_for_capable_treatments(
    face_image, mesh_geometry, settings, recording_editor
):
    from .conftest import make_context

    context = make_context(face_image, mesh_geometry, settings, recording_editor)
    for treatment in all_treatments():
        expected = "generative" if treatment.spec.supports_generative else "parametric"
        assert treatment.engine(context) == expected


# ---------------------------------------------------------- built-in behaviour


@pytest.mark.parametrize("key", sorted(BUILT_IN_KEYS))
def test_zero_intensity_is_always_a_no_op(key, mesh_context):
    treatment = get_treatment(key)
    out = treatment.apply(mesh_context.original, mesh_context, 0)
    assert np.array_equal(out, mesh_context.original)


@pytest.mark.parametrize("key", sorted(BUILT_IN_KEYS))
def test_every_treatment_returns_a_valid_image(key, mesh_context):
    out = get_treatment(key).apply(mesh_context.original, mesh_context, 70)
    assert out.shape == mesh_context.original.shape
    assert out.dtype == np.uint8


@pytest.mark.parametrize("key", ["wrinkle", "tone", "pigment"])
def test_adjustments_change_the_image_inside_the_skin_mask(key, mesh_context):
    out = get_treatment(key).apply(mesh_context.original, mesh_context, 90)
    changed = np.any(out != mesh_context.original, axis=2)
    assert changed[mesh_context.skin_mask > 0].any()


@pytest.mark.parametrize("key", ["wrinkle", "tone", "pigment"])
def test_adjustments_leave_pixels_far_outside_the_mask_untouched(key, mesh_context):
    out = get_treatment(key).apply(mesh_context.original, mesh_context, 90)
    box = mask_bbox(mesh_context.skin_mask, pad=16)
    assert box is not None
    _, y, _, h = box
    assert np.array_equal(out[:y], mesh_context.original[:y])
    assert np.array_equal(out[y + h :], mesh_context.original[y + h :])


def test_wrinkle_smoothing_reduces_high_frequency_detail(mesh_context):
    import cv2

    treatment = get_treatment("wrinkle")
    out = treatment.apply(mesh_context.original, mesh_context, 100)
    inside = mesh_context.skin_mask > 0

    def detail(image):
        grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return float(cv2.Laplacian(grey, cv2.CV_64F)[inside].var())

    assert detail(out) < detail(mesh_context.original)


def test_stronger_intensity_produces_a_larger_change(mesh_context):
    treatment = get_treatment("wrinkle")
    original = mesh_context.original.astype(np.float32)
    gentle = treatment.apply(mesh_context.original, mesh_context, 20).astype(np.float32)
    strong = treatment.apply(mesh_context.original, mesh_context, 100).astype(np.float32)
    assert np.abs(strong - original).mean() > np.abs(gentle - original).mean()


def test_lip_volume_edits_around_the_lip_ring(mesh_context):
    out = get_treatment("lip").apply(mesh_context.original, mesh_context, 80)
    ring = lip_ring_mask(mesh_context.original, mesh_context.geometry.landmarks)
    assert ring is not None
    changed = np.any(out != mesh_context.original, axis=2)
    assert changed[ring > 0].any()


def test_under_eye_softening_lightens_the_under_eye_region(mesh_context):
    import cv2

    dark = mesh_context.original.copy()
    region = under_eye_union(dark, mesh_context.geometry.landmarks)
    assert region is not None
    dark[region > 0] = (90, 70, 78)

    context = TreatmentContext(
        original=dark,
        geometry=mesh_context.geometry,
        skin_mask=mesh_context.skin_mask,
        settings=mesh_context.settings,
        editor=mesh_context.editor,
    )
    out = get_treatment("under_eye").apply(dark, context, 90)

    def lightness(image):
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        return float(lab[:, :, 0][region > 0].mean())

    assert lightness(out) > lightness(dark)


def test_a_procedure_notes_why_it_did_nothing_when_the_region_is_missing(face_image, settings):
    """A treatment that cannot find its region must explain itself, not fail silently."""
    from .conftest import make_context, make_geometry

    geometry = make_geometry(with_landmarks=True)
    collapsed = np.full_like(geometry.landmarks, 100.0)
    broken = type(geometry)(
        bbox=geometry.bbox,
        detector=geometry.detector,
        landmarks=collapsed,
        roll_deg=geometry.roll_deg,
    )
    context = make_context(face_image, broken, settings)

    out = get_treatment("lip").apply(face_image, context, 60)
    assert np.array_equal(out, face_image)
    assert any("could not be located" in note for note in context.notes)


# ------------------------------------------------------- generative dispatch


def test_a_capable_treatment_uses_the_generative_editor_when_available(
    face_image, mesh_geometry, settings, recording_editor
):
    from .conftest import make_context

    context = make_context(face_image, mesh_geometry, settings, recording_editor)
    out = get_treatment("lip").apply(face_image, context, 60)

    assert len(recording_editor.calls) == 1
    call = recording_editor.calls[0]
    assert "lip" in call["prompt"]
    assert call["negative_prompt"]
    assert 0.1 <= call["strength"] <= 0.9
    assert np.any(np.all(out == (255, 0, 255), axis=2)), "editor output was not composited"


def test_a_failing_generative_pass_falls_back_to_parametric_and_says_so(
    face_image, mesh_geometry, settings
):
    from .conftest import RecordingEditor, make_context

    editor = RecordingEditor(fail=True)
    context = make_context(face_image, mesh_geometry, settings, editor)
    out = get_treatment("under_eye").apply(face_image, context, 60)

    assert len(editor.calls) == 1
    assert not np.array_equal(out, face_image), "parametric fallback did nothing"
    assert any("parametric engine" in note for note in context.notes)


def test_context_note_is_deduplicated(mesh_context):
    mesh_context.note("same")
    mesh_context.note("same")
    assert mesh_context.notes == ["same"]


def test_under_eye_softening_pulls_chroma_towards_the_surrounding_skin(mesh_context):
    """Regression: a fixed chroma shift pushed pale skin into a green cast.

    The correction now interpolates towards the median chroma of the skin
    outside the region, which cannot overshoot past neutral.
    """
    import cv2

    base = mesh_context.original.copy()
    region = under_eye_union(base, mesh_context.geometry.landmarks)
    assert region is not None
    # A bluish-purple shadow, which is what the treatment is meant to correct.
    base[region > 0] = (108, 92, 124)

    context = TreatmentContext(
        original=base,
        geometry=mesh_context.geometry,
        skin_mask=mesh_context.skin_mask,
        settings=mesh_context.settings,
        editor=mesh_context.editor,
    )
    out = get_treatment("under_eye").apply(base, context, 100)

    def chroma(image, selection):
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        return np.array(
            [lab[:, :, 1][selection].mean(), lab[:, :, 2][selection].mean()], np.float32
        )

    inside = region > 0
    surrounding = (mesh_context.skin_mask > 0) & ~inside
    reference = chroma(base, surrounding)
    before = chroma(base, inside)
    after = chroma(out, inside)

    assert np.linalg.norm(after - reference) < np.linalg.norm(before - reference)
    # Never past the target: an overshoot is what produced the green cast.
    for channel in range(2):
        low, high = sorted((before[channel], reference[channel]))
        assert low - 1.0 <= after[channel] <= high + 1.0
