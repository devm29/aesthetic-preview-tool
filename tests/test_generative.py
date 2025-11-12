"""Tests for the optional generative editor.

No test here imports torch or diffusers, downloads a model, or needs a GPU.
The SDXL pipeline is replaced by a fake so the ROI crop/resize/composite logic
- the part this repo actually owns - is exercised.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest
from PIL import Image

from src.config import Settings
from src.generative import INSTALL_HINT, NullEditor, SdxlInpainter, build_editor


class FakePipeline:
    """Returns a solid red image at the size it was handed."""

    def __init__(self) -> None:
        self.kwargs = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        size = kwargs["image"].size
        return types.SimpleNamespace(images=[Image.new("RGB", size, (200, 0, 0))])


class FakeGenerator:
    def __init__(self, device=None):
        self.device = device
        self.seed = None

    def manual_seed(self, seed):
        self.seed = seed
        return self


@pytest.fixture
def fake_torch(monkeypatch):
    module = types.SimpleNamespace(Generator=FakeGenerator)
    monkeypatch.setitem(sys.modules, "torch", module)
    return module


# ------------------------------------------------------------------ NullEditor


def test_null_editor_is_never_available():
    editor = NullEditor("because reasons")
    assert editor.available() is False
    assert editor.unavailable_reason() == "because reasons"


def test_null_editor_raises_rather_than_silently_returning_the_input():
    editor = NullEditor("switched off")
    with pytest.raises(RuntimeError, match="switched off"):
        editor.inpaint(np.zeros((8, 8, 3), np.uint8), np.zeros((8, 8), np.uint8), "x")


# ----------------------------------------------------------------- build_editor


def test_build_editor_returns_a_null_editor_when_generative_is_off():
    editor = build_editor(Settings(enable_generative=False))
    assert editor.available() is False
    assert "DERMA_ENABLE_GENERATIVE" in editor.unavailable_reason()


def test_build_editor_explains_the_missing_ml_stack(monkeypatch):
    monkeypatch.setattr(SdxlInpainter, "_deps", staticmethod(lambda: None))
    editor = build_editor(Settings(enable_generative=True))
    assert editor.available() is False
    assert editor.unavailable_reason() == INSTALL_HINT


def test_build_editor_returns_the_inpainter_when_the_stack_is_present(monkeypatch):
    monkeypatch.setattr(SdxlInpainter, "_deps", staticmethod(lambda: ("torch", object)))
    editor = build_editor(Settings(enable_generative=True))
    assert isinstance(editor, SdxlInpainter)
    assert editor.name == "sdxl-inpaint"


def test_sdxl_inpainter_load_raises_the_install_hint_without_deps(monkeypatch):
    monkeypatch.setattr(SdxlInpainter, "_deps", staticmethod(lambda: None))
    with pytest.raises(RuntimeError, match=r"requirements-ml\.txt"):
        SdxlInpainter(Settings())._load()


# ------------------------------------------------------------------- inpainting


def _inpainter(monkeypatch, settings: Settings) -> tuple:
    editor = SdxlInpainter(settings)
    pipeline = FakePipeline()
    monkeypatch.setattr(editor, "_load", lambda: (pipeline, "cpu"))
    return editor, pipeline


def test_inpaint_returns_the_input_for_an_empty_mask(monkeypatch, fake_torch):
    editor, _ = _inpainter(monkeypatch, Settings())
    image = np.zeros((32, 32, 3), np.uint8)
    out = editor.inpaint(image, np.zeros((32, 32), np.uint8), "prompt")
    assert out is image


def test_inpaint_only_composites_inside_the_mask(monkeypatch, fake_torch):
    editor, _ = _inpainter(monkeypatch, Settings(sdxl_max_side=768))
    image = np.zeros((64, 64, 3), np.uint8)
    mask = np.zeros((64, 64), np.uint8)
    mask[20:44, 20:44] = 255

    out = editor.inpaint(image, mask, "prompt", seed=0)

    assert out.shape == image.shape
    assert out.dtype == np.uint8
    assert np.all(out[:8, :8] == 0), "pixels outside the mask were modified"
    assert np.all(out[24:40, 24:40, 0] == 200), "the mask interior was not repainted"


def test_inpaint_downscales_a_large_roi_to_the_model_budget(monkeypatch, fake_torch):
    editor, pipeline = _inpainter(monkeypatch, Settings(sdxl_max_side=128))
    image = np.zeros((600, 600, 3), np.uint8)
    mask = np.zeros((600, 600), np.uint8)
    mask[50:550, 50:550] = 255

    out = editor.inpaint(image, mask, "prompt")

    assert max(pipeline.kwargs["image"].size) <= 128
    assert out.shape == image.shape


def test_inpaint_forwards_prompts_and_clamps_the_sampling_parameters(monkeypatch, fake_torch):
    editor, pipeline = _inpainter(monkeypatch, Settings())
    mask = np.zeros((48, 48), np.uint8)
    mask[10:38, 10:38] = 255

    editor.inpaint(
        np.zeros((48, 48, 3), np.uint8),
        mask,
        "make it nice",
        negative_prompt="cartoon",
        strength=9.0,
        steps=999,
        seed=7,
    )

    assert pipeline.kwargs["prompt"] == "make it nice"
    assert pipeline.kwargs["negative_prompt"] == "cartoon"
    assert pipeline.kwargs["strength"] == pytest.approx(0.95)
    assert pipeline.kwargs["num_inference_steps"] == 50
    assert pipeline.kwargs["generator"].seed == 7


def test_inpaint_passes_no_generator_without_a_seed(monkeypatch, fake_torch):
    editor, pipeline = _inpainter(monkeypatch, Settings())
    mask = np.zeros((48, 48), np.uint8)
    mask[10:38, 10:38] = 255
    editor.inpaint(np.zeros((48, 48, 3), np.uint8), mask, "p")
    assert pipeline.kwargs["generator"] is None


def test_an_empty_negative_prompt_is_sent_as_none(monkeypatch, fake_torch):
    editor, pipeline = _inpainter(monkeypatch, Settings())
    mask = np.zeros((48, 48), np.uint8)
    mask[10:38, 10:38] = 255
    editor.inpaint(np.zeros((48, 48, 3), np.uint8), mask, "p", negative_prompt="")
    assert pipeline.kwargs["negative_prompt"] is None
