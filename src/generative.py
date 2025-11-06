"""Optional generative (SDXL inpainting) editor.

The editor is an injected collaborator rather than a module-level import, so
the pipeline and its tests never need torch or diffusers installed. The default
editor is :class:`NullEditor`, which reports itself unavailable.
"""

from __future__ import annotations

import logging
import threading
from typing import Protocol

import cv2
import numpy as np
from PIL import Image

from .config import Settings, get_settings
from .imaging import mask_bbox

logger = logging.getLogger(__name__)

INSTALL_HINT = "Generative editing needs the optional ML stack: pip install -r requirements-ml.txt"


class GenerativeEditor(Protocol):
    """Anything that can repaint the inside of a mask."""

    name: str

    def available(self) -> bool:
        """Whether an inpaint call could succeed right now."""

    def unavailable_reason(self) -> str:
        """Human-readable explanation shown in the UI when unavailable."""

    def inpaint(
        self,
        image_rgb: np.ndarray,
        mask: np.ndarray,
        prompt: str,
        negative_prompt: str = "",
        strength: float = 0.55,
        steps: int = 14,
        seed: int | None = None,
    ) -> np.ndarray:
        """Return a copy of the image with the masked region regenerated."""


class NullEditor:
    """Stand-in used whenever generative editing is off or unsupported."""

    name = "none"

    def __init__(self, reason: str = "Generative editing is disabled.") -> None:
        self._reason = reason

    def available(self) -> bool:
        return False

    def unavailable_reason(self) -> str:
        return self._reason

    def inpaint(self, image_rgb: np.ndarray, mask: np.ndarray, prompt: str, **_) -> np.ndarray:
        raise RuntimeError(self._reason)


class SdxlInpainter:
    """SDXL inpainting, loaded lazily and only inside the mask's bounding box."""

    name = "sdxl-inpaint"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._pipe = None
        self._device = None
        self._lock = threading.Lock()

    @staticmethod
    def _deps():
        try:
            import torch
            from diffusers import StableDiffusionXLInpaintPipeline
        except Exception:  # pragma: no cover - depends on optional install
            return None
        return torch, StableDiffusionXLInpaintPipeline

    def available(self) -> bool:
        return self._deps() is not None

    def unavailable_reason(self) -> str:
        return INSTALL_HINT

    def _load(self):
        if self._pipe is not None:
            return self._pipe, self._device
        deps = self._deps()
        if deps is None:
            raise RuntimeError(INSTALL_HINT)
        torch, pipeline_cls = deps
        with self._lock:
            if self._pipe is None:
                if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                    device, dtype = torch.device("mps"), torch.float32
                elif torch.cuda.is_available():
                    device, dtype = torch.device("cuda"), torch.float16
                else:
                    device, dtype = torch.device("cpu"), torch.float32
                logger.info("Loading %s on %s", self._settings.sdxl_model_id, device)
                pipe = pipeline_cls.from_pretrained(
                    self._settings.sdxl_model_id,
                    torch_dtype=dtype,
                    variant="fp16" if dtype == torch.float16 else None,
                )
                pipe.to(device)
                pipe.enable_attention_slicing()
                self._pipe, self._device = pipe, device
        return self._pipe, self._device

    def inpaint(
        self,
        image_rgb: np.ndarray,
        mask: np.ndarray,
        prompt: str,
        negative_prompt: str = "",
        strength: float = 0.55,
        steps: int = 14,
        seed: int | None = None,
    ) -> np.ndarray:
        pipe, device = self._load()
        box = mask_bbox(mask, pad=16)
        if box is None:
            return image_rgb
        x, y, w, h = box
        roi = np.ascontiguousarray(image_rgb[y : y + h, x : x + w])
        roi_mask = np.ascontiguousarray(mask[y : y + h, x : x + w])

        max_side = self._settings.sdxl_max_side
        scale = min(1.0, max_side / float(max(h, w)))
        if scale < 1.0:
            size = (max(8, int(w * scale)), max(8, int(h * scale)))
            model_img = cv2.resize(roi, size, interpolation=cv2.INTER_AREA)
            model_mask = cv2.resize(roi_mask, size, interpolation=cv2.INTER_NEAREST)
        else:
            model_img, model_mask = roi, roi_mask

        import torch

        generator = None
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(int(seed))

        rendered = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            image=Image.fromarray(model_img),
            mask_image=Image.fromarray(model_mask),
            guidance_scale=5.0,
            strength=float(np.clip(strength, 0.1, 0.95)),
            num_inference_steps=int(np.clip(steps, 4, 50)),
            generator=generator,
        ).images[0]

        out_roi = np.asarray(rendered)
        if out_roi.shape[:2] != (h, w):
            out_roi = cv2.resize(out_roi, (w, h), interpolation=cv2.INTER_CUBIC)
        weight = (roi_mask.astype(np.float32) / 255.0)[..., None]
        merged = out_roi.astype(np.float32) * weight + roi.astype(np.float32) * (1 - weight)

        result = image_rgb.copy()
        result[y : y + h, x : x + w] = np.clip(merged, 0, 255).astype(np.uint8)
        return result


def build_editor(settings: Settings | None = None) -> GenerativeEditor:
    """Return the configured editor, or a NullEditor explaining why not."""
    settings = settings or get_settings()
    if not settings.enable_generative:
        return NullEditor(
            "Generative editing is off. Set DERMA_ENABLE_GENERATIVE=true to enable it."
        )
    editor = SdxlInpainter(settings)
    if not editor.available():
        return NullEditor(INSTALL_HINT)
    return editor
