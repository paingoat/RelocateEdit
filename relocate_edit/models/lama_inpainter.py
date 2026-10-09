"""big-lama inpainting, with the high-resolution feature refiner on request.

The checkpoint is loaded into the FFC generator directly. That skips the
training module, so PyTorch Lightning, albumentations, and the ADE20k loss
are not imported.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import torch
import yaml

from relocate_edit.models.base import BaseModelWrapper
from relocate_edit.ops.mask_ops import dilate_mask
from relocate_edit.utils.devices import torch_load
from relocate_edit.utils.third_party import ensure_on_path, vendor_path

_DOWNLOAD = "Run scripts/02_download_weights.sh to fetch big-lama."


class _RefinerHandle(torch.nn.Module):
    """The refiner only reads these attributes off the training module."""

    def __init__(self, generator):
        super().__init__()
        self.generator = generator
        self.concat_mask = True
        self.add_noise_kwargs = None


class LamaInpainter(BaseModelWrapper):
    name = "LaMa"

    def __init__(self, config, device: str):
        super().__init__(device)
        self.config = config
        self.handle = None

    def load(self):
        self.require_cuda()
        root = Path(self.config.checkpoint_dir)
        config_path = root / "config.yaml"
        checkpoint = root / "models" / "best.ckpt"
        self.require_file(str(config_path), _DOWNLOAD)
        self.require_file(str(checkpoint), _DOWNLOAD)
        ensure_on_path(vendor_path("lama"))
        from saicinpainting.training.modules.ffc import FFCResNetGenerator

        with open(config_path, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        generator_cfg = dict(raw["generator"])
        generator_cfg.pop("kind", None)
        accepted = set(inspect.signature(FFCResNetGenerator.__init__).parameters) - {"self"}
        generator_cfg = {key: value for key, value in generator_cfg.items() if key in accepted}
        generator = FFCResNetGenerator(**generator_cfg)

        state = torch_load(checkpoint, map_location="cpu")
        weights = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
        prefix = "generator."
        generator_weights = {key[len(prefix):]: value for key, value in weights.items() if key.startswith(prefix)}
        if not generator_weights:
            raise RuntimeError(f"{checkpoint} has no 'generator.*' weights.")
        generator.load_state_dict(generator_weights, strict=True)
        generator.eval()
        for parameter in generator.parameters():
            parameter.requires_grad = False
        generator.to(self.device)
        self.model = generator
        self.handle = _RefinerHandle(generator).to(self.device)
        self.handle.eval()

    def inpaint(self, image: np.ndarray, mask: np.ndarray, dilation: int | None = None, refine: bool | None = None):
        self.ensure_loaded()
        hole = dilate_mask(mask, self.config.dilation if dilation is None else int(dilation))
        if not hole.any():
            raise ValueError("Inpaint mask is empty.")
        use_refine = self.config.refine if refine is None else bool(refine)
        if use_refine:
            self._disable_inplace_relu()
            result, refined_note = self._refine(image, hole)
        else:
            result = self._forward(image, hole)
            refined_note = "single forward, no feature refinement"
        info = {
            "refine": use_refine,
            "dilation_px": int(self.config.dilation if dilation is None else dilation),
            "hole_area": int(hole.sum()),
            "detail": refined_note,
        }
        return result, hole, info

    def _forward(self, image: np.ndarray, hole: np.ndarray) -> np.ndarray:
        height, width = hole.shape
        image_t, mask_t = self._tensors(image, hole, pad=True)
        with torch.inference_mode():
            masked = image_t * (1 - mask_t)
            predicted = self.model(torch.cat([masked, mask_t], dim=1))
            composed = mask_t * predicted + (1 - mask_t) * image_t
        out = composed[0].permute(1, 2, 0).detach().float().cpu().numpy()
        out = np.clip(out[:height, :width] * 255.0, 0, 255).astype(np.uint8)
        return out

    def _refine(self, image: np.ndarray, hole: np.ndarray):
        from saicinpainting.evaluation.refinement import refine_predict

        height, width = hole.shape
        image_t, mask_t = self._tensors(image, hole, pad=False)
        batch = {
            "image": image_t,
            "mask": mask_t,
            "unpad_to_size": (torch.tensor([height]), torch.tensor([width])),
        }
        index = 0 if self.device.index is None else int(self.device.index)
        cfg = self.config
        with torch.enable_grad():
            predicted = refine_predict(
                batch,
                self.handle,
                gpu_ids=str(index),
                modulo=8,
                n_iters=int(cfg.n_iters),
                lr=float(cfg.lr),
                min_side=int(cfg.min_side),
                max_scales=int(cfg.max_scales),
                px_budget=int(cfg.px_budget),
            )
        self.model.to(self.device)
        out = predicted[0].permute(1, 2, 0).float().cpu().numpy()
        out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
        note = "feature refinement"
        if out.shape[0] != height or out.shape[1] != width:
            import cv2
            note = f"feature refinement resized the image to {out.shape[1]}x{out.shape[0]} to stay under px_budget, then scaled it back"
            out = cv2.resize(out, (width, height), interpolation=cv2.INTER_LINEAR)
        return out, note

    def _tensors(self, image: np.ndarray, hole: np.ndarray, pad: bool):
        image_t = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
        mask_t = torch.from_numpy(hole.astype(np.float32))[None, None]
        if pad:
            image_t = _pad_modulo(image_t)
            mask_t = _pad_modulo(mask_t)
        return image_t.to(self.device), mask_t.to(self.device)

    def _disable_inplace_relu(self):
        for module in self.model.modules():
            if isinstance(module, torch.nn.ReLU):
                module.inplace = False


def _pad_modulo(tensor: torch.Tensor, modulo: int = 8) -> torch.Tensor:
    height, width = tensor.shape[-2:]
    out_h = (height + modulo - 1) // modulo * modulo
    out_w = (width + modulo - 1) // modulo * modulo
    return torch.nn.functional.pad(tensor, (0, out_w - width, 0, out_h - height), mode="replicate")
