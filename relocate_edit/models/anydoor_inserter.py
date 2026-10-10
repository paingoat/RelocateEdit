"""Insert the source object into the target mask with AnyDoor.

`process_pairs` follows the Gradio demo, which matches the training collage:
a Sobel reference, a 4-channel hint, and pad value -1 for pixels that are not
part of the mask. Shape control is on by default because the target mask is
the scaled silhouette, not a loose box.
"""

from __future__ import annotations

import numpy as np
import torch

from relocate_edit.models.base import BaseModelWrapper
from relocate_edit.ops.anydoor_data_utils import (
    box2squre,
    clip_inner_box,
    expand_bbox,
    expand_image_mask,
    get_bbox_from_mask,
    nonempty_slice,
    pad_to_square,
    sobel,
)
from relocate_edit.utils.devices import ensure_pkg_resources
from relocate_edit.utils.third_party import ensure_on_path, vendor_path

_DOWNLOAD = "Run scripts/02_download_weights.sh. Stripping the AnyDoor checkpoint needs about 32 GB of RAM."


class AnyDoorInserter(BaseModelWrapper):
    name = "AnyDoor"

    def __init__(self, config, device: str):
        super().__init__(device)
        self.config = config
        self.sampler = None

    def load(self):
        self.require_cuda()
        ensure_pkg_resources()
        self.require_file(self.config.checkpoint, _DOWNLOAD)
        self.require_file(self.config.config, "configs/anydoor.yaml is missing.")
        ensure_on_path(vendor_path("anydoor"))
        from cldm.ddim_hacked import DDIMSampler
        from cldm.model import load_state_dict
        from ldm.util import instantiate_from_config
        from omegaconf import OmegaConf

        if self.config.save_memory:
            from cldm.hack import enable_sliced_attention
            enable_sliced_attention()

        raw = OmegaConf.load(self.config.config)
        if self.config.dino_weight:
            raw.model.params.cond_stage_config.params.weight = self.config.dino_weight
        model = instantiate_from_config(raw.model).cpu()
        model.load_state_dict(load_state_dict(self.config.checkpoint, location="cpu"))
        model.to(self.device)
        model.eval()
        self.model = model
        self.sampler = DDIMSampler(model)

    def insert(
        self,
        ref_image: np.ndarray,
        ref_mask: np.ndarray,
        background: np.ndarray,
        target_mask: np.ndarray,
        steps: int | None = None,
        guidance: float | None = None,
        strength: float | None = None,
        seed: int | None = None,
        shape_control: bool | None = None,
    ) -> np.ndarray:
        model = self.ensure_loaded()
        ref_mask = (ref_mask > 0).astype(np.uint8)
        target_mask = (target_mask > 0).astype(np.uint8)
        if int(ref_mask.sum()) < 10:
            raise ValueError("The reference mask is empty, so AnyDoor has no object to insert.")
        if int(target_mask.sum()) < 10:
            raise ValueError("The target mask is empty. Run the relocate step first.")

        steps = int(self.config.steps if steps is None else steps)
        guidance = float(self.config.guidance if guidance is None else guidance)
        strength = float(self.config.strength if strength is None else strength)
        seed = int(self.config.seed if seed is None else seed)
        shape_control = self.config.shape_control if shape_control is None else bool(shape_control)
        _seed_all(seed)

        item = process_pairs(ref_image, ref_mask, background, target_mask, enable_shape_control=shape_control)
        hint = torch.from_numpy(item["hint"].copy()).float().to(self.device)[None]
        hint = hint.permute(0, 3, 1, 2).contiguous()
        reference = torch.from_numpy(item["ref"].copy()).float().to(self.device)[None]
        reference = reference.permute(0, 3, 1, 2).contiguous()
        zeros = torch.zeros((1, 3, 224, 224), device=self.device)

        conditioning = {"c_concat": [hint], "c_crossattn": [model.get_learned_conditioning(reference)]}
        uncond = {"c_concat": [hint], "c_crossattn": [model.get_learned_conditioning(zeros)]}
        if self.config.save_memory:
            model.low_vram_shift(is_diffusing=True)
        model.control_scales = [strength] * 13
        samples, _ = self.sampler.sample(
            steps,
            1,
            (4, 64, 64),
            conditioning,
            verbose=False,
            eta=0.0,
            unconditional_guidance_scale=guidance,
            unconditional_conditioning=uncond,
        )
        if self.config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        decoded = model.decode_first_stage(samples)
        decoded = (decoded.permute(0, 2, 3, 1) * 127.5 + 127.5).clamp(0, 255)
        prediction = decoded[0, 1:].detach().float().cpu().numpy()
        canvas = background.copy()
        canvas = crop_back(prediction, canvas, item["extra_sizes"], item["tar_box_yyxx_crop"])
        y1, y2, x1, x2 = [int(v) for v in item["tar_box_yyxx"]]
        y1, x1 = max(0, y1), max(0, x1)
        y2, x2 = min(canvas.shape[0], y2), min(canvas.shape[1], x2)
        original = background.copy()
        original[y1:y2, x1:x2] = canvas[y1:y2, x1:x2]
        return original.astype(np.uint8)


def process_pairs(ref_image, ref_mask, tar_image, tar_mask, max_ratio=0.8, enable_shape_control=False):
    ref_box = get_bbox_from_mask(ref_mask)
    ref_mask_3 = np.stack([ref_mask, ref_mask, ref_mask], -1)
    masked_ref = ref_image * ref_mask_3 + 255 * (1 - ref_mask_3)
    y1, y2 = nonempty_slice(ref_box[0], ref_box[1], masked_ref.shape[0])
    x1, x2 = nonempty_slice(ref_box[2], ref_box[3], masked_ref.shape[1])
    masked_ref = masked_ref[y1:y2, x1:x2, :]
    ref_mask = ref_mask[y1:y2, x1:x2]

    ratio = np.random.randint(11, 15) / 10
    masked_ref, ref_mask = expand_image_mask(masked_ref, ref_mask, ratio=ratio)
    masked_ref = pad_to_square(masked_ref, pad_value=255, random=False)
    masked_ref = cv2_resize(masked_ref, (224, 224))

    ref_mask_3 = np.stack([ref_mask, ref_mask, ref_mask], -1)
    ref_mask_3 = pad_to_square(ref_mask_3 * 255, pad_value=0, random=False)
    ref_mask_3 = cv2_resize(ref_mask_3, (224, 224))
    ref_mask = ref_mask_3[:, :, 0]
    ref_image_collage = sobel(masked_ref, ref_mask / 255.0)

    tar_box = get_bbox_from_mask(tar_mask)
    tar_box = expand_bbox(tar_mask, tar_box, ratio=(1.1, 1.2))
    tar_box_full = tar_box
    tar_box_crop = expand_bbox(tar_image, tar_box, ratio=(1.3, 3.0))
    tar_box_crop = box2squre(tar_image, tar_box_crop)
    y1, y2 = nonempty_slice(tar_box_crop[0], tar_box_crop[1], tar_image.shape[0])
    x1, x2 = nonempty_slice(tar_box_crop[2], tar_box_crop[3], tar_image.shape[1])
    tar_box_crop = (y1, y2, x1, x2)
    cropped = tar_image[y1:y2, x1:x2, :]
    cropped_mask = tar_mask[y1:y2, x1:x2]
    iy1, iy2, ix1, ix2 = clip_inner_box(tar_box, tar_box_crop, cropped.shape[:2])

    collage_rgb = cv2_resize(ref_image_collage, (ix2 - ix1, iy2 - iy1))
    collage = cropped.copy()
    collage[iy1:iy2, ix1:ix2, :] = collage_rgb
    collage_mask = np.zeros_like(cropped, dtype=np.float32)
    collage_mask[iy1:iy2, ix1:ix2, :] = 1.0
    if enable_shape_control:
        collage_mask = np.stack([cropped_mask, cropped_mask, cropped_mask], -1).astype(np.float32)

    height_1, width_1 = collage.shape[0], collage.shape[1]
    collage = pad_to_square(collage, pad_value=0, random=False).astype(np.uint8)
    collage_mask = pad_to_square(collage_mask, pad_value=2, random=False).astype(np.uint8)
    height_2, width_2 = collage.shape[0], collage.shape[1]

    collage = cv2_resize(collage, (512, 512)).astype(np.float32)
    collage_mask = cv2_resize(collage_mask, (512, 512), nearest=True).astype(np.float32)
    collage_mask[collage_mask == 2] = -1
    collage = collage / 127.5 - 1.0
    hint = np.concatenate([collage, collage_mask[:, :, :1]], -1)

    return {
        "ref": (masked_ref.astype(np.float32) / 255.0),
        "hint": hint.astype(np.float32),
        "extra_sizes": np.array([height_1, width_1, height_2, width_2]),
        "tar_box_yyxx_crop": np.array(tar_box_crop),
        "tar_box_yyxx": np.array(tar_box_full),
    }


def crop_back(pred, tar_image, extra_sizes, tar_box_yyxx_crop):
    import cv2

    height_1, width_1, height_2, width_2 = [int(v) for v in extra_sizes]
    y1, y2, x1, x2 = [int(v) for v in tar_box_yyxx_crop]
    pred = np.clip(pred, 0, 255).astype(np.uint8)
    pred = cv2.resize(pred, (width_2, height_2))
    if width_1 == height_1:
        cropped = pred
    elif width_1 < width_2:
        pad = int((width_2 - width_1) / 2)
        cropped = pred[:, pad:pad + width_1, :]
    else:
        pad = int((height_2 - height_1) / 2)
        cropped = pred[pad:pad + height_1, :, :]

    margin = 3
    if cropped.shape[0] <= 2 * margin or cropped.shape[1] <= 2 * margin:
        margin = 0
    dest_y1, dest_y2 = y1 + margin, y2 - margin
    dest_x1, dest_x2 = x1 + margin, x2 - margin
    if dest_y2 <= dest_y1 or dest_x2 <= dest_x1:
        return tar_image
    patch = cropped[margin:cropped.shape[0] - margin, margin:cropped.shape[1] - margin, :]
    patch = cv2.resize(patch, (dest_x2 - dest_x1, dest_y2 - dest_y1))
    tar_image[dest_y1:dest_y2, dest_x1:dest_x2, :] = patch
    return tar_image


def cv2_resize(image, size, nearest=False):
    import cv2

    interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.resize(image.astype(np.uint8), size, interpolation=interpolation)


def _seed_all(seed: int) -> None:
    import random

    if seed < 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
