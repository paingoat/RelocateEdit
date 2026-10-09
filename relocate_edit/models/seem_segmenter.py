"""SEEM Focal-L demo: one positive prompt in, one object mask out.

A loop drawn around an object is filled first. The network then sees a small
blob inside that region, not the outline. All 101 query masks are reranked
with the prompt similarity plus how well each mask sits inside the loop.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from relocate_edit.models.base import BaseModelWrapper
from relocate_edit.ops.mask_ops import box_iou, largest_component, mask_bbox, resize_mask
from relocate_edit.ops.scribble import build_prompt, fill_loop, stroke_points
from relocate_edit.utils.third_party import ensure_on_path, vendor_path
from relocate_edit.utils.visualization import candidate_strip, draw_bbox, draw_points, overlay_mask

_DOWNLOAD = "Run scripts/02_download_weights.sh to fetch seem_focall_v0.pt."


class SeemSegmenter(BaseModelWrapper):
    name = "SEEM"

    def __init__(self, config, device: str):
        super().__init__(device)
        self.config = config
        self.class_names = None

    def load(self):
        self.require_cuda()
        self.require_file(self.config.checkpoint, _DOWNLOAD)
        self.require_file(self.config.config, "The SEEM yaml is missing from configs/.")
        ensure_on_path(vendor_path("seem"))
        from modeling.BaseModel import BaseModel
        from modeling import build_model
        from utils.arguments import load_opt_from_config_files
        from utils.constants import COCO_PANOPTIC_CLASSES

        opt = load_opt_from_config_files([self.config.config])
        index = 0 if self.device.index is None else self.device.index
        torch.cuda.set_device(index)
        opt["CUDA"] = True
        opt["world_size"] = 1
        opt["local_size"] = 1
        opt["rank"] = 0
        opt["local_rank"] = index
        opt["device"] = self.device

        model = BaseModel(opt, build_model(opt))
        model = model.from_pretrained(self.config.checkpoint).eval()
        model.to(self.device)
        names = list(COCO_PANOPTIC_CLASSES) + ["background"]
        with torch.no_grad():
            model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(names, is_eval=True)
        self.class_names = names
        self.model = model

    def segment(self, image: np.ndarray, scribble: np.ndarray, prompt_mode: str | None = None):
        model = self.ensure_loaded()
        mode = prompt_mode or self.config.prompt_mode
        if mode not in {"loop", "stroke"}:
            raise ValueError("prompt_mode must be 'loop' or 'stroke'.")
        if scribble.shape[:2] != image.shape[:2]:
            raise ValueError("Scribble and image sizes do not match. Confirm the scribble again.")

        resized, prompt, region, points = self._prepare(image, scribble, mode)
        height, width = resized.shape[:2]
        images = torch.from_numpy(np.ascontiguousarray(resized)).permute(2, 0, 1).to(self.device)
        stroke = torch.from_numpy(prompt.astype(np.float32))[None, None].to(self.device)
        stroke = stroke > 0.5
        for key in ("spatial", "visual", "grounding", "audio"):
            model.model.task_switch[key] = False
        model.model.task_switch["spatial"] = True
        batch = {"image": images, "height": height, "width": width, "stroke": stroke}

        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            results, image_size, _extra = model.model.evaluate_demo([batch])

        binary, scores, probabilities = self._rerank(results, image_size, region, height, width)
        order = np.argsort(scores)[::-1]
        best = int(order[0])
        if scores[best] < -1e8:
            raise RuntimeError("SEEM did not return a mask inside the scribble. Draw the loop tighter, or switch to stroke mode and paint on the object.")

        chosen = largest_component(binary[best])
        full = resize_mask(chosen, image.shape[:2])
        if not full.any():
            raise RuntimeError("The selected mask disappeared while resizing. Try a larger loop.")
        bbox = mask_bbox(full)
        scale_x = image.shape[1] / width
        scale_y = image.shape[0] / height
        points_full = [(x * scale_x, y * scale_y) for x, y in points]

        overlay = overlay_mask(image, full)
        overlay = draw_bbox(overlay, bbox)
        overlay = draw_points(overlay, points_full)

        top = []
        top_scores = []
        for index in order[:3]:
            if scores[index] < -1e8:
                continue
            top.append(resize_mask(largest_component(binary[index]), image.shape[:2]))
            top_scores.append(round(float(scores[index]), 4))
        strip = candidate_strip(image, top)

        class_name = self._class_name(results, best)
        info = {
            "prompt_mode": mode,
            "class_name": class_name,
            "score": round(float(scores[best]), 4),
            "prompt_similarity": round(float(probabilities[best]), 4),
            "bbox_xyxy": list(bbox),
            "mask_area": int(full.sum()),
            "model_hw": [height, width],
            "top3_scores": top_scores,
        }
        return full, bbox, overlay, strip, info

    def _prepare(self, image, scribble, mode):
        from torchvision.transforms import Resize

        pil = Image.fromarray(image)
        resized_pil = Resize(512, interpolation=Image.BICUBIC)(pil)
        resized = np.asarray(resized_pil)
        stroke = Image.fromarray(scribble.astype(np.uint8) * 255).resize(resized_pil.size, Image.NEAREST)
        stroke = np.asarray(stroke) > 0
        if mode == "stroke":
            if not stroke.any():
                raise ValueError("Stroke mode needs paint on the object, not an empty canvas.")
            return resized, stroke, stroke, stroke_points(stroke)
        region = fill_loop(stroke)
        prompt, points = build_prompt(region, dilation=self.config.dilation, extra_points=self.config.extra_points)
        return resized, prompt, region, points

    def _rerank(self, results, image_size, region, height, width):
        logits = results["pred_masks"].float()
        pad_h, pad_w = int(image_size[-2]), int(image_size[-1])
        up = F.interpolate(logits, size=(pad_h, pad_w), mode="bilinear", align_corners=False)
        binary = (up[0, :, :height, :width] > 0).detach().cpu().numpy()

        visual = results["pred_maskembs"].float()
        spatial = results["pred_pspatials"].float()
        similarity = torch.matmul(visual, spatial.transpose(1, 2))[0, :, 0]
        probabilities = torch.softmax(similarity, dim=0).detach().cpu().numpy()

        region_box = mask_bbox(region)
        weights = self.config
        scores = np.full(binary.shape[0], -1e9, dtype=np.float64)
        for index in range(binary.shape[0]):
            mask = binary[index]
            area = int(mask.sum())
            if area < 10:
                continue
            inside = float((mask & region).sum()) / area
            try:
                overlap = box_iou(mask_bbox(mask), region_box)
            except ValueError:
                overlap = 0.0
            scores[index] = (
                weights.w_sim * float(probabilities[index])
                + weights.w_in * inside
                + weights.w_iou * overlap
            )
        return binary, scores, probabilities

    def _class_name(self, results, index: int) -> str:
        try:
            class_id = int(results["pred_logits"][0, index].float().argmax())
            return self.class_names[class_id]
        except Exception:
            return "unknown"
