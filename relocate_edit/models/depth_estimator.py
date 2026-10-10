"""Depth Anything V2. Relative disparity by default, metric meters optional."""

from __future__ import annotations

import numpy as np

from relocate_edit.models.base import BaseModelWrapper
from relocate_edit.utils.devices import torch_load
from relocate_edit.utils.third_party import ensure_on_path, vendor_path
from relocate_edit.utils.visualization import depth_colormap

_CONFIGS = {
    "vits": {"features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}

_DOWNLOAD = "Run scripts/02_download_weights.sh. Metric checkpoints need --with-metric."


class DepthEstimator(BaseModelWrapper):
    name = "Depth Anything V2"

    def __init__(self, config, device: str):
        super().__init__(device)
        self.config = config
        self.loaded_mode = None

    def load(self, mode: str | None = None):
        if self.device.type == "cuda":
            self.require_cuda()
        mode = mode or self.config.mode
        if mode not in self.config.checkpoints:
            raise ValueError(f"Unknown depth mode '{mode}'.")
        checkpoint = self.require_file(self.config.checkpoints[mode], _DOWNLOAD)
        encoder = self.config.encoder
        if encoder not in _CONFIGS:
            raise ValueError(f"Unknown Depth Anything encoder '{encoder}'.")
        ensure_on_path(vendor_path("depth_anything_v2"))
        from depth_anything_v2.dpt import DepthAnythingV2

        max_depth = None if mode == "relative" else float(self.config.max_depth[mode])
        model = DepthAnythingV2(encoder=encoder, max_depth=max_depth, **_CONFIGS[encoder])
        state = torch_load(checkpoint, map_location="cpu")
        model.load_state_dict(state)
        model.device = self.device
        model.to(self.device).eval()
        self.model = model
        self.loaded_mode = mode

    def ensure_loaded(self, mode: str | None = None):
        mode = mode or self.config.mode
        if self.model is None or self.loaded_mode != mode:
            self.load(mode)
        return self.model

    def estimate(self, image_rgb: np.ndarray, mode: str | None = None, input_size: int | None = None):
        mode = mode or self.config.mode
        model = self.ensure_loaded(mode)
        model.device = self.device
        # infer_image expects BGR uint8, which is what cv2.imread returns.
        bgr = image_rgb[:, :, ::-1]
        depth = model.infer_image(bgr, input_size or self.config.input_size)
        metric = mode != "relative"
        info = {
            "mode": mode,
            "metric": metric,
            "convention": "meters, larger is farther" if metric else "relative disparity, larger is closer",
            "min": round(float(np.nanmin(depth)), 4),
            "max": round(float(np.nanmax(depth)), 4),
            "median": round(float(np.nanmedian(depth)), 4),
            "encoder": self.config.encoder,
        }
        return depth.astype(np.float32), depth_colormap(depth), metric, info
