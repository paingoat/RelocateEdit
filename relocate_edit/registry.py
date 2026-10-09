"""Lazy model registry. With offload, only the active model stays on the GPU."""

from __future__ import annotations

import torch

from relocate_edit.models.anydoor_inserter import AnyDoorInserter
from relocate_edit.models.depth_estimator import DepthEstimator
from relocate_edit.models.lama_inpainter import LamaInpainter
from relocate_edit.models.seem_segmenter import SeemSegmenter


class ModelRegistry:
    def __init__(self, config):
        self.config = config
        self.offload = bool(config.offload)
        self.device = config.device
        self._wrappers = {}

    def wrapper(self, name: str):
        if name not in self._wrappers:
            self._wrappers[name] = self._build(name)
        return self._wrappers[name]

    def activate(self, name: str):
        wrapper = self.wrapper(name)
        if self.offload:
            for other, item in self._wrappers.items():
                if other != name and item.model is not None:
                    item.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        wrapper.to(self.device)
        return wrapper

    def get(self, name: str):
        wrapper = self.activate(name)
        wrapper.ensure_loaded()
        return wrapper

    def preload(self):
        for name in ("seem", "depth", "lama", "anydoor"):
            self.get(name)
        if self.offload:
            for item in self._wrappers.values():
                if item.model is not None:
                    item.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _build(self, name: str):
        device = self.device
        if name == "seem":
            return SeemSegmenter(self.config.seem, device)
        if name == "depth":
            return DepthEstimator(self.config.depth, device)
        if name == "lama":
            return LamaInpainter(self.config.lama, device)
        if name == "anydoor":
            return AnyDoorInserter(self.config.anydoor, device)
        raise KeyError(name)
