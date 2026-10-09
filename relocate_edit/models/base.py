"""Common load / device behavior for the four model wrappers."""

from __future__ import annotations

import torch

from relocate_edit.utils.devices import move_module, sync_stored_device


class BaseModelWrapper:
    name = "model"

    def __init__(self, device: str):
        self.device = torch.device(device)
        self.model = None

    def load(self):
        raise NotImplementedError

    def ensure_loaded(self):
        if self.model is None:
            self.load()
        return self.model

    def to(self, device):
        self.device = torch.device(device)
        if self.model is not None:
            move_module(self.model, self.device)
            # AnyDoor is a LightningModule. Its ``device`` is a read-only
            # property, so the second activate() used to crash here.
            sync_stored_device(self.model, self.device)
        return self

    def require_cuda(self):
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError(
                f"{self.name} needs a CUDA GPU. The configured device is '{self.device}'."
            )

    @staticmethod
    def require_file(path: str, hint: str):
        from pathlib import Path

        if not Path(path).is_file():
            raise FileNotFoundError(f"Missing weight file: {path}\n{hint}")
        return path
