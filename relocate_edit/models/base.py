"""Common load / device behavior for the four model wrappers."""

from __future__ import annotations

import torch

from relocate_edit.utils.devices import (
    move_module,
    sync_stored_device,
    unsupported_cuda_arch_message,
)


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
        index = torch.cuda.current_device() if self.device.index is None else self.device.index
        message = unsupported_cuda_arch_message(
            torch.cuda.get_device_name(index),
            torch.cuda.get_device_capability(index),
            torch.cuda.get_arch_list(),
            torch.__version__,
            torch.version.cuda,
        )
        if message:
            raise RuntimeError(message)
        try:
            torch.ones(1, device=self.device)
            torch.cuda.synchronize(self.device)
        except RuntimeError as exc:
            if "no kernel image" not in str(exc):
                raise
            raise RuntimeError(
                f"This PyTorch build ({torch.__version__}, CUDA {torch.version.cuda}) "
                f"cannot run kernels on {torch.cuda.get_device_name(index)}. "
                "Reinstall the environment with: bash scripts/01_create_env.sh"
            ) from exc

    @staticmethod
    def require_file(path: str, hint: str):
        from pathlib import Path

        if not Path(path).is_file():
            raise FileNotFoundError(f"Missing weight file: {path}\n{hint}")
        return path
