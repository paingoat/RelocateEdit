"""Move a module, including tensors that were stored with `.cuda()` and never registered."""

from __future__ import annotations

import torch


def torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def move_module(module: torch.nn.Module, device) -> torch.nn.Module:
    device = torch.device(device)
    module.to(device)
    for submodule in module.modules():
        for key, value in list(vars(submodule).items()):
            if torch.is_tensor(value):
                setattr(submodule, key, value.to(device))
    return module
