"""Move a module, including tensors that were stored with `.cuda()` and never registered."""

from __future__ import annotations

import torch

_MISSING = object()


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


def sync_stored_device(module: torch.nn.Module, device) -> None:
    """Refresh plain ``.device`` attributes after a move.

    Lightning (AnyDoor) and SEEM expose ``device`` as a read-only property that
    follows the parameters. Assigning it raises ``AttributeError``. Depth
    Anything and the DINOv2 encoder instead store a real attribute, and that
    one has to be updated or the next forward pass stays on the old device.
    """
    device = torch.device(device)
    for submodule in module.modules():
        current = vars(submodule).get("device", _MISSING)
        if current is _MISSING:
            continue
        if current is not None and not isinstance(current, (str, torch.device)):
            continue
        try:
            submodule.device = device
        except (AttributeError, TypeError):
            continue
