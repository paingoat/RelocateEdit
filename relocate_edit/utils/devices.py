"""Move a module, including tensors that were stored with `.cuda()` and never registered."""

from __future__ import annotations

import torch

_MISSING = object()


def unsupported_cuda_arch_message(device_name, capability, archs, torch_version, cuda_version):
    """None when this PyTorch build contains a kernel for the GPU."""
    major, minor = capability
    needed = f"sm_{major}{minor}"
    present = list(archs)
    if needed in present:
        return None
    listed = ", ".join(present) if present else "none"
    cuda = cuda_version or "unknown"
    return (
        f"{device_name} ({needed}) is not supported by this PyTorch build "
        f"({torch_version}, CUDA {cuda}). Included architectures: {listed}. "
        "Reinstall the environment with: bash scripts/01_create_env.sh"
    )


def ensure_pkg_resources() -> None:
    """Provide ``pkg_resources.declare_namespace`` when setuptools no longer does.

    Setuptools 82 removed ``pkg_resources``. ``lightning_fabric`` still calls
    ``declare_namespace`` as soon as it is imported. LaMa's checkpoint and
    AnyDoor both import Lightning, so that missing module stops step 4 and step 5.
    """
    try:
        import pkg_resources
    except ModuleNotFoundError:
        pkg_resources = None
    if pkg_resources is not None and hasattr(pkg_resources, "declare_namespace"):
        return

    import pkgutil
    import sys
    import types

    module = types.ModuleType("pkg_resources")

    def declare_namespace(package_name: str) -> None:
        imported = sys.modules.get(package_name)
        if imported is None:
            return
        path = getattr(imported, "__path__", None)
        if path is None:
            return
        imported.__path__ = pkgutil.extend_path(list(path), package_name)

    module.declare_namespace = declare_namespace
    sys.modules["pkg_resources"] = module


def torch_load(path, map_location="cpu"):
    ensure_pkg_resources()
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
