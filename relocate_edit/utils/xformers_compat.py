"""Whether the installed xformers binary can launch kernels on this GPU.

A failed xformers kernel leaves the CUDA context in an error state, so this
has to be decided before the call. The CUDA 12.8 wheel of xformers 0.0.31
stops at Hopper (sm_90). Blackwell is sm_100 and sm_120.
"""

from __future__ import annotations


def xformers_supports_device(device) -> bool:
    import torch

    device = torch.device(device)
    if device.type != "cuda":
        return True
    major, minor = torch.cuda.get_device_capability(device)
    compiled = compiled_architectures()
    if compiled is None:
        return major < 10
    return architecture_listed(compiled, major, minor)


def compiled_architectures() -> list[str] | None:
    try:
        from xformers import _cpp_lib
    except Exception:
        return None
    meta = getattr(_cpp_lib, "_build_metadata", None)
    env = getattr(meta, "build_env", None) or {}
    raw = env.get("TORCH_CUDA_ARCH_LIST") or ""
    if not str(raw).strip():
        return None
    parts: list[str] = []
    for chunk in str(raw).replace(";", ",").split(","):
        parts.extend(chunk.split())
    cleaned = [part.strip() for part in parts if part.strip()]
    return cleaned or None


def architecture_listed(archs, major: int, minor: int) -> bool:
    target = f"{major}.{minor}"
    for token in archs:
        base = str(token).split("+", 1)[0]
        if base.endswith("a"):
            base = base[:-1]
        if base == target:
            return True
    return False
