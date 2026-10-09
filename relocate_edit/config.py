from __future__ import annotations

"""Load `configs/default.yaml` into dataclasses.

Paths in the yaml are relative to the repository root unless they are absolute.
"""

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value):
    if value is None or value == "":
        return None
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return str(path)


def _build(cls, data, path_fields=()):
    data = dict(data or {})
    known = {item.name for item in fields(cls)}
    kwargs = {}
    for name in known:
        if name not in data:
            continue
        value = data[name]
        if name in path_fields and isinstance(value, str):
            value = resolve_path(value)
        kwargs[name] = value
    return cls(**kwargs)


@dataclass
class SeemConfig:
    config: str = "configs/seem_focall_demo.yaml"
    checkpoint: str = "weights/seem/seem_focall_v0.pt"
    prompt_mode: str = "loop"
    dilation: int = 3
    extra_points: int = 4
    w_sim: float = 0.5
    w_in: float = 0.3
    w_iou: float = 0.2


@dataclass
class DepthConfig:
    encoder: str = "vitl"
    mode: str = "relative"
    input_size: int = 518
    checkpoints: dict = field(default_factory=lambda: {
        "relative": "weights/depth/depth_anything_v2_vitl.pth",
        "metric_indoor": "weights/depth/depth_anything_v2_metric_hypersim_vitl.pth",
        "metric_outdoor": "weights/depth/depth_anything_v2_metric_vkitti_vitl.pth",
    })
    max_depth: dict = field(default_factory=lambda: {
        "metric_indoor": 20.0,
        "metric_outdoor": 80.0,
    })


@dataclass
class RelocateConfig:
    scale_factor: float = 1.0
    s_min: float = 0.2
    s_max: float = 5.0
    iterations: int = 3
    boundary: str = "shift"


@dataclass
class LamaConfig:
    checkpoint_dir: str = "weights/lama/big-lama"
    dilation: int = 15
    refine: bool = True
    n_iters: int = 15
    lr: float = 0.002
    min_side: int = 512
    max_scales: int = 3
    px_budget: int = 1_800_000


@dataclass
class AnyDoorConfig:
    config: str = "configs/anydoor.yaml"
    checkpoint: str = "weights/anydoor/anydoor.ckpt"
    dino_weight: str | None = None
    steps: int = 30
    guidance: float = 4.5
    strength: float = 1.0
    seed: int = 42
    shape_control: bool = True
    save_memory: bool = False


@dataclass
class AppConfig:
    device: str = "cuda:0"
    offload: bool = False
    outputs_dir: str = "output"
    seem: SeemConfig = field(default_factory=SeemConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    relocate: RelocateConfig = field(default_factory=RelocateConfig)
    lama: LamaConfig = field(default_factory=LamaConfig)
    anydoor: AnyDoorConfig = field(default_factory=AnyDoorConfig)


def load_config(path=None) -> AppConfig:
    config_path = Path(path) if path else ROOT / "configs" / "default.yaml"
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    with open(config_path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    seem = _build(SeemConfig, raw.get("seem"), ("config", "checkpoint"))
    depth_raw = dict(raw.get("depth") or {})
    checkpoints = dict(DepthConfig().checkpoints)
    checkpoints.update(depth_raw.get("checkpoints") or {})
    depth_raw["checkpoints"] = {key: resolve_path(value) for key, value in checkpoints.items()}
    max_depth = dict(DepthConfig().max_depth)
    max_depth.update(depth_raw.get("max_depth") or {})
    depth_raw["max_depth"] = max_depth
    depth = _build(DepthConfig, depth_raw)

    relocate = _build(RelocateConfig, raw.get("relocate"))
    lama = _build(LamaConfig, raw.get("lama"), ("checkpoint_dir",))
    anydoor = _build(AnyDoorConfig, raw.get("anydoor"), ("config", "checkpoint", "dino_weight"))

    app = _build(AppConfig, {key: raw[key] for key in raw if key in {"device", "offload", "outputs_dir"}})
    app.outputs_dir = resolve_path(app.outputs_dir)
    app.seem = seem
    app.depth = depth
    app.relocate = relocate
    app.lama = lama
    app.anydoor = anydoor
    return app
