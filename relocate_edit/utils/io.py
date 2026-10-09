"""Write one stage of a run to `outputs/<run_id>/`."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image


def new_run_dir(outputs_dir: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = Path(outputs_dir) / stamp
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def save_stage(run_dir: str, stage: str, images: dict, info: dict) -> None:
    folder = Path(run_dir)
    folder.mkdir(parents=True, exist_ok=True)
    for name, array in images.items():
        _write_png(folder / f"{stage}_{name}.png", array)
    with open(folder / f"{stage}.json", "w", encoding="utf-8") as handle:
        json.dump(_jsonable(info), handle, indent=2, ensure_ascii=False)


def _write_png(path: Path, array: np.ndarray) -> None:
    array = np.asarray(array)
    if array.dtype == bool:
        array = array.astype(np.uint8) * 255
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        Image.fromarray(array, mode="L").save(path)
    else:
        Image.fromarray(array[..., :3]).save(path)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
