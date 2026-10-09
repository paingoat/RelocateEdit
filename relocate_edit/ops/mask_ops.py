"""Mask helpers shared by the scribble prompt and the segmenter."""

from __future__ import annotations

import numpy as np


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Inclusive xyxy box of a boolean mask."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("Empty mask.")
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def box_iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1 + 1) * max(0, iy2 - iy1 + 1)
    area_a = (ax2 - ax1 + 1) * (ay2 - ay1 + 1)
    area_b = (bx2 - bx1 + 1) * (by2 - by1 + 1)
    return float(inter / max(area_a + area_b - inter, 1))


def centroid_xy(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def resize_mask(mask: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbor resize. `out_hw` is (height, width)."""
    oh, ow = out_hw
    ih, iw = mask.shape
    if (ih, iw) == (oh, ow):
        return mask.astype(bool)
    ys = np.clip(np.floor(np.arange(oh) * ih / oh).astype(np.int32), 0, ih - 1)
    xs = np.clip(np.floor(np.arange(ow) * iw / ow).astype(np.int32), 0, iw - 1)
    return mask[ys][:, xs].astype(bool)


def largest_component(mask: np.ndarray) -> np.ndarray:
    from scipy.ndimage import label

    labeled, count = label(mask.astype(bool))
    if count == 0:
        return mask.astype(bool)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return labeled == int(sizes.argmax())


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    from scipy.ndimage import binary_dilation

    radius = int(radius)
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    footprint = xx * xx + yy * yy <= radius * radius
    return binary_dilation(mask.astype(bool), structure=footprint)
