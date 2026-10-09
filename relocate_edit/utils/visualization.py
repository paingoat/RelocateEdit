"""Overlays used by the Gradio panels. All functions return RGB uint8."""

from __future__ import annotations

import cv2
import matplotlib
import numpy as np


def overlay_mask(image: np.ndarray, mask: np.ndarray, color=(40, 210, 90), alpha=0.45) -> np.ndarray:
    out = image.copy()
    selected = mask.astype(bool)
    if not selected.any():
        return out
    tint = np.array(color, dtype=np.float32)
    blended = out[selected].astype(np.float32) * (1.0 - alpha) + tint * alpha
    out[selected] = blended.astype(np.uint8)
    return out


def draw_bbox(image: np.ndarray, bbox: tuple, color=(255, 170, 0), thickness: int = 2) -> np.ndarray:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    out = image.copy()
    cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    return out


def draw_points(image: np.ndarray, points, color=(255, 40, 40), radius: int = 5) -> np.ndarray:
    out = image.copy()
    for x, y in points:
        cv2.circle(out, (int(round(x)), int(round(y))), radius, color, -1)
    return out


def draw_arrow(image: np.ndarray, start, end, color=(220, 30, 30)) -> np.ndarray:
    out = image.copy()
    cv2.arrowedLine(
        out,
        (int(round(start[0])), int(round(start[1]))),
        (int(round(end[0])), int(round(end[1]))),
        color,
        3,
        tipLength=0.22,
    )
    cv2.circle(out, (int(round(end[0])), int(round(end[1]))), 6, color, -1)
    return out


def depth_colormap(depth: np.ndarray) -> np.ndarray:
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        return np.zeros(depth.shape[:2] + (3,), dtype=np.uint8)
    lo, hi = np.percentile(finite, 2), np.percentile(finite, 98)
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((depth - lo) / (hi - lo), 0, 1)
    colored = matplotlib.colormaps["Spectral_r"](norm)[..., :3]
    return (colored * 255).astype(np.uint8)


def side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    height = max(left.shape[0], right.shape[0])
    pair = [_fit_height(left, height), _fit_height(right, height)]
    gap = np.full((height, 8, 3), 255, dtype=np.uint8)
    return np.concatenate([pair[0], gap, pair[1]], axis=1)


def candidate_strip(image: np.ndarray, masks, max_width: int = 1400) -> np.ndarray:
    tiles = [overlay_mask(image, mask) for mask in masks]
    if not tiles:
        return image
    strip = np.concatenate(tiles, axis=1)
    return limit_width(strip, max_width)


def limit_width(image: np.ndarray, max_width: int) -> np.ndarray:
    if image.shape[1] <= max_width:
        return image
    scale = max_width / image.shape[1]
    height = max(1, int(round(image.shape[0] * scale)))
    return cv2.resize(image, (max_width, height), interpolation=cv2.INTER_AREA)


def mask_preview(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    preview = np.zeros_like(image)
    preview[mask.astype(bool)] = 255
    return preview


def object_cutout(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Object on a white background, for the insert-stage panel."""
    selected = mask.astype(bool)
    cutout = np.full_like(image, 255)
    cutout[selected] = image[selected]
    return cutout


def _fit_height(image: np.ndarray, height: int) -> np.ndarray:
    if image.shape[0] == height:
        return image
    scale = height / image.shape[0]
    width = max(1, int(round(image.shape[1] * scale)))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
