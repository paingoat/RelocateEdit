"""Move a mask to a target point and scale it from the depth map.

The anchor is the mask centroid, and the target point is where that centroid
should land. Scale comes from the depth at the contact band (the bottom of the
mask) and the depth at the matching point after the move:

- relative disparity (larger = closer):  scale = d_dst / d_src
- metric depth in meters (larger = farther):  scale = Z_src / Z_dst

The ratio is iterated a few times because the destination contact point itself
depends on the scale. The result is clamped, then multiplied by the UI factor.

Boundary modes, applied after the scale is chosen:

- shift: translate so the whole object stays inside the image
- shrink: reduce the scale until the object fits, keeping the target point
- clip: keep the position and drop the part that falls outside the image
"""

from __future__ import annotations

import numpy as np


def relocate(
    image: np.ndarray,
    mask: np.ndarray,
    depth: np.ndarray,
    target_xy: tuple[float, float],
    scale_factor: float = 1.0,
    s_min: float = 0.2,
    s_max: float = 5.0,
    iterations: int = 3,
    boundary: str = "shift",
    metric: bool = False,
) -> dict:
    mask = mask.astype(bool)
    if mask.shape != depth.shape[:2]:
        raise ValueError("Depth map and source mask must have the same height and width.")
    if not mask.any():
        raise ValueError("Source mask is empty.")
    if boundary not in {"shift", "shrink", "clip"}:
        raise ValueError(f"Unknown boundary mode '{boundary}'. Use shift, shrink, or clip.")

    height, width = mask.shape
    scale, anchor, d_src, d_dst, contact = solve_scale(
        depth, mask, target_xy, scale_factor, s_min, s_max, metric, iterations,
    )
    bbox = _mask_bbox(mask)
    target = (float(target_xy[0]), float(target_xy[1]))

    if boundary == "shrink":
        scale = shrink_scale(scale, anchor, target, bbox, (height, width), s_min)
    if boundary == "shift" or not _fits(_map_bbox(bbox, scale, anchor, target), (height, width)):
        if boundary != "clip":
            target = shift_target(scale, anchor, target, bbox, (height, width))

    forward = forward_matrix(scale, anchor, target)
    placed = warp_nearest(mask, forward, (height, width))
    reference, visible_ratio = visible_source(mask, scale, anchor, target, (height, width))
    preview = paste_preview(image, mask, forward)
    return {
        "mask": placed,
        "reference_mask": reference,
        "preview": preview,
        "scale": float(scale),
        "anchor_xy": anchor,
        "placed_xy": target,
        "contact_xy": contact,
        "depth_src": float(d_src),
        "depth_dst": float(d_dst),
        "visible_ratio": float(visible_ratio),
        "boundary": boundary,
    }


def solve_scale(depth, mask, target_xy, scale_factor, s_min, s_max, metric, iterations):
    anchor = _centroid(mask)
    d_src, contact = contact_depth(mask, depth)
    scale = float(np.clip(scale_factor, s_min, s_max))
    d_dst = d_src
    for _ in range(max(int(iterations), 1)):
        dest_x = target_xy[0] + scale * (contact[0] - anchor[0])
        dest_y = target_xy[1] + scale * (contact[1] - anchor[1])
        d_dst = sample_depth(depth, dest_x, dest_y)
        if metric:
            raw = d_src / max(d_dst, 1e-6)
        else:
            raw = d_dst / max(d_src, 1e-6)
        scale = float(np.clip(raw * float(scale_factor), s_min, s_max))
    return scale, anchor, float(d_src), float(d_dst), contact


def contact_depth(mask: np.ndarray, depth: np.ndarray):
    """Median depth in the bottom band of the mask, and that band's center."""
    ys, xs = np.where(mask)
    y_cut = np.quantile(ys, 0.8)
    band = ys >= y_cut
    if not band.any():
        band = np.ones(len(ys), dtype=bool)
    values = depth[ys[band], xs[band]]
    values = values[np.isfinite(values)]
    if values.size == 0:
        d_src = 1.0
    else:
        d_src = float(np.median(values))
    contact = (float(np.median(xs[band])), float(np.median(ys[band])))
    return max(d_src, 1e-6), contact


def sample_depth(depth: np.ndarray, x: float, y: float, radius: int = 4) -> float:
    height, width = depth.shape[:2]
    cx = int(np.clip(round(x), 0, width - 1))
    cy = int(np.clip(round(y), 0, height - 1))
    y0, y1 = max(0, cy - radius), min(height, cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(width, cx + radius + 1)
    patch = depth[y0:y1, x0:x1]
    patch = patch[np.isfinite(patch)]
    if patch.size == 0:
        return 1.0
    return float(np.median(patch))


def forward_matrix(scale: float, anchor: tuple[float, float], target: tuple[float, float]) -> np.ndarray:
    """2x3 map from source pixels to destination pixels: q = s * (p - a) + t."""
    ax, ay = anchor
    tx, ty = target
    scale = float(scale)
    return np.array(
        [[scale, 0.0, tx - scale * ax], [0.0, scale, ty - scale * ay]],
        dtype=np.float64,
    )


def warp_nearest(mask: np.ndarray, forward: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    src_y, src_x, valid = _source_indices(forward, out_hw, mask.shape)
    out = np.zeros(out_hw, dtype=bool)
    out[valid] = mask[src_y[valid], src_x[valid]]
    return out


def paste_preview(image: np.ndarray, mask: np.ndarray, forward: np.ndarray) -> np.ndarray:
    canvas = image.copy()
    src_y, src_x, valid = _source_indices(forward, image.shape[:2], mask.shape)
    sampled = np.zeros(image.shape[:2], dtype=bool)
    sampled[valid] = mask[src_y[valid], src_x[valid]]
    yy, xx = np.where(sampled)
    canvas[yy, xx] = image[src_y[yy, xx], src_x[yy, xx]]
    return canvas


def visible_source(mask, scale, anchor, target, hw):
    """Source pixels whose mapped location lands inside the image."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return mask.astype(bool), 0.0
    qx = target[0] + scale * (xs - anchor[0])
    qy = target[1] + scale * (ys - anchor[1])
    height, width = hw
    keep = (qx >= -0.5) & (qx < width - 0.5) & (qy >= -0.5) & (qy < height - 0.5)
    reference = np.zeros(mask.shape, dtype=bool)
    reference[ys[keep], xs[keep]] = True
    return reference, float(keep.mean())


def shrink_scale(scale, anchor, target, bbox, hw, s_min) -> float:
    if _fits(_map_bbox(bbox, scale, anchor, target), hw):
        return float(scale)
    lo, hi = float(s_min), float(scale)
    best = float(s_min)
    for _ in range(24):
        mid = (lo + hi) / 2.0
        if _fits(_map_bbox(bbox, mid, anchor, target), hw):
            best = mid
            lo = mid
        else:
            hi = mid
    return float(best)


def shift_target(scale, anchor, target, bbox, hw):
    """Smallest translation that brings the mapped box inside the image."""
    minx, miny, maxx, maxy = _map_bbox(bbox, scale, anchor, target)
    height, width = hw
    dx = _shift_1d(minx, maxx, width - 1)
    dy = _shift_1d(miny, maxy, height - 1)
    return (float(target[0] + dx), float(target[1] + dy))


def _shift_1d(lo, hi, limit) -> float:
    if hi - lo > limit:
        return (limit / 2.0) - (lo + hi) / 2.0
    if lo < 0:
        return -lo
    if hi > limit:
        return limit - hi
    return 0.0


def _centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.where(mask)
    return float(xs.mean()), float(ys.mean())


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _map_bbox(bbox, scale, anchor, target):
    x1, y1, x2, y2 = bbox
    corners = ((x1, y1), (x2, y1), (x1, y2), (x2, y2))
    xs, ys = [], []
    for x, y in corners:
        xs.append(target[0] + scale * (x - anchor[0]))
        ys.append(target[1] + scale * (y - anchor[1]))
    return min(xs), min(ys), max(xs), max(ys)


def _fits(mapped, hw) -> bool:
    minx, miny, maxx, maxy = mapped
    height, width = hw
    return minx >= 0 and miny >= 0 and maxx <= width - 1 and maxy <= height - 1


def _source_indices(forward, out_hw, src_hw):
    """Inverse of q = s * p + b, so a destination pixel reads the source pixel."""
    out_h, out_w = out_hw
    src_h, src_w = src_hw
    scale = float(forward[0, 0])
    if abs(scale) < 1e-6:
        raise ValueError("Scale is too close to zero to warp the mask.")
    bx = float(forward[0, 2])
    by = float(forward[1, 2])
    ys, xs = np.indices((out_h, out_w))
    src_x = np.rint((xs - bx) / scale).astype(np.int32)
    src_y = np.rint((ys - by) / scale).astype(np.int32)
    valid = (src_x >= 0) & (src_x < src_w) & (src_y >= 0) & (src_y < src_h)
    return src_y, src_x, valid
