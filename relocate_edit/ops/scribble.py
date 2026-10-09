"""Turn a rough loop into the positive prompt SEEM was trained to see.

SEEM's stroke prompt is a set of pixels that lie on the object. A loop drawn
around the object is the opposite of that: those pixels are background. This
module fills the loop, then places a small blob at the interior point farthest
from the boundary, which is the same reduction SEEM uses for a box prompt.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes, distance_transform_edt


def fill_loop(stroke: np.ndarray) -> np.ndarray:
    """Return the region enclosed by a rough closed scribble."""
    stroke = stroke.astype(bool)
    if int(stroke.sum()) < 3:
        raise ValueError("The scribble is empty. Draw a loop around the object.")
    closed = binary_closing(stroke, structure=np.ones((5, 5), dtype=bool), iterations=2)
    filled = binary_fill_holes(closed)
    # A gap in the loop leaves nothing to fill. Fall back to the convex hull.
    if filled.sum() < max(int(stroke.sum()) * 3, int(stroke.sum()) + 50):
        hull = _fill_convex_hull(stroke)
        if hull.sum() > filled.sum():
            filled = hull
    return filled.astype(bool)


def build_prompt(region: np.ndarray, dilation: int = 3, extra_points: int = 4):
    """Positive prompt mask plus the (x, y) points used to draw it.

    The main blob sits on the distance-transform maximum of an eroded region,
    then a few more interior points are added so a loose loop still hits the
    object instead of only its center pixel.
    """
    region = region.astype(bool)
    if not region.any():
        raise ValueError("The filled scribble region is empty.")
    radius = max(int(dilation), 1)
    eroded = _erode(region, radius)
    work = eroded if eroded.any() else region
    distance = distance_transform_edt(work)
    cy, cx = np.unravel_index(int(distance.argmax()), distance.shape)
    prompt = np.zeros(region.shape, dtype=bool)
    _stamp(prompt, int(cy), int(cx), radius)
    points = [(int(cx), int(cy))]

    suppressed = distance.copy()
    _zero_disk(suppressed, int(cy), int(cx), max(radius * 4, 8))
    for _ in range(int(extra_points)):
        index = int(suppressed.argmax())
        if suppressed.flat[index] <= 0:
            break
        y, x = np.unravel_index(index, suppressed.shape)
        _stamp(prompt, int(y), int(x), max(1, radius // 2))
        points.append((int(x), int(y)))
        _zero_disk(suppressed, int(y), int(x), max(radius * 4, 8))
    # Keep the prompt inside the region so the outline itself is never positive.
    prompt &= region
    if not prompt.any():
        prompt[cy, cx] = True
    return prompt, points


def stroke_points(stroke: np.ndarray, limit: int = 32) -> list[tuple[int, int]]:
    ys, xs = np.where(stroke.astype(bool))
    if len(xs) == 0:
        return []
    step = max(1, len(xs) // limit)
    return [(int(x), int(y)) for x, y in zip(xs[::step], ys[::step])]


def _erode(mask: np.ndarray, radius: int) -> np.ndarray:
    from scipy.ndimage import binary_erosion

    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    footprint = xx * xx + yy * yy <= radius * radius
    return binary_erosion(mask, structure=footprint)


def _stamp(mask: np.ndarray, y: int, x: int, radius: int) -> None:
    y0, y1 = max(0, y - radius), min(mask.shape[0], y + radius + 1)
    x0, x1 = max(0, x - radius), min(mask.shape[1], x + radius + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask[y0:y1, x0:x1] |= (yy - y) ** 2 + (xx - x) ** 2 <= radius ** 2


def _zero_disk(values: np.ndarray, y: int, x: int, radius: int) -> None:
    y0, y1 = max(0, y - radius), min(values.shape[0], y + radius + 1)
    x0, x1 = max(0, x - radius), min(values.shape[1], x + radius + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    values[y0:y1, x0:x1][(yy - y) ** 2 + (xx - x) ** 2 <= radius ** 2] = 0


def _fill_convex_hull(stroke: np.ndarray) -> np.ndarray:
    from scipy.spatial import ConvexHull

    ys, xs = np.where(stroke)
    if len(xs) < 3:
        return stroke.copy()
    points = np.stack([xs, ys], axis=1).astype(np.float64)
    try:
        hull = ConvexHull(points)
    except Exception:
        return stroke.copy()
    polygon = points[hull.vertices]
    return _raster_polygon(polygon, stroke.shape)


def _raster_polygon(polygon: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Even-odd fill of a convex polygon. `polygon` is (N, 2) in (x, y)."""
    height, width = shape
    out = np.zeros(shape, dtype=bool)
    xs = polygon[:, 0]
    ys = polygon[:, 1]
    y_min = max(int(np.floor(ys.min())), 0)
    y_max = min(int(np.ceil(ys.max())), height - 1)
    count = len(polygon)
    for y in range(y_min, y_max + 1):
        crossings = []
        for i in range(count):
            x1, y1 = xs[i], ys[i]
            x2, y2 = xs[(i + 1) % count], ys[(i + 1) % count]
            if (y1 <= y < y2) or (y2 <= y < y1):
                if y2 == y1:
                    continue
                crossings.append(x1 + (y - y1) * (x2 - x1) / (y2 - y1))
        crossings.sort()
        for start, end in zip(crossings[0::2], crossings[1::2]):
            x0 = max(int(np.ceil(start)), 0)
            x1 = min(int(np.floor(end)), width - 1)
            if x1 >= x0:
                out[y, x0:x1 + 1] = True
    return out
