"""Geometry helpers ported from AnyDoor's `datasets/data_utils.py`.

Only the functions used by the inference collage are kept. Masks are 0/1.
"""

from __future__ import annotations

import numpy as np


def sobel(img, mask, thresh=50):
    """High-frequency collage of the reference object."""
    import cv2
    height, width = img.shape[0], img.shape[1]
    img = cv2.resize(img, (256, 256))
    mask = (cv2.resize(mask.astype(np.float32), (256, 256)) > 0.5).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)

    sobel_x = cv2.convertScaleAbs(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3))
    sobel_y = cv2.convertScaleAbs(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3))
    edges = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
    edges = np.max(edges, -1) * mask
    edges[edges < thresh] = 0.0
    edges = np.stack([edges, edges, edges], -1)
    edges = (edges.astype(np.float32) / 255 * img.astype(np.float32)).astype(np.uint8)
    return cv2.resize(edges, (width, height))


def expand_image_mask(image, mask, ratio=1.4):
    height, width = image.shape[0], image.shape[1]
    out_h, out_w = int(height * ratio), int(width * ratio)
    pad_y = int((out_h - height) // 2)
    pad_x = int((out_w - width) // 2)
    image = np.pad(
        image,
        ((pad_y, out_h - height - pad_y), (pad_x, out_w - width - pad_x), (0, 0)),
        "constant",
        constant_values=255,
    )
    mask = np.pad(
        mask,
        ((pad_y, out_h - height - pad_y), (pad_x, out_w - width - pad_x)),
        "constant",
        constant_values=0,
    )
    return image, mask


def get_bbox_from_mask(mask):
    height, width = mask.shape[0], mask.shape[1]
    if mask.sum() < 10:
        return 0, height, 0, width
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return (int(y1), int(y2), int(x1), int(x2))


def expand_bbox(mask, yyxx, ratio=(1.2, 2.0), min_crop=0):
    y1, y2, x1, x2 = yyxx
    ratio = np.random.randint(int(ratio[0] * 10), int(ratio[1] * 10)) / 10
    height, width = mask.shape[0], mask.shape[1]
    xc, yc = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    box_h = max(ratio * (y2 - y1 + 1), min_crop)
    box_w = max(ratio * (x2 - x1 + 1), min_crop)
    x1 = int(xc - box_w * 0.5)
    x2 = int(xc + box_w * 0.5)
    y1 = int(yc - box_h * 0.5)
    y2 = int(yc + box_h * 0.5)
    return (max(0, y1), min(height, y2), max(0, x1), min(width, x2))


def box2squre(image, box):
    height, width = image.shape[0], image.shape[1]
    y1, y2, x1, x2 = box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    box_h, box_w = y2 - y1, x2 - x1
    if box_h >= box_w:
        x1 = cx - box_h // 2
        x2 = cx + box_h // 2
    else:
        y1 = cy - box_w // 2
        y2 = cy + box_w // 2
    return (max(0, y1), min(height, y2), max(0, x1), min(width, x2))


def pad_to_square(image, pad_value=255, random=False):
    height, width = image.shape[0], image.shape[1]
    if height == width:
        return image
    extra = abs(height - width)
    pad_1 = int(np.random.randint(0, extra)) if random else int(extra / 2)
    pad_2 = extra - pad_1
    if height > width:
        pad = ((0, 0), (pad_1, pad_2), (0, 0))
    else:
        pad = ((pad_1, pad_2), (0, 0), (0, 0))
    if image.ndim == 2:
        pad = pad[:2]
    return np.pad(image, pad, "constant", constant_values=pad_value)


def box_in_box(small_box, big_box):
    y1, y2, x1, x2 = small_box
    y1_b, _, x1_b, _ = big_box
    return (y1 - y1_b, y2 - y1_b, x1 - x1_b, x2 - x1_b)


def clip_inner_box(small_box, big_box, crop_hw):
    """Rectangle of ``small_box`` inside the crop, in crop coordinates.

    ``box2squre`` clips the crop to the image, so the target box can stick out
    by a pixel. A negative start index wraps around in NumPy instead of
    meaning "before the crop".
    """
    y1, y2, x1, x2 = box_in_box(small_box, big_box)
    crop_h, crop_w = crop_hw
    y1 = max(0, int(y1))
    x1 = max(0, int(x1))
    y2 = min(int(crop_h), int(y2))
    x2 = min(int(crop_w), int(x2))
    if y2 - y1 < 2 or x2 - x1 < 2:
        raise ValueError(
            "The target mask is too close to the image border for AnyDoor to build a collage. "
            "Choose a point a little further inside."
        )
    return y1, y2, x1, x2


def nonempty_slice(start, end, limit):
    """Half-open slice. A one-pixel box has equal ends and would otherwise be empty."""
    start = max(0, int(start))
    end = int(end)
    if end <= start:
        end = start + 1
    return start, min(int(limit), end)
