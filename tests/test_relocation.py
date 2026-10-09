"""Depth scale and the three ways a mask can meet the image border."""

import unittest

import numpy as np

from relocate_edit.ops.relocation import relocate


def _block(shape, y0, y1, x0, x1):
    mask = np.zeros(shape, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _centroid(mask):
    ys, xs = np.where(mask)
    return float(xs.mean()), float(ys.mean())


class RelocationTest(unittest.TestCase):
    def test_flat_depth_keeps_scale_and_moves_centroid(self):
        image = np.zeros((100, 120, 3), dtype=np.uint8)
        mask = _block((100, 120), 20, 40, 30, 50)
        depth = np.ones((100, 120), dtype=np.float32)
        moved = relocate(image, mask, depth, (70, 25), scale_factor=1.0, boundary="clip")
        self.assertAlmostEqual(moved["scale"], 1.0, places=2)
        cx, cy = _centroid(moved["mask"])
        self.assertAlmostEqual(cx, 70, delta=1.5)
        self.assertAlmostEqual(cy, 25, delta=1.5)
        self.assertGreater(moved["visible_ratio"], 0.95)

    def test_relative_depth_grows_when_destination_is_closer(self):
        image = np.zeros((80, 80, 3), dtype=np.uint8)
        mask = _block((80, 80), 30, 50, 10, 30)
        depth = np.ones((80, 80), dtype=np.float32)
        depth[:, 40:] = 2.0
        moved = relocate(image, mask, depth, (60, 40), scale_factor=1.0, boundary="clip", metric=False)
        self.assertAlmostEqual(moved["scale"], 2.0, delta=0.15)

    def test_metric_depth_grows_when_destination_is_closer(self):
        image = np.zeros((80, 80, 3), dtype=np.uint8)
        mask = _block((80, 80), 30, 50, 10, 30)
        depth = np.full((80, 80), 2.0, dtype=np.float32)
        depth[:, 40:] = 1.0
        moved = relocate(image, mask, depth, (60, 40), scale_factor=1.0, boundary="clip", metric=True)
        self.assertAlmostEqual(moved["scale"], 2.0, delta=0.15)

    def test_clip_drops_pixels_past_the_border(self):
        image = np.zeros((80, 80, 3), dtype=np.uint8)
        mask = _block((80, 80), 30, 50, 30, 50)
        depth = np.ones((80, 80), dtype=np.float32)
        moved = relocate(image, mask, depth, (2, 2), scale_factor=1.0, boundary="clip")
        self.assertLess(moved["visible_ratio"], 0.9)
        self.assertLess(moved["reference_mask"].sum(), mask.sum())

    def test_shift_brings_the_object_back_inside(self):
        image = np.zeros((80, 80, 3), dtype=np.uint8)
        mask = _block((80, 80), 30, 50, 30, 50)
        depth = np.ones((80, 80), dtype=np.float32)
        moved = relocate(image, mask, depth, (2, 2), scale_factor=1.0, boundary="shift")
        self.assertGreater(moved["visible_ratio"], 0.95)
        ys, xs = np.where(moved["mask"])
        self.assertGreaterEqual(xs.min(), 0)
        self.assertGreaterEqual(ys.min(), 0)
        self.assertLess(xs.max(), 80)
        self.assertLess(ys.max(), 80)

    def test_shrink_reduces_scale_until_the_object_fits(self):
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        mask = _block((50, 50), 10, 40, 10, 40)
        depth = np.ones((50, 50), dtype=np.float32)
        moved = relocate(
            image, mask, depth, (25, 25),
            scale_factor=5.0, s_min=0.2, s_max=5.0, boundary="shrink",
        )
        self.assertLess(moved["scale"], 5.0)
        ys, xs = np.where(moved["mask"])
        self.assertGreaterEqual(xs.min(), 0)
        self.assertLess(xs.max(), 50)
        self.assertGreaterEqual(ys.min(), 0)
        self.assertLess(ys.max(), 50)


if __name__ == "__main__":
    unittest.main()
