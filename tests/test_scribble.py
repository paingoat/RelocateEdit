"""The loop around an object must become an interior prompt, not the outline."""

import unittest

import numpy as np

from relocate_edit.ops.scribble import build_prompt, fill_loop


class FillLoopTest(unittest.TestCase):
    def test_closed_ring_fills_and_prompt_is_inside(self):
        stroke = np.zeros((80, 100), dtype=bool)
        stroke[10:14, 15:70] = True
        stroke[50:54, 15:70] = True
        stroke[10:54, 15:19] = True
        stroke[10:54, 66:70] = True

        region = fill_loop(stroke)
        self.assertGreater(region.sum(), stroke.sum() * 3)
        x1, y1, x2, y2 = 15, 10, 69, 53
        self.assertTrue(region[30, 40])
        self.assertFalse(region[0, 0])

        prompt, points = build_prompt(region, dilation=3, extra_points=3)
        px, py = points[0]
        self.assertTrue(region[py, px])
        self.assertGreater(py, y1)
        self.assertLess(py, y2)
        self.assertGreater(px, x1)
        self.assertLess(px, x2)
        # The outline itself is not the only positive evidence.
        self.assertTrue((prompt & ~stroke).any())


if __name__ == "__main__":
    unittest.main()
