"""AnyDoor's crop math must stay inside the image even when the mask touches the border."""

import unittest

from relocate_edit.ops.anydoor_data_utils import clip_inner_box, nonempty_slice


class AnyDoorBoxTest(unittest.TestCase):
    def test_one_pixel_box_is_not_an_empty_slice(self):
        self.assertEqual(nonempty_slice(5, 5, 10), (5, 6))
        self.assertEqual(nonempty_slice(0, 9, 10), (0, 9))

    def test_inner_box_that_sticks_out_of_the_crop_is_clipped(self):
        small = (-4, 20, -3, 15)
        big = (0, 40, 0, 40)
        y1, y2, x1, x2 = clip_inner_box(small, big, (30, 30))
        self.assertEqual((y1, y2, x1, x2), (0, 20, 0, 15))

    def test_no_overlap_is_an_error_instead_of_a_wrapped_slice(self):
        with self.assertRaises(ValueError):
            clip_inner_box((0, 2, 0, 2), (20, 40, 20, 40), (10, 10))


if __name__ == "__main__":
    unittest.main()
