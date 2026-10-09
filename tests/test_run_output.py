"""A full run saves each step as it finishes, and the UI yields between stages."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from relocate_edit.utils.io import new_run_dir, save_step


class SaveStepTest(unittest.TestCase):
    def test_timestamped_step_folder(self):
        image = np.zeros((4, 6, 3), dtype=np.uint8)
        mask = np.zeros((4, 6), dtype=bool)
        mask[1:3, 2:4] = True
        with tempfile.TemporaryDirectory() as tmp:
            run = new_run_dir(tmp)
            self.assertRegex(Path(run).name, r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")
            save_step(run, "A", {"image": image, "scribble": mask}, {"scribble_px": 4})
            folder = Path(run) / "A"
            self.assertTrue((folder / "image.png").is_file())
            self.assertTrue((folder / "scribble.png").is_file())
            self.assertIn("scribble_px", (folder / "info.json").read_text(encoding="utf-8"))


class StreamRunTest(unittest.TestCase):
    def test_each_stage_is_yielded_before_the_next_starts(self):
        try:
            from relocate_edit.types import Session
            from relocate_edit.ui.callbacks import run_all
        except ModuleNotFoundError as exc:
            self.skipTest(str(exc))
        order = []

        class Pipeline:
            def run_segment(self, session, prompt_mode=None):
                order.append("1")
                session.segment = SimpleNamespace(overlay="mask", candidates="cands", info={"step": 1})
                return session

            def run_depth(self, session, mode=None):
                order.append("2")
                self.assert_previous(order, "1")
                session.depth = SimpleNamespace(colormap="depth", info={"step": 2})
                return session

            def run_relocate(self, session, scale_factor=None, boundary=None):
                order.append("3")
                session.relocate = SimpleNamespace(overlay="moved", preview="paste", info={"step": 3})
                return session

            def run_inpaint(self, session, dilation=None, refine=None):
                order.append("4")
                session.inpaint = SimpleNamespace(mask_preview="hole", image="clean", info={"step": 4})
                return session

            def run_insert(self, session, **kwargs):
                order.append("5")
                session.insert = SimpleNamespace(
                    cutout="cut", image="result", comparison="pair", info={"step": 5},
                )
                session.run_dir = "output/2026-10-09_12-00-00"
                return session

            @staticmethod
            def assert_previous(seen, step):
                assert step in seen

        session = Session()
        session.image = np.zeros((4, 4, 3), dtype=np.uint8)
        session.scribble = np.ones((4, 4), dtype=bool)
        session.target_xy = (1, 1)
        updates = list(run_all(
            session, "loop", "relative", 1.0, "shift", 15, True,
            30, 4.5, 1.0, 42, True, Pipeline(),
        ))
        self.assertEqual(order, ["1", "2", "3", "4", "5"])
        self.assertEqual(len(updates), 6)
        self.assertIsNone(updates[0][1])
        self.assertEqual(updates[1][1], "mask")
        self.assertIsNone(updates[1][4])
        self.assertEqual(updates[2][4], "depth")
        self.assertEqual(updates[3][6], "moved")
        self.assertEqual(updates[4][10], "clean")
        self.assertEqual(updates[5][13], "result")
        self.assertIn("Xong toàn bộ", updates[5][-1])

    def test_full_run_rejects_a_missing_target_before_any_model(self):
        try:
            from relocate_edit.types import Session
            from relocate_edit.ui.callbacks import run_all
        except ModuleNotFoundError as exc:
            self.skipTest(str(exc))
        started = []

        class Pipeline:
            def run_segment(self, session, prompt_mode=None):
                started.append("1")

        session = Session()
        session.image = np.zeros((4, 4, 3), dtype=np.uint8)
        session.scribble = np.ones((4, 4), dtype=bool)
        with self.assertRaises(ValueError):
            next(run_all(
                session, "loop", "relative", 1.0, "shift", 15, True,
                30, 4.5, 1.0, 42, True, Pipeline(),
            ))
        self.assertEqual(started, [])

    def test_a_new_photo_clears_the_target_until_the_next_click(self):
        try:
            from relocate_edit.types import Session
            from relocate_edit.ui.callbacks import confirm_scribble, select_target_at
        except ModuleNotFoundError as exc:
            self.skipTest(str(exc))
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        layer = np.zeros((20, 30, 4), dtype=np.uint8)
        layer[4:12, 6:16, 3] = 255
        editor = {"background": image, "layers": [layer], "composite": None}
        session = Session()
        session.target_xy = (2, 2)
        session.relocate = object()
        pipeline = SimpleNamespace(
            save_scribble=lambda _session, _view: "output/run",
            save_target=lambda _session, _view: None,
        )
        session, _, _ = confirm_scribble(editor, session, pipeline)
        self.assertIsNone(session.target_xy)
        self.assertIsNone(session.relocate)
        session, _, status = select_target_at(session, 9.2, 7.6, pipeline)
        self.assertEqual(session.target_xy, (9, 8))
        self.assertIn("(9, 8)", status)


if __name__ == "__main__":
    unittest.main()
