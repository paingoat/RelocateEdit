"""Run the five stages, and forget downstream results when an input changes."""

from __future__ import annotations

from relocate_edit.ops.mask_ops import mask_bbox
from relocate_edit.ops.relocation import relocate
from relocate_edit.registry import ModelRegistry
from relocate_edit.types import DepthResult, InpaintResult, InsertResult, RelocateResult, SegmentResult
from relocate_edit.utils.io import new_run_dir, save_stage
from relocate_edit.utils.visualization import mask_preview, object_cutout, overlay_mask, side_by_side


class RelocatePipeline:
    def __init__(self, config):
        self.config = config
        self.registry = ModelRegistry(config)

    def preload(self):
        self.registry.preload()

    def run_segment(self, session, prompt_mode=None):
        self._require_image(session)
        if session.scribble is None or not session.scribble.any():
            raise ValueError("Draw a loop around the object, then confirm it.")
        segmenter = self.registry.get("seem")
        mask, bbox, overlay, candidates, info = segmenter.segment(session.image, session.scribble, prompt_mode)
        session.segment = SegmentResult(mask, bbox, overlay, candidates, info)
        session.relocate = None
        session.inpaint = None
        session.insert = None
        save_stage(self._run_dir(session), "01_segment", {
            "overlay": overlay,
            "candidates": candidates,
            "mask": mask,
        }, info)
        return session

    def run_depth(self, session, mode=None):
        self._require_image(session)
        estimator = self.registry.activate("depth")
        depth, colormap, metric, info = estimator.estimate(session.image, mode=mode)
        session.depth = DepthResult(depth, colormap, metric, info)
        session.relocate = None
        session.insert = None
        save_stage(self._run_dir(session), "02_depth", {"colormap": colormap}, info)
        return session

    def run_relocate(self, session, scale_factor=None, boundary=None):
        self._require_image(session)
        if session.segment is None:
            raise ValueError("Run SEEM before moving the mask.")
        if session.depth is None:
            raise ValueError("Run depth estimation before moving the mask.")
        if session.target_xy is None:
            raise ValueError("Click the image to choose the target point.")
        cfg = self.config.relocate
        moved = relocate(
            session.image,
            session.segment.mask,
            session.depth.depth,
            session.target_xy,
            scale_factor=cfg.scale_factor if scale_factor is None else float(scale_factor),
            s_min=cfg.s_min,
            s_max=cfg.s_max,
            iterations=cfg.iterations,
            boundary=cfg.boundary if boundary is None else boundary,
            metric=session.depth.metric,
        )
        overlay = overlay_mask(session.image, moved["mask"], color=(70, 140, 255))
        info = {
            "scale": round(moved["scale"], 4),
            "boundary": moved["boundary"],
            "depth_src": round(moved["depth_src"], 4),
            "depth_dst": round(moved["depth_dst"], 4),
            "visible_ratio": round(moved["visible_ratio"], 4),
            "anchor_xy": [round(v, 1) for v in moved["anchor_xy"]],
            "placed_xy": [round(v, 1) for v in moved["placed_xy"]],
            "target_xy": [round(float(v), 1) for v in session.target_xy],
            "target_bbox_xyxy": list(mask_bbox(moved["mask"])) if moved["mask"].any() else None,
        }
        session.relocate = RelocateResult(
            moved["mask"], moved["reference_mask"], overlay, moved["preview"], moved["scale"], info,
        )
        session.insert = None
        save_stage(self._run_dir(session), "03_relocate", {
            "overlay": overlay,
            "preview": moved["preview"],
            "mask": moved["mask"],
        }, info)
        return session

    def run_inpaint(self, session, dilation=None, refine=None):
        self._require_image(session)
        if session.segment is None:
            raise ValueError("Run SEEM before erasing the source object.")
        inpainter = self.registry.get("lama")
        clean, hole, info = inpainter.inpaint(session.image, session.segment.mask, dilation=dilation, refine=refine)
        preview = mask_preview(session.image, hole)
        session.inpaint = InpaintResult(clean, hole, preview, info)
        session.insert = None
        save_stage(self._run_dir(session), "04_inpaint", {
            "clean": clean,
            "dilated_mask": preview,
        }, info)
        return session

    def run_insert(self, session, steps=None, guidance=None, strength=None, seed=None, shape_control=None):
        self._require_image(session)
        if session.segment is None or session.relocate is None or session.inpaint is None:
            raise ValueError("Run SEEM, relocate, and LaMa before inserting the object.")
        inserter = self.registry.get("anydoor")
        reference_mask = session.relocate.reference_mask
        if int(reference_mask.sum()) < 10:
            reference_mask = session.segment.mask
        result = inserter.insert(
            session.image,
            reference_mask,
            session.inpaint.image,
            session.relocate.mask,
            steps=steps,
            guidance=guidance,
            strength=strength,
            seed=seed,
            shape_control=shape_control,
        )
        cutout = object_cutout(session.image, reference_mask)
        comparison = side_by_side(session.image, result)
        info = {
            "steps": int(self.config.anydoor.steps if steps is None else steps),
            "guidance": float(self.config.anydoor.guidance if guidance is None else guidance),
            "strength": float(self.config.anydoor.strength if strength is None else strength),
            "seed": int(self.config.anydoor.seed if seed is None else seed),
            "shape_control": self.config.anydoor.shape_control if shape_control is None else bool(shape_control),
            "scale": session.relocate.scale,
        }
        session.insert = InsertResult(result, cutout, comparison, info)
        save_stage(self._run_dir(session), "05_insert", {
            "result": result,
            "cutout": cutout,
            "comparison": comparison,
        }, info)
        return session

    def run_all(self, session, **params):
        self.run_segment(session, prompt_mode=params.get("prompt_mode"))
        self.run_depth(session, mode=params.get("depth_mode"))
        self.run_relocate(session, scale_factor=params.get("scale_factor"), boundary=params.get("boundary"))
        self.run_inpaint(session, dilation=params.get("dilation"), refine=params.get("refine"))
        self.run_insert(
            session,
            steps=params.get("steps"),
            guidance=params.get("guidance"),
            strength=params.get("strength"),
            seed=params.get("seed"),
            shape_control=params.get("shape_control"),
        )
        return session

    def _run_dir(self, session) -> str:
        if not session.run_dir:
            session.run_dir = new_run_dir(self.config.outputs_dir)
        return session.run_dir

    @staticmethod
    def _require_image(session):
        if session.image is None:
            raise ValueError("Upload an image and confirm the scribble first.")
