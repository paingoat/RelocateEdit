from __future__ import annotations

"""Data passed between pipeline stages.

Images are RGB uint8 arrays of shape (H, W, 3). Masks are boolean arrays of
shape (H, W). `info` dictionaries are JSON-serializable and are what the UI
shows next to each stage.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SegmentResult:
    mask: np.ndarray
    bbox: tuple
    overlay: np.ndarray
    candidates: np.ndarray
    info: dict


@dataclass
class DepthResult:
    depth: np.ndarray
    colormap: np.ndarray
    metric: bool
    info: dict


@dataclass
class RelocateResult:
    mask: np.ndarray
    reference_mask: np.ndarray
    overlay: np.ndarray
    preview: np.ndarray
    scale: float
    info: dict


@dataclass
class InpaintResult:
    image: np.ndarray
    dilated_mask: np.ndarray
    mask_preview: np.ndarray
    info: dict


@dataclass
class InsertResult:
    image: np.ndarray
    cutout: np.ndarray
    comparison: np.ndarray
    info: dict


@dataclass
class Session:
    """One user session. Later stages are cleared when an earlier input changes."""

    image: np.ndarray | None = None
    scribble: np.ndarray | None = None
    target_xy: tuple | None = None
    run_dir: str | None = None
    segment: SegmentResult | None = None
    depth: DepthResult | None = None
    relocate: RelocateResult | None = None
    inpaint: InpaintResult | None = None
    insert: InsertResult | None = None
    extras: dict = field(default_factory=dict)
