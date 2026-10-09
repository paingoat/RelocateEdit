"""UI callbacks. Each one returns the session plus the images that stage shows."""

from __future__ import annotations

import gradio as gr
import numpy as np

from relocate_edit.ops.mask_ops import centroid_xy, resize_mask
from relocate_edit.types import Session
from relocate_edit.utils.visualization import draw_arrow, overlay_mask


def confirm_scribble(editor_value, session: Session, pipeline):
    image, scribble = extract_scribble(editor_value)
    session.image = image
    session.scribble = scribble
    session.target_xy = None
    session.run_dir = None
    session.segment = None
    session.depth = None
    session.relocate = None
    session.inpaint = None
    session.insert = None
    view = _base_view(session)
    session.extras["base_view"] = view
    run_dir = pipeline.save_scribble(session, view)
    status = (
        f"Đã nhận ảnh {image.shape[1]}x{image.shape[0]}, nét khoanh {int(scribble.sum())} pixel. "
        f"Bấm vào ảnh để chọn điểm đích. Đã lưu vào output/{_run_name(run_dir)}."
    )
    return session, view, status


def select_target_at(session: Session, x, y, pipeline):
    # Coordinates come from a page script, not gr.Image.select. Gradio 4.44
    # leaves an invisible upload button over the picture, so that event never fires.
    if session is None or session.image is None:
        raise gr.Error("Xác nhận nét khoanh trước khi chọn điểm đích.")
    try:
        x = int(round(float(x)))
        y = int(round(float(y)))
    except (TypeError, ValueError):
        raise gr.Error("Không đọc được tọa độ cú click. Hãy bấm trực tiếp lên ảnh.") from None
    height, width = session.image.shape[:2]
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    session.target_xy = (x, y)
    session.relocate = None
    session.insert = None
    origin = _origin(session)
    view = draw_arrow(session.extras.get("base_view", _base_view(session)), origin, (x, y))
    pipeline.save_target(session, view)
    status = f"Điểm đích: ({x}, {y}). Mũi tên đi từ tâm vật tới điểm này."
    return session, view, status


def run_segment(session, prompt_mode, pipeline):
    session = pipeline.run_segment(session, prompt_mode=prompt_mode)
    result = session.segment
    return session, result.overlay, result.candidates, result.info


def run_depth(session, mode, pipeline):
    session = pipeline.run_depth(session, mode=mode)
    result = session.depth
    return session, result.colormap, result.info


def run_relocate(session, scale_factor, boundary, pipeline):
    session = pipeline.run_relocate(session, scale_factor=scale_factor, boundary=boundary)
    result = session.relocate
    return session, result.overlay, result.preview, result.info


def run_inpaint(session, dilation, refine, pipeline):
    session = pipeline.run_inpaint(session, dilation=int(dilation), refine=bool(refine))
    result = session.inpaint
    return session, result.mask_preview, result.image, result.info


def run_insert(session, steps, guidance, strength, seed, shape_control, pipeline):
    session = pipeline.run_insert(
        session,
        steps=int(steps),
        guidance=float(guidance),
        strength=float(strength),
        seed=int(seed),
        shape_control=bool(shape_control),
    )
    result = session.insert
    return session, result.cutout, result.image, result.comparison, result.info


def run_all(session, prompt_mode, depth_mode, scale_factor, boundary, dilation, refine,
            steps, guidance, strength, seed, shape_control, pipeline):
    """Yield after each stage so the panels update before the next model runs."""
    session.depth = None
    session.relocate = None
    session.inpaint = None
    session.insert = None
    yield _panels(session, "Đang chạy bước 1 — SEEM...")
    pipeline.run_segment(session, prompt_mode=prompt_mode)
    yield _panels(session, "Đã xong bước 1. Đang chạy bước 2 — depth...")
    pipeline.run_depth(session, mode=depth_mode)
    yield _panels(session, "Đã xong bước 2. Đang chạy bước 3 — dời mask...")
    pipeline.run_relocate(session, scale_factor=scale_factor, boundary=boundary)
    yield _panels(session, "Đã xong bước 3. Đang chạy bước 4 — xóa vật nguồn...")
    pipeline.run_inpaint(session, dilation=int(dilation), refine=bool(refine))
    yield _panels(session, "Đã xong bước 4. Đang chạy bước 5 — chèn vật...")
    pipeline.run_insert(
        session,
        steps=int(steps),
        guidance=float(guidance),
        strength=float(strength),
        seed=int(seed),
        shape_control=bool(shape_control),
    )
    yield _panels(session, f"Xong toàn bộ. Đã lưu trong output/{_run_name(session.run_dir)}.")


def _panels(session: Session, status: str):
    segment = session.segment
    depth = session.depth
    relocate = session.relocate
    inpaint = session.inpaint
    insert = session.insert
    return (
        session,
        None if segment is None else segment.overlay,
        None if segment is None else segment.candidates,
        None if segment is None else segment.info,
        None if depth is None else depth.colormap,
        None if depth is None else depth.info,
        None if relocate is None else relocate.overlay,
        None if relocate is None else relocate.preview,
        None if relocate is None else relocate.info,
        None if inpaint is None else inpaint.mask_preview,
        None if inpaint is None else inpaint.image,
        None if inpaint is None else inpaint.info,
        None if insert is None else insert.cutout,
        None if insert is None else insert.image,
        None if insert is None else insert.comparison,
        None if insert is None else insert.info,
        status,
    )


def _run_name(run_dir) -> str:
    from pathlib import Path
    return Path(run_dir).name if run_dir else ""


def extract_scribble(editor_value):
    if editor_value is None:
        raise ValueError("Hãy tải ảnh và vẽ vòng quanh vật.")
    background = _field(editor_value, "background")
    composite = _field(editor_value, "composite")
    layers = _field(editor_value, "layers") or []
    image = background if _is_image(background) else composite
    if not _is_image(image):
        raise ValueError("Chưa có ảnh. Hãy tải một ảnh lên khung bên trái.")
    image = _as_rgb(image)
    scribble = np.zeros(image.shape[:2], dtype=bool)
    for layer in layers:
        if not _is_image(layer):
            continue
        scribble |= _layer_mask(layer, image.shape[:2])
    if not scribble.any() and _is_image(composite) and _is_image(background):
        diff = np.abs(_as_rgb(composite).astype(np.int16) - image.astype(np.int16)).sum(axis=2)
        scribble = diff > 30
    if not scribble.any():
        raise ValueError("Chưa thấy nét vẽ. Dùng cọ đỏ để khoanh một vòng kín quanh vật.")
    return image, scribble


def _field(value, key):
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _is_image(value) -> bool:
    return isinstance(value, np.ndarray) and value.ndim >= 2 and value.size > 0


def _as_rgb(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _layer_mask(layer: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    layer = np.asarray(layer)
    if layer.ndim == 3 and layer.shape[-1] == 4:
        mask = layer[..., 3] > 10
    elif layer.ndim == 3:
        mask = layer.max(axis=2) > 10
    else:
        mask = layer > 10
    if mask.shape != out_hw:
        mask = resize_mask(mask, out_hw)
    return mask


def _base_view(session: Session) -> np.ndarray:
    return overlay_mask(session.image, session.scribble, color=(255, 50, 50), alpha=0.9)


def _origin(session: Session):
    if session.segment is not None:
        point = centroid_xy(session.segment.mask)
        if point is not None:
            return point
    point = centroid_xy(session.scribble)
    if point is None:
        height, width = session.image.shape[:2]
        return width / 2.0, height / 2.0
    return point
