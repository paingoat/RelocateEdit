"""Top-to-bottom Gradio layout: confirm the loop, click the target, then run each stage."""

from __future__ import annotations

import gradio as gr

from relocate_edit.types import Session
from relocate_edit.ui import callbacks as cb


# Gradio 4.44 paints a full-size upload button over every image (its `.center` /
# `.flex` rules override `.hidden`). Clicks hit that button, so Image.select never
# runs. The page script reads the click against the real bitmap and submits it.
_TARGET_CSS = """
#target-view { cursor: crosshair; }
#target-view img {
    object-fit: contain !important;
    cursor: crosshair !important;
}
#target-view button.hidden,
#target-view [aria-label="Remove Image"] {
    display: none !important;
    pointer-events: none !important;
}
"""

_TARGET_HEAD = """
<script>
(function () {
    if (window.__relocateTargetInstalled) return;
    window.__relocateTargetInstalled = true;

    function pointOnImage(img, evt) {
        var rect = img.getBoundingClientRect();
        var nw = img.naturalWidth;
        var nh = img.naturalHeight;
        if (!nw || !nh || !rect.width || !rect.height) return null;
        var scale = Math.min(rect.width / nw, rect.height / nh);
        var dw = nw * scale;
        var dh = nh * scale;
        var left = rect.left + (rect.width - dw) / 2;
        var top = rect.top + (rect.height - dh) / 2;
        var x = (evt.clientX - left) / scale;
        var y = (evt.clientY - top) / scale;
        if (x < 0 || y < 0 || x >= nw || y >= nh) return null;
        return [Math.round(x), Math.round(y)];
    }

    function largestImage(root) {
        var imgs = root.querySelectorAll("img");
        var img = null;
        var best = 0;
        for (var i = 0; i < imgs.length; i++) {
            var rect = imgs[i].getBoundingClientRect();
            var area = rect.width * rect.height;
            if (area > best) {
                best = area;
                img = imgs[i];
            }
        }
        return img;
    }

    function pickButton() {
        // elem_id is copied onto the block wrapper and the <button>, so the
        // first match is the wrapper. Click the real button or the event is lost.
        var nodes = document.querySelectorAll("#target-pick");
        for (var i = 0; i < nodes.length; i++) {
            if (nodes[i].tagName === "BUTTON" && !nodes[i].disabled) return nodes[i];
        }
        for (var j = 0; j < nodes.length; j++) {
            var inner = nodes[j].querySelector("button:not([disabled])");
            if (inner) return inner;
        }
        return null;
    }

    document.addEventListener("click", function (evt) {
        var root = document.getElementById("target-view");
        if (!root || !root.contains(evt.target)) return;
        var img = largestImage(root);
        if (!img) return;
        var point = pointOnImage(img, evt);
        if (!point) return;
        var pick = pickButton();
        if (!pick) return;
        var now = Date.now();
        if (window.__relocatePickAt && now - window.__relocatePickAt < 200) {
            evt.preventDefault();
            evt.stopPropagation();
            return;
        }
        window.__relocatePickAt = now;
        window.__relocateTargetXY = point;
        evt.preventDefault();
        evt.stopPropagation();
        pick.click();
    }, true);
})();
</script>
"""

_PICK_JS = """(session, x, y) => {
    const point = window.__relocateTargetXY;
    if (!point) return [session, x, y];
    return [session, point[0], point[1]];
}"""


def build_demo(pipeline):
    cfg = pipeline.config

    def _bind(fn):
        def wrapped(*args):
            try:
                return fn(*args, pipeline)
            except Exception as exc:
                raise gr.Error(str(exc)) from exc
        return wrapped

    with gr.Blocks(title="RelocateEdit", css=_TARGET_CSS, head=_TARGET_HEAD) as demo:
        session = gr.State(Session())
        gr.Markdown(
            """
# RelocateEdit

Di chuyển một vật trong ảnh. Làm lần lượt từ trên xuống dưới.

1. Tải ảnh và **khoanh một vòng kín** quanh vật nguồn, rồi bấm xác nhận.
2. **Bấm vào ảnh** ở điểm muốn đặt vật. Mũi tên được vẽ từ tâm vòng khoanh tới điểm đó.
3. Chạy từng giai đoạn để xem mask, depth, mask đích, nền đã xóa, và ảnh cuối. Hoặc bấm chạy toàn bộ.
            """.strip()
        )

        gr.Markdown("## Bước A — khoanh vật nguồn")
        with gr.Row():
            editor = gr.ImageEditor(
                label="Tải ảnh và vẽ vòng quanh vật",
                type="numpy",
                sources=["upload"],
                image_mode="RGB",
                brush=gr.Brush(default_size=8, colors=["#ff2d2d"], default_color="#ff2d2d", color_mode="fixed"),
                transforms=[],
                layers=True,
                height=520,
            )
            with gr.Column():
                confirm = gr.Button("Xác nhận nét khoanh", variant="primary")
                status = gr.Textbox(label="Trạng thái", lines=2, interactive=False)

        gr.Markdown("## Bước B — chọn điểm đích")
        with gr.Row():
            target_view = gr.Image(
                label="Bấm vào ảnh để chọn điểm đích",
                type="numpy",
                interactive=False,
                show_download_button=False,
                show_fullscreen_button=False,
                elem_id="target-view",
                height=520,
            )
        click_x = gr.Number(value=0, visible=False)
        click_y = gr.Number(value=0, visible=False)
        pick_target = gr.Button(elem_id="target-pick", visible=False)
        confirm.click(cb.confirm_scribble, [editor, session], [session, target_view, status])
        pick_target.click(
            cb.select_target_at,
            [session, click_x, click_y],
            [session, target_view, status],
            js=_PICK_JS,
            show_progress="hidden",
        )

        run_all_button = gr.Button("Chạy toàn bộ pipeline", variant="primary")

        gr.Markdown("## 1. SEEM — chọn vật nguồn")
        prompt_mode = gr.Radio(
            [("Vòng khoanh quanh vật", "loop"), ("Tô trực tiếp lên vật", "stroke")],
            value=cfg.seem.prompt_mode,
            label="Cách tạo prompt",
        )
        segment_button = gr.Button("Chạy bước 1")
        with gr.Row():
            segment_overlay = gr.Image(label="Mask nguồn", type="numpy")
            segment_candidates = gr.Image(label="3 ứng viên cao nhất", type="numpy")
        segment_info = gr.JSON(label="Thông tin SEEM")

        gr.Markdown("## 2. Depth Anything V2 — bản đồ độ sâu")
        depth_mode = gr.Radio(
            [("Tương đối (disparity, lớn = gần)", "relative"),
             ("Metric trong nhà, mét", "metric_indoor"),
             ("Metric ngoài trời, mét", "metric_outdoor")],
            value=cfg.depth.mode,
            label="Loại depth",
        )
        depth_button = gr.Button("Chạy bước 2")
        depth_image = gr.Image(label="Depth", type="numpy")
        depth_info = gr.JSON(label="Thông tin depth")

        gr.Markdown("## 3. Dời mask — scale theo depth")
        with gr.Row():
            scale_factor = gr.Slider(0.2, 5.0, value=cfg.relocate.scale_factor, step=0.05, label="Hệ số scale thêm")
            boundary = gr.Radio(
                [("Đẩy vào trong ảnh", "shift"), ("Thu nhỏ cho vừa", "shrink"), ("Cắt phần tràn", "clip")],
                value=cfg.relocate.boundary,
                label="Khi mask vượt biên",
            )
        relocate_button = gr.Button("Chạy bước 3")
        with gr.Row():
            relocate_overlay = gr.Image(label="Mask đích", type="numpy")
            relocate_preview = gr.Image(label="Dán thử, chưa sinh ảnh", type="numpy")
        relocate_info = gr.JSON(label="Thông tin dời mask")

        gr.Markdown("## 4. LaMa — xóa vật nguồn")
        with gr.Row():
            dilation = gr.Slider(0, 40, value=cfg.lama.dilation, step=1, label="Nở mask (pixel)")
            refine = gr.Checkbox(value=cfg.lama.refine, label="Feature refinement cho ảnh lớn")
        inpaint_button = gr.Button("Chạy bước 4")
        with gr.Row():
            inpaint_mask = gr.Image(label="Mask đã nở", type="numpy")
            inpaint_clean = gr.Image(label="Nền đã xóa vật", type="numpy")
        inpaint_info = gr.JSON(label="Thông tin LaMa")

        gr.Markdown("## 5. AnyDoor — chèn vật vào mask đích")
        with gr.Row():
            steps = gr.Slider(1, 100, value=cfg.anydoor.steps, step=1, label="Số bước")
            guidance = gr.Slider(0.5, 15.0, value=cfg.anydoor.guidance, step=0.1, label="Guidance")
            strength = gr.Slider(0.0, 2.0, value=cfg.anydoor.strength, step=0.05, label="Control strength")
        with gr.Row():
            seed = gr.Number(value=cfg.anydoor.seed, precision=0, label="Seed")
            shape_control = gr.Checkbox(value=cfg.anydoor.shape_control, label="Bám hình dạng mask đích")
        insert_button = gr.Button("Chạy bước 5")
        with gr.Row():
            insert_cutout = gr.Image(label="Vật nguồn", type="numpy")
            insert_result = gr.Image(label="Ảnh kết quả", type="numpy")
        insert_compare = gr.Image(label="Trước / sau", type="numpy")
        insert_info = gr.JSON(label="Thông tin AnyDoor")

        segment_button.click(_bind(cb.run_segment), [session, prompt_mode], [session, segment_overlay, segment_candidates, segment_info])
        depth_button.click(_bind(cb.run_depth), [session, depth_mode], [session, depth_image, depth_info])
        relocate_button.click(_bind(cb.run_relocate), [session, scale_factor, boundary], [session, relocate_overlay, relocate_preview, relocate_info])
        inpaint_button.click(_bind(cb.run_inpaint), [session, dilation, refine], [session, inpaint_mask, inpaint_clean, inpaint_info])
        insert_button.click(
            _bind(cb.run_insert),
            [session, steps, guidance, strength, seed, shape_control],
            [session, insert_cutout, insert_result, insert_compare, insert_info],
        )
        run_all_button.click(
            _bind(cb.run_all),
            [session, prompt_mode, depth_mode, scale_factor, boundary, dilation, refine,
             steps, guidance, strength, seed, shape_control],
            [session, segment_overlay, segment_candidates, segment_info,
             depth_image, depth_info,
             relocate_overlay, relocate_preview, relocate_info,
             inpaint_mask, inpaint_clean, inpaint_info,
             insert_cutout, insert_result, insert_compare, insert_info],
        )
    return demo
