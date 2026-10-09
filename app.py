"""Launch the RelocateEdit Gradio app.

Example:
    python app.py --share --offload
"""

from __future__ import annotations

import argparse
from pathlib import Path

from relocate_edit.config import ROOT, load_config
from relocate_edit.ui.gradio_app import build_demo
from relocate_edit.pipeline import RelocatePipeline


def _patch_gradio_schema():
    """Gradio 4.44 walks API schemas and assumes every node is a dict.

    Pydantic 2.10 writes ``additionalProperties: true``. That boolean is passed
    back into the schema walker, which then raises APIInfoParseError and the
    page never renders.
    """
    import gradio_client.utils as client_utils

    original_get_type = client_utils.get_type
    original_to_python = client_utils._json_schema_to_python_type

    def get_type(schema):
        if not isinstance(schema, dict):
            return "Any"
        return original_get_type(schema)

    def to_python(schema, defs):
        if not isinstance(schema, dict):
            return "Any"
        return original_to_python(schema, defs)

    client_utils.get_type = get_type
    client_utils._json_schema_to_python_type = to_python


def parse_args():
    parser = argparse.ArgumentParser(description="RelocateEdit object relocation UI")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--share", action="store_true", help="Open a public gradio.live link")
    parser.add_argument("--preload", action="store_true", help="Load all four models at startup")
    parser.add_argument("--offload", action="store_true", help="Keep only the active model on the GPU")
    parser.add_argument("--server-name", default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=7860)
    return parser.parse_args()


def main():
    _patch_gradio_schema()
    args = parse_args()
    config = load_config(args.config)
    if args.offload:
        config.offload = True
    pipeline = RelocatePipeline(config)
    if args.preload:
        pipeline.preload()
    demo = build_demo(pipeline)
    demo.queue().launch(
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port,
        allowed_paths=[str(Path(config.outputs_dir))],
    )


if __name__ == "__main__":
    main()
