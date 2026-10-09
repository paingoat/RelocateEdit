# Vendored inference code

RelocateEdit keeps a trimmed copy of each model so the app can run in one conda
environment. Training code, demos, and custom CUDA ops are not included.
Weights are downloaded separately into `weights/` and are not part of this tree.

| Folder | Upstream | What is kept |
| --- | --- | --- |
| `seem/` | [SEEM](https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once) (MIT) | `modeling/` without the deformable-attention CUDA op, plus `utils/` without the visualizer. Demo config: Focal-L v0 (`seem_focall_v0.pt`). |
| `depth_anything_v2/` | [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) (Apache-2.0 code; Large weights are CC-BY-NC-4.0) | The relative-depth package. |
| `lama/` | [LaMa](https://github.com/advimman/lama) (Apache-2.0) plus the feature refiner from [arXiv:2206.13644](https://arxiv.org/abs/2206.13644) | FFC generator and `evaluation/refinement.py` only. |
| `anydoor/` | [AnyDoor](https://github.com/ali-vilab/AnyDoor) (Apache-2.0) and DINOv2 (Apache-2.0) | ControlLDM inference modules and the ViT-g backbone. |

Each folder still has its upstream `LICENSE`.

## Patches

### SEEM

- `seem/utils/distributed.py`: `mpi4py` is optional. Single-process inference no longer calls `hostname -I` or MPI.
- `seem/modeling/BaseModel.py`: `torch.load(..., weights_only=False)` so a PyTorch 2.6+ default does not reject the checkpoint.
- The deformable encoder is still imported inside a `try/except` and is not used. The demo config uses the FPN encoder, so `modeling/vision/encoder/ops` is not vendored.
- The app never imports the SEEM Gradio demo or Whisper.

### Depth Anything V2

- Empty `__init__.py` files so the folder can live inside this repo.
- `dpt.py`: optional `max_depth`. When set, the head ends in a sigmoid and the output is in meters (metric checkpoints). Relative mode is unchanged: nonnegative disparity, larger means closer.
- `dpt.py`: `image2tensor` uses `self.device` when the wrapper sets it.

### LaMa

- `training/modules/ffc.py`: dropped the unused `saicinpainting.utils` import so the generator does not pull in PyTorch Lightning.
- `evaluation/refinement.py`: `pad_tensor_to_modulo` and `move_to_device` are inlined. The pix2pixhd import is replaced by an empty `ResnetBlock` used only for an `isinstance` check. big-lama's blocks are `FFCResnetBlock`.

### AnyDoor

- `ldm/modules/encoders/modules.py`: only `FrozenDinoV2Encoder` remains, so `open_clip` and the text-encoder stack are not imported. The ViT-g is built with `vit_giant2(...)`. Its weight path is a constructor argument and may be empty, because the AnyDoor checkpoint already contains those weights.
- `ldm/models/diffusion/ddpm.py`: `rank_zero_only` falls back to the Lightning 2.x path.
- `cldm/model.py`: `torch.load(..., weights_only=False)`.
- `dinov2/layers/swiglu_ffn.py`: the FFN is the local packed `w12`/`w3` module, with the same hidden-size rounding as upstream `SwiGLUFFNFused`. That matches `anydoor.ckpt`. xformers `SwiGLU` is not used, because its parameter names differ across releases. xformers can still accelerate attention.
- `configs/anydoor.yaml` in this repo (not upstream): `cond_stage_config.params.weight` is null and the encoder is constructed on CPU. The wrapper moves the whole model afterwards.

## Import roots

The wrappers add one of these directories to `sys.path` before importing:

- `third_party/seem` for `modeling` and `utils`
- `third_party/depth_anything_v2` for `depth_anything_v2`
- `third_party/lama` for `saicinpainting`
- `third_party/anydoor` for `cldm` and `ldm`
- `third_party/anydoor/dinov2` for `dinov2` (added by the encoder module)
