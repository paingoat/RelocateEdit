"""Download the four checkpoints. Does not run the models.

AnyDoor's published file is about 17 GB because it still contains optimizer
state. This script keeps only the weight state dict. That step needs roughly
32 GB of RAM. The raw file is deleted after the smaller checkpoint is written.
"""

from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"


def main():
    parser = argparse.ArgumentParser(description="Download RelocateEdit weights")
    parser.add_argument("--with-metric", action="store_true", help="Also download indoor and outdoor metric depth")
    parser.add_argument("--with-dino", action="store_true", help="Also download dinov2_vitg14_pretrain.pth (optional)")
    parser.add_argument("--keep-raw", action="store_true", help="Keep the 17 GB AnyDoor file after stripping")
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", str(WEIGHTS / "hf-cache"))
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    from huggingface_hub import hf_hub_download

    seem = WEIGHTS / "seem"
    _hf(hf_hub_download, "xdecoder/SEEM", "seem_focall_v0.pt", seem, repo_type="model")

    depth = WEIGHTS / "depth"
    _hf(
        hf_hub_download,
        "depth-anything/Depth-Anything-V2-Large",
        "depth_anything_v2_vitl.pth",
        depth,
        repo_type="model",
    )
    if args.with_metric:
        _hf(
            hf_hub_download,
            "depth-anything/Depth-Anything-V2-Metric-Hypersim-Large",
            "depth_anything_v2_metric_hypersim_vitl.pth",
            depth,
            repo_type="model",
        )
        _hf(
            hf_hub_download,
            "depth-anything/Depth-Anything-V2-Metric-VKITTI-Large",
            "depth_anything_v2_metric_vkitti_vitl.pth",
            depth,
            repo_type="model",
        )

    lama_root = WEIGHTS / "lama"
    lama_zip = _hf(hf_hub_download, "smartywu/big-lama", "big-lama.zip", lama_root, repo_type="model")
    if _find_lama(lama_root) is None:
        print(f"Extracting {lama_zip}")
        with zipfile.ZipFile(lama_zip) as archive:
            archive.extractall(lama_root)
    lama_dir = _find_lama(lama_root)
    if lama_dir is None:
        raise FileNotFoundError(f"big-lama.zip did not contain models/best.ckpt under {lama_root}")
    print(f"LaMa checkpoint: {lama_dir}")

    anydoor_dir = WEIGHTS / "anydoor"
    raw = _hf(
        hf_hub_download,
        "xichenhku/AnyDoor",
        "epoch=1-step=8687.ckpt",
        anydoor_dir,
        repo_type="space",
    )
    stripped = anydoor_dir / "anydoor.ckpt"
    if not stripped.is_file():
        _strip_checkpoint(raw, stripped)
    if not args.keep_raw and Path(raw).resolve() != stripped.resolve() and Path(raw).is_file():
        Path(raw).unlink()
        print(f"Removed raw checkpoint {raw}")

    if args.with_dino:
        _download_dino(anydoor_dir / "dinov2_vitg14_pretrain.pth")

    _prefetch_tokenizer()
    print("Weights are in", WEIGHTS)


def _find_lama(root: Path):
    expected = root / "big-lama" / "models" / "best.ckpt"
    if expected.is_file():
        return expected.parent.parent
    for path in root.rglob("best.ckpt"):
        if path.parent.name == "models":
            return path.parent.parent
    return None


def _hf(hf_hub_download, repo_id, filename, dest: Path, repo_type: str) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Fetching {repo_id} / {filename}")
    path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type, local_dir=str(dest))
    return Path(path)


def _strip_checkpoint(src: Path, dest: Path) -> None:
    import torch

    print(f"Stripping optimizer state from {src} (needs a lot of RAM)")
    try:
        checkpoint = torch.load(src, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(src, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    torch.save({"state_dict": state_dict}, dest)
    print(f"Wrote {dest}")


def _download_dino(dest: Path) -> None:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        return
    url = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth"
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, dest)


def _prefetch_tokenizer() -> None:
    from transformers import CLIPTokenizer

    print("Caching the CLIP tokenizer used by SEEM")
    CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    import nltk

    nltk.download("punkt", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)


if __name__ == "__main__":
    main()
