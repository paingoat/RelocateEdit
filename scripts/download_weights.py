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

    _load_dotenv(ROOT / ".env")
    _configure_cache()
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


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _configure_cache() -> None:
    token = os.environ.get("HF_TOKEN", "")
    if token in ("", "hf_your_token_here"):
        os.environ.pop("HF_TOKEN", None)
    hub_token = os.environ.get("HUGGING_FACE_HUB_TOKEN", "")
    if hub_token in ("", "hf_your_token_here"):
        os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)

    if Path("/workspace").is_dir():
        data_root = Path(os.environ.get("RELOCATE_DATA_ROOT", "/workspace/data"))
    else:
        data_root = Path(os.environ.get("RELOCATE_DATA_ROOT", ROOT / ".cache"))
    data_root.mkdir(parents=True, exist_ok=True)
    data_weights = data_root / "weights"
    data_weights.mkdir(parents=True, exist_ok=True)
    if Path("/workspace").is_dir() and (WEIGHTS.is_symlink() or not WEIGHTS.exists()):
        if WEIGHTS.is_symlink():
            WEIGHTS.unlink()
        WEIGHTS.symlink_to(data_weights, target_is_directory=True)
    os.environ.setdefault("HF_HOME", str(data_root / "huggingface"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(os.environ["HF_HOME"]) / "hub"))
    os.environ.setdefault("HF_HUB_CACHE", os.environ["HUGGINGFACE_HUB_CACHE"])
    os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(os.environ["HF_HOME"]) / "transformers"))
    os.environ.setdefault("NLTK_DATA", str(data_root / "nltk_data"))
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "1") or "1"
    for key in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "NLTK_DATA"):
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    using_token = "HF_TOKEN" in os.environ or "HUGGING_FACE_HUB_TOKEN" in os.environ
    print(f"HF cache: {os.environ['HF_HOME']}")
    print(f"hf_transfer: {os.environ['HF_HUB_ENABLE_HF_TRANSFER']}")
    print(f"HF token: {'set' if using_token else 'not set'}")


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
