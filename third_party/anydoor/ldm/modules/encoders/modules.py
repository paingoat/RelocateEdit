"""DINOv2 conditioner used by AnyDoor.

The upstream file also imports open_clip and a pinned transformers stack for
text encoders AnyDoor inference never calls. Those imports are omitted here.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn


def _ensure_dinov2_importable():
    # this file: third_party/anydoor/ldm/modules/encoders/modules.py
    dino_root = Path(__file__).resolve().parents[3] / "dinov2"
    entry = str(dino_root)
    if entry not in sys.path:
        sys.path.insert(0, entry)


class AbstractEncoder(nn.Module):
    def encode(self, *args, **kwargs):
        raise NotImplementedError


class FrozenDinoV2Encoder(AbstractEncoder):
    """DINOv2 ViT-g/14 image encoder. Projects 1536-d tokens to 1024-d."""

    def __init__(self, device="cpu", freeze=True, weight=None):
        super().__init__()
        _ensure_dinov2_importable()
        from dinov2.models.vision_transformer import vit_giant2

        dinov2 = vit_giant2(
            img_size=518,
            patch_size=14,
            init_values=1.0,
            ffn_layer="swiglufused",
            block_chunks=0,
        )
        if weight:
            try:
                state_dict = torch.load(weight, map_location="cpu", weights_only=False)
            except TypeError:
                state_dict = torch.load(weight, map_location="cpu")
            dinov2.load_state_dict(state_dict, strict=False)
        self.model = dinov2.to(device)
        self.device = device
        if freeze:
            self.freeze()
        self.image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.projector = nn.Linear(1536, 1024)

    def freeze(self):
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, image):
        if isinstance(image, list):
            image = torch.cat(image, 0)
        device = self.projector.weight.device
        image = (image.to(device) - self.image_mean.to(device)) / self.image_std.to(device)
        features = self.model.forward_features(image)
        tokens = features["x_norm_patchtokens"]
        image_features = features["x_norm_clstoken"].unsqueeze(1)
        hint = torch.cat([image_features, tokens], 1)
        return self.projector(hint)

    def encode(self, image):
        return self(image)
