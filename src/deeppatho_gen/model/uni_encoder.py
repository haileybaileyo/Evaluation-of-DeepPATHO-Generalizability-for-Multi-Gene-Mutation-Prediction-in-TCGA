"""Frozen UNI patch encoder (MahmoodLab, ViT-L/16, 303M params, CC BY-NC-ND),
used as an alternative frozen backbone to Phikon-v2 (phikon_encoder.py).

Gated on HuggingFace -- unlike the phikon_encoder.py docstring's claim that
UNI/UNI2-h are unreachable without institutional approval, this project's
configured HF token *does* have approved access to MahmoodLab/UNI (verified
via a live hf_hub_download of its config.json). MahmoodLab/UNI2-h remains
gated/unapproved for this token as of this writing -- see docs/uni_access.md.

Same spatial-feature-map reasoning as PhikonPatchEncoder: UNI is also a
224x224, patch_size=16 ViT (config.json: vit_large_patch16_224, 1024
features, global_pool="token"), so it produces the same 14x14 token grid,
and we take the unpooled patch tokens (dropping the CLS token) for the same
reason MMA needs a spatial map, not a single pooled vector.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

UNI_MODEL_ID = "MahmoodLab/UNI"
UNI_INPUT_SIZE = 224
UNI_GRID = 14  # 224 / patch_size(16)
UNI_CHANNELS = 1024
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class UNIPatchEncoder(nn.Module):
    """Wraps a frozen UNI ViT-L/16. Input: (B, 3, H, W) in [0, 1] (any H, W
    -- resized internally). Output: (B, 1024, 14, 14) spatial feature map."""

    def __init__(self, freeze: bool = True):
        super().__init__()
        self.vit = timm.create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5,
            num_classes=0, dynamic_img_size=True,
        )
        weights_path = hf_hub_download(UNI_MODEL_ID, filename="pytorch_model.bin")
        state_dict = torch.load(weights_path, map_location="cpu")
        self.vit.load_state_dict(state_dict, strict=True)

        self.freeze = freeze
        if freeze:
            for p in self.vit.parameters():
                p.requires_grad = False
            self.vit.eval()

        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.out_channels = UNI_CHANNELS

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.vit.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=(UNI_INPUT_SIZE, UNI_INPUT_SIZE), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std

        context = torch.no_grad() if self.freeze else torch.enable_grad()
        with context:
            tokens = self.vit.forward_features(x)  # (B, 1 + 196, 1024): CLS + patch tokens
        tokens = tokens[:, 1:, :]  # drop CLS -> (B, 196, 1024)
        b, n, c = tokens.shape
        grid = int(n ** 0.5)
        assert grid == UNI_GRID, f"unexpected token grid {grid}x{grid}, expected {UNI_GRID}x{UNI_GRID}"
        return tokens.transpose(1, 2).reshape(b, c, grid, grid)
