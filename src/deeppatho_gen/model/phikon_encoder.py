"""Frozen Phikon-v2 patch encoder, used as a drop-in replacement for our
from-scratch InceptionTower -- this is the "instance-level backbone that
already learned basic histopathology texture" the EGFR roadmap calls for,
substituting for the (infeasible -- see project notes) idea of transferring
from a trained ALK model. Phikon-v2 (Owkin, DINOv2-based ViT-L, 303M params,
CC-BY-NC-ND, openly downloadable with no gated approval) is pretrained via
self-supervision on large-scale histopathology, unlike gated alternatives
(UNI/UNI2-h) that need institutional approval we can't obtain programmatically.

Architectural note: Phikon-v2's ViT outputs one pooled embedding per image
by default, but our MMA module (model/attention.py) needs a *spatial*
feature map (C, H, W) to compute its per-position self-attention (Eq. 2-3
in Kosaraju et al.). We instead take the *unpooled* patch-token sequence
(196 tokens for a 224x224 image at patch_size=16, i.e. a 14x14 grid) and
reshape it back into (1024, 14, 14) -- this keeps MMA's spatial attention
meaningful instead of degenerating to a single vector.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

PHIKON_MODEL_ID = "owkin/phikon-v2"
PHIKON_INPUT_SIZE = 224
PHIKON_GRID = 14  # 224 / patch_size(16)
PHIKON_CHANNELS = 1024
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PhikonPatchEncoder(nn.Module):
    """Wraps a frozen Phikon-v2 ViT. Input: (B, 3, H, W) in [0, 1] (any H, W
    -- resized internally). Output: (B, 1024, 14, 14) spatial feature map."""

    def __init__(self, freeze: bool = True):
        super().__init__()
        self.vit = AutoModel.from_pretrained(PHIKON_MODEL_ID)
        self.freeze = freeze
        if freeze:
            for p in self.vit.parameters():
                p.requires_grad = False
            self.vit.eval()

        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.out_channels = PHIKON_CHANNELS

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.vit.eval()  # keep BN/dropout (if any) in eval mode even when the wrapper is .train()'d
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=(PHIKON_INPUT_SIZE, PHIKON_INPUT_SIZE), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std

        context = torch.no_grad() if self.freeze else torch.enable_grad()
        with context:
            out = self.vit(pixel_values=x)
        tokens = out.last_hidden_state[:, 1:, :]  # drop CLS token -> (B, 196, 1024)
        b, n, c = tokens.shape
        grid = int(n ** 0.5)
        assert grid == PHIKON_GRID, f"unexpected token grid {grid}x{grid}, expected {PHIKON_GRID}x{PHIKON_GRID}"
        return tokens.transpose(1, 2).reshape(b, c, grid, grid)
