"""20x-only single-magnification baseline: the same Inception tower as
DeepPATHOClassifier's local branch, with no global (5x) branch and no
cross-magnification attention. Used only to isolate whether MMA itself
contributes anything beyond a plain single-tower classifier -- everything
else (stem, Inception block design, dropout, FC head shape) is kept
identical to DeepPATHOClassifier so the comparison isolates the one
variable (MMA) rather than confounding it with unrelated architecture
differences.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .inception_blocks import InceptionTower


class SingleMagnificationClassifier(nn.Module):
    def __init__(self, num_classes: int, dropout_local: float = 0.2, fc_dims: tuple[int, ...] = (1024, 256)):
        super().__init__()
        self.tower_local = InceptionTower()
        self.drop_local = nn.Dropout2d(dropout_local)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Same total feature width as DeepPATHOClassifier's pooled vector
        # (tower_out * 4, since that model concats 4 pooled tensors) is NOT
        # matched here on purpose -- matching param count exactly would mean
        # inflating this single tower artificially. We report parameter
        # counts for both models explicitly instead, and let the FC head
        # width follow naturally from one tower's output (tower_out * 1).
        feat_dim = self.tower_local.out_channels
        layers = []
        prev = feat_dim
        for d in fc_dims:
            layers += [nn.Linear(prev, d), nn.ReLU(inplace=True), nn.Dropout(0.4)]
            prev = d
        layers.append(nn.Linear(prev, num_classes))
        self.head = nn.Sequential(*layers)

    def forward(self, patch_local_20x: torch.Tensor, patch_global_5x: torch.Tensor | None = None) -> torch.Tensor:
        # patch_global_5x accepted-and-ignored so this class is drop-in
        # compatible with the same training loop / DataLoader batches used
        # for DeepPATHOClassifier (which takes both magnifications).
        feat = self.tower_local(patch_local_20x)
        feat = self.drop_local(feat)
        pooled = self.gap(feat).flatten(1)
        return self.head(pooled)
