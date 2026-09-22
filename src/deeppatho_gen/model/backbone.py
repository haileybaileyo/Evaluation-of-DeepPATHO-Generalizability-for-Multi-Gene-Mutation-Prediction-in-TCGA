"""Full instance-level DeepPATHO model: twin Inception towers (5x global,
20x local) synchronized by Multi-Magnification Attention, then a
multi-label sigmoid head over target genes (extending the paper's
single-task ALK head to the multi-gene setup from Coudray et al.'s
mutation-prediction experiment).

forward(patch_local_20x, patch_global_5x) -> logits (B, num_classes)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .attention import MultiMagnificationAttention
from .inception_blocks import InceptionTower


class DeepPATHOClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        dropout_local: float = 0.2,
        dropout_global: float = 0.7,
        mag_ratio: float = 5.0 / 20.0,
        fc_dims: tuple[int, ...] = (1024, 256),
    ):
        super().__init__()
        self.tower_local = InceptionTower()
        self.tower_global = InceptionTower()
        assert self.tower_local.out_channels == self.tower_global.out_channels

        self.mma = MultiMagnificationAttention(self.tower_local.out_channels, mag_ratio=mag_ratio)

        self.drop_local = nn.Dropout2d(dropout_local)
        self.drop_global = nn.Dropout2d(dropout_global)

        # Concatenate synchronized + original pooled features from both
        # branches, per "synchronized feature maps ... and original feature
        # maps ... concatenated, flattened, connected to FC layers."
        feat_dim = self.tower_local.out_channels * 4  # R_L, R_G, A_L(orig), A_G(orig)
        gap = nn.AdaptiveAvgPool2d(1)
        self.gap = gap

        layers = []
        prev = feat_dim
        for d in fc_dims:
            layers += [nn.Linear(prev, d), nn.ReLU(inplace=True), nn.Dropout(0.4)]
            prev = d
        layers.append(nn.Linear(prev, num_classes))
        self.head = nn.Sequential(*layers)

    def forward(self, patch_local_20x: torch.Tensor, patch_global_5x: torch.Tensor) -> torch.Tensor:
        feat_local = self.tower_local(patch_local_20x)
        feat_global = self.tower_global(patch_global_5x)

        r_global, r_local = self.mma(feat_global, feat_local)
        r_local = self.drop_local(r_local)
        r_global = self.drop_global(r_global)

        pooled = torch.cat(
            [
                self.gap(r_local).flatten(1),
                self.gap(r_global).flatten(1),
                self.gap(feat_local).flatten(1),
                self.gap(feat_global).flatten(1),
            ],
            dim=1,
        )
        return self.head(pooled)  # raw logits; apply sigmoid at loss/inference time

    # --- exposed for PATHO-CAM (interpret/patho_cam.py) ---
    def local_branch_activations(self, patch_local_20x: torch.Tensor, patch_global_5x: torch.Tensor):
        """Returns (feat_local_pre_mma, r_local_post_mma) with gradients enabled,
        matching the "intermediate outputs of the local attention block" that
        PATHO-CAM (paper Eq. 6-7) differentiates through."""
        feat_local = self.tower_local(patch_local_20x)
        feat_global = self.tower_global(patch_global_5x)
        _, r_local = self.mma(feat_global, feat_local)
        return feat_local, r_local
