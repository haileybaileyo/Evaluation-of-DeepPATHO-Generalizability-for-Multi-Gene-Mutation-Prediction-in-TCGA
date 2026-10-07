"""DeepPATHO-style classifier with a frozen Phikon-v2 backbone in place of
the from-scratch Inception towers. MMA (model/attention.py) and the FC head
are randomly initialized and fully trainable -- they're the only parts that
have never seen any data, so freezing them alongside Phikon would leave
them at random init forever. This is the practical reading of the roadmap's
"fine-tune only the last FC layer": Phikon is the actual pretrained
component (frozen, as intended), while MMA is new architecture bolted on
top of it that necessarily has to train from scratch either way.

Both magnification branches (5x, 20x) share ONE frozen Phikon instance
(same weights, called twice) rather than two separate copies -- it's a
stateless frozen feature extractor, so there's no benefit to duplicating
303M frozen parameters in memory.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .attention import MultiMagnificationAttention
from .phikon_encoder import PhikonPatchEncoder


class PhikonDeepPATHOClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        freeze_backbone: bool = True,
        mag_ratio: float = 5.0 / 20.0,
        fc_dims: tuple[int, ...] = (512, 128),
    ):
        super().__init__()
        self.encoder = PhikonPatchEncoder(freeze=freeze_backbone)
        c = self.encoder.out_channels

        self.mma = MultiMagnificationAttention(c, mag_ratio=mag_ratio)
        self.gap = nn.AdaptiveAvgPool2d(1)

        feat_dim = c * 4  # R_L, R_G, A_L(orig), A_G(orig) -- same concat pattern as DeepPATHOClassifier
        layers = []
        prev = feat_dim
        for d in fc_dims:
            layers += [nn.Linear(prev, d), nn.ReLU(inplace=True), nn.Dropout(0.4)]
            prev = d
        layers.append(nn.Linear(prev, num_classes))
        self.head = nn.Sequential(*layers)

    def forward(self, patch_local_20x: torch.Tensor, patch_global_5x: torch.Tensor) -> torch.Tensor:
        feat_local = self.encoder(patch_local_20x)
        feat_global = self.encoder(patch_global_5x)

        r_global, r_local = self.mma(feat_global, feat_local)

        pooled = torch.cat(
            [self.gap(r_local).flatten(1), self.gap(r_global).flatten(1),
             self.gap(feat_local).flatten(1), self.gap(feat_global).flatten(1)],
            dim=1,
        )
        return self.head(pooled)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
