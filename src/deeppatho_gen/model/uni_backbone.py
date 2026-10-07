"""DeepPATHO-style classifier with a frozen UNI backbone in place of the
from-scratch Inception towers -- the UNI counterpart to
phikon_backbone.py's PhikonDeepPATHOClassifier. Identical structure; only
the encoder class differs, since UNI and Phikon-v2 both expose a 1024-
channel, 14x14 spatial feature map (see uni_encoder.py / phikon_encoder.py).

Both magnification branches (5x, 20x) share ONE frozen UNI instance, for the
same reason as Phikon: it's a stateless frozen feature extractor.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .attention import MultiMagnificationAttention
from .uni_encoder import UNIPatchEncoder


class UNIDeepPATHOClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        freeze_backbone: bool = True,
        mag_ratio: float = 5.0 / 20.0,
        fc_dims: tuple[int, ...] = (512, 128),
    ):
        super().__init__()
        self.encoder = UNIPatchEncoder(freeze=freeze_backbone)
        c = self.encoder.out_channels

        self.mma = MultiMagnificationAttention(c, mag_ratio=mag_ratio)
        self.gap = nn.AdaptiveAvgPool2d(1)

        feat_dim = c * 4  # R_L, R_G, A_L(orig), A_G(orig)
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
