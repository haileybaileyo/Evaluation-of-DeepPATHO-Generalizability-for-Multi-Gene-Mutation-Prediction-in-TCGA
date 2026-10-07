"""Plain Inception v3 baseline (torchvision), 20x patch only: no 5x branch and
no multi-magnification attention (MMA).

Purpose: the "basic CNN/Inception" baseline requested in the lab meeting, to
show that the data split / training pipeline behave sensibly with a standard,
well-known model before comparing against UNI and DeepPATHO (Inception + MMA).
Coudray et al. 2018 used Inception v3 for the same TCGA-LUAD mutation task.

Drop-in compatible with PatchPairDataset's batches: forward() accepts both
magnifications and ignores the 5x one (same convention as
single_mag_baseline.SingleMagnificationClassifier).

Input convention: (B, 3, H, W) in [0, 1]. ImageNet mean/std normalization is
applied here. torchvision's pretrained Inception v3 has transform_input=True,
which expects ImageNet-normalized input and re-maps it internally to the
(x - 0.5) / 0.5 scaling the original weights were trained with.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import Inception_V3_Weights, inception_v3

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class InceptionV3Classifier(nn.Module):
    def __init__(self, num_classes: int = 1, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()
        weights = Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None
        # aux_logits must stay True while loading the pretrained state dict;
        # the auxiliary head is dropped right after so train/eval both return
        # a single logits tensor.
        self.net = inception_v3(weights=weights, aux_logits=True, transform_input=pretrained, init_weights=not pretrained)
        self.net.aux_logits = False
        self.net.AuxLogits = None
        self.net.dropout = nn.Dropout(dropout)
        self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, patch_local_20x: torch.Tensor, patch_global_5x: torch.Tensor | None = None) -> torch.Tensor:
        x = (patch_local_20x - self.mean) / self.std
        return self.net(x)
