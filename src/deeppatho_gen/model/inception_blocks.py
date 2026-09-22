"""Lightweight Inception-style blocks used as the per-branch feature extractor.

NOTE ON FIDELITY: Kosaraju et al. 2025 describe five stacked Inception
blocks ("Inception A to E") per branch, following the general Inception v3
block design, but the exact channel widths/kernel counts are only in their
Supplementary Table S6, which is not part of the openly available paper
text. Rather than guess numbers we can't verify, this module implements a
standard, parameter-reasonable Inception-style block (four parallel
branches: 1x1, 1x1->3x3, 1x1->3x3->3x3, pool->1x1) and stacks five of them,
matching the paper's depth and design *pattern* exactly, while leaving the
precise per-layer widths as a tunable choice documented in configs/.
"""
from __future__ import annotations

import torch.nn as nn
import torch


def conv_bn_relu(in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1, stride: int = 1) -> nn.Sequential:
    padding = dilation * (kernel_size - 1) // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class InceptionBlock(nn.Module):
    """One Inception-style block: four parallel branches concatenated."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        branch_ch = out_ch // 4

        self.branch1 = conv_bn_relu(in_ch, branch_ch, 1)

        self.branch2 = nn.Sequential(
            conv_bn_relu(in_ch, branch_ch, 1),
            conv_bn_relu(branch_ch, branch_ch, 3),
        )

        self.branch3 = nn.Sequential(
            conv_bn_relu(in_ch, branch_ch, 1),
            conv_bn_relu(branch_ch, branch_ch, 3),
            conv_bn_relu(branch_ch, branch_ch, 3),
        )

        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            conv_bn_relu(in_ch, out_ch - 3 * branch_ch, 1),
        )

        self.out_channels = out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)


class DilatedStem(nn.Module):
    """Multi-scale receptive field layers: a pair of dilated conv blocks
    (paper: dilation rate 2x2, tuned empirically) followed by max pooling,
    matching the "initial dilated convolutional blocks + pooling" stem
    described in the Architecture section."""

    def __init__(self, in_ch: int = 3, out_ch: int = 64, dilation: int = 2):
        super().__init__()
        self.block = nn.Sequential(
            conv_bn_relu(in_ch, out_ch, 3, dilation=dilation),
            conv_bn_relu(out_ch, out_ch, 3, dilation=dilation),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.out_channels = out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class InceptionTower(nn.Module):
    """Stem + 5 Inception blocks (Inception A-E in the paper's naming)."""

    def __init__(self, in_ch: int = 3, stem_ch: int = 64, widths=(96, 128, 160, 192, 192)):
        super().__init__()
        self.stem = DilatedStem(in_ch, stem_ch)
        blocks = []
        prev_ch = stem_ch
        for i, w in enumerate(widths):
            blocks.append(InceptionBlock(prev_ch, w))
            prev_ch = w
            if i in (0, 2):  # two extra downsamples, keeping spatial dims manageable
                blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.blocks = nn.Sequential(*blocks)
        self.out_channels = prev_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.stem(x))
