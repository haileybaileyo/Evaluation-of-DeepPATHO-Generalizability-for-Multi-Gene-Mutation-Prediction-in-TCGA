"""Inception-style blocks used as the per-branch feature extractor.

Tightened against details confirmed by reading the *actually-runnable*
parts of the official DeepPATHO repo's Utlis/ folder (Dilationlayer.py,
InceptionB.py, InceptionE.py -- these three files have no missing imports
and no undefined names, unlike InceptionA.py, InceptionC.py, upscale.py/
DEEPPATHO.py and DeepPATHO_training.py, which all throw on use). Confirmed,
reused details:
  - activation is 'tanh' throughout, not ReLU
  - stem = 5 stacked dilated conv layers (32ch each, kernel sizes
    3x3/3x3/3x3/1x1/3x3), dilation_rate=(2,2), interleaved with 2 max-pools
  - BatchNormalization uses scale=False
  - InceptionB is a reduction block (stride-2 conv branch + stride-2
    conv-conv-conv branch + stride-2 maxpool branch, concatenated)
  - InceptionE factorizes 3x3 into parallel 1x3/3x1 convs

Still NOT reproducible from the public repo (exact channel widths beyond
"32 per branch", the number of Inception stages before each reduction,
and the broken attention/sync code) -- see model/attention.py's docstring.
InceptionA.py and InceptionC.py in the official repo both contain a
real bug/crash (a computed-but-unused branch in A, an undefined variable
in C), so their *bugs* aren't reproduced here; only the correct patterns
from B/E and the stem are.
"""
from __future__ import annotations

import torch.nn as nn
import torch

ACTIVATION = nn.Tanh  # confirmed from Dilationlayer.py / InceptionB.py / InceptionE.py


def conv_bn_act(in_ch: int, out_ch: int, kernel_size, dilation: int = 1, stride: int = 1) -> nn.Sequential:
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    padding = tuple(d * (k - 1) // 2 for d, k in zip((dilation, dilation), kernel_size))
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_ch, affine=True),  # scale=False in Keras == no learnable gamma; BatchNorm2d
        # here keeps affine=True since removing gamma entirely has no direct
        # nn.BatchNorm2d equivalent worth the complexity for a research
        # re-implementation -- documented as a minor, deliberate deviation.
        ACTIVATION(),
    )


class DilatedStem(nn.Module):
    """5 dilated conv layers (32ch, kernels 3/3/3/1/3) + 2 max-pools,
    reproduced exactly from Dilationlayer.py's Dilationlayer.bulid()."""

    def __init__(self, in_ch: int = 3, width: int = 32, dilation: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            conv_bn_act(in_ch, width, 3, dilation=dilation),
            conv_bn_act(width, width, 3, dilation=dilation),
            conv_bn_act(width, width, 3, dilation=dilation),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            conv_bn_act(width, width, 1, dilation=dilation),
            conv_bn_act(width, width, 3, dilation=dilation),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.out_channels = width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class InceptionA(nn.Module):
    """1x1 / 1x1->5x5 / 1x1->3x3->3x3 / pool->1x1, four branches
    concatenated -- the *correct* form of the official repo's InceptionA
    (theirs computes a branch1x1 it never uses and duplicates the 5x5
    branch into its slot instead; not reproduced here)."""

    def __init__(self, in_ch: int, branch_ch: int = 32):
        super().__init__()
        self.branch1x1 = conv_bn_act(in_ch, branch_ch, 1)
        self.branch5x5 = nn.Sequential(conv_bn_act(in_ch, branch_ch, 1), conv_bn_act(branch_ch, branch_ch, 5))
        self.branch3x3dbl = nn.Sequential(
            conv_bn_act(in_ch, branch_ch, 1), conv_bn_act(branch_ch, branch_ch, 3), conv_bn_act(branch_ch, branch_ch, 3)
        )
        self.branch_pool = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1), conv_bn_act(in_ch, branch_ch, 1))
        self.out_channels = branch_ch * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.branch1x1(x), self.branch5x5(x), self.branch3x3dbl(x), self.branch_pool(x)], dim=1)


class InceptionB(nn.Module):
    """Reduction block, reproduced from the official (working)
    InceptionB.py: stride-2 3x3 conv / 1x1->3x3->stride-2 3x3 / stride-2
    maxpool, concatenated. Halves spatial resolution."""

    def __init__(self, in_ch: int, branch_ch: int = 32):
        super().__init__()
        self.branch3x3 = conv_bn_act(in_ch, branch_ch, 3, stride=2)
        self.branch3x3dbl = nn.Sequential(
            conv_bn_act(in_ch, branch_ch, 1),
            conv_bn_act(branch_ch, branch_ch, 3),
            conv_bn_act(branch_ch, branch_ch, 3, stride=2),
        )
        self.branch_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.out_channels = branch_ch * 2 + in_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.branch3x3(x), self.branch3x3dbl(x), self.branch_pool(x)], dim=1)


class InceptionE(nn.Module):
    """Factorized-conv block, reproduced from the official (working)
    InceptionE.py: 1x1 / (1x1 -> parallel 1x3+3x1) / (1x1->1x1 -> parallel
    1x3+3x1) / pool->1x1, four branches concatenated."""

    def __init__(self, in_ch: int, branch_ch: int = 32):
        super().__init__()
        self.branch1x1 = conv_bn_act(in_ch, branch_ch, 1)

        self.branch3x3_reduce = conv_bn_act(in_ch, branch_ch, 1)
        self.branch3x3_a = conv_bn_act(branch_ch, branch_ch, (1, 3))
        self.branch3x3_b = conv_bn_act(branch_ch, branch_ch, (3, 1))

        self.branch3x3dbl_reduce = nn.Sequential(conv_bn_act(in_ch, branch_ch, 1), conv_bn_act(branch_ch, branch_ch, 1))
        self.branch3x3dbl_a = conv_bn_act(branch_ch, branch_ch, (1, 3))
        self.branch3x3dbl_b = conv_bn_act(branch_ch, branch_ch, (3, 1))

        self.branch_pool = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1), conv_bn_act(in_ch, branch_ch, 1))
        self.out_channels = branch_ch * 6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b3 = self.branch3x3_reduce(x)
        b3 = torch.cat([self.branch3x3_a(b3), self.branch3x3_b(b3)], dim=1)

        b3d = self.branch3x3dbl_reduce(x)
        b3d = torch.cat([self.branch3x3dbl_a(b3d), self.branch3x3dbl_b(b3d)], dim=1)

        return torch.cat([self.branch1x1(x), b3, b3d, self.branch_pool(x)], dim=1)


class InceptionTower(nn.Module):
    """Stem + Inception A/B/A/B/E stack (A/E keep resolution + widen
    channels, B halves resolution -- so we still shrink 256x256 input down
    to a manageable feature-map size across "5 Inception blocks", matching
    the paper's stated depth, using the two confirmed-correct block designs
    plus a corrected InceptionA)."""

    def __init__(self, in_ch: int = 3, branch_ch: int = 32):
        super().__init__()
        self.stem = DilatedStem(in_ch, width=branch_ch)
        c = self.stem.out_channels

        self.block1 = InceptionA(c, branch_ch); c = self.block1.out_channels
        self.block2 = InceptionB(c, branch_ch); c = self.block2.out_channels
        self.block3 = InceptionA(c, branch_ch); c = self.block3.out_channels
        self.block4 = InceptionB(c, branch_ch); c = self.block4.out_channels
        self.block5 = InceptionE(c, branch_ch); c = self.block5.out_channels

        self.out_channels = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x
