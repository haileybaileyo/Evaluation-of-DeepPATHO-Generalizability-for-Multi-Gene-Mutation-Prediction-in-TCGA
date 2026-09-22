"""PATHO-CAM: gradient-weighted class activation map on the local (20x)
attention-block output, re-implementing Kosaraju et al. 2025 Eq. 6-7.

  alpha_k = sum_{i,j} ReLU( d P_ALK+ / d A^k_ij )         (Eq. 6)
  I = sum_k alpha_k * A^k                                  (Eq. 7)

The paper computes this for their single ALK logit; here it's generalized
to any of the multi-gene output logits so it works for the phase-1 genes.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..model.backbone import DeepPATHOClassifier


@torch.enable_grad()
def patho_cam(
    model: DeepPATHOClassifier,
    patch_local_20x: torch.Tensor,
    patch_global_5x: torch.Tensor,
    class_index: int,
) -> torch.Tensor:
    """Returns the discriminative map I, shape (B, H, W), normalized to [0, 1]
    per-sample (paper: "normalized based on each patch, each slide, and the
    entire data" -- per-patch normalization is what's implemented here; the
    slide/dataset-level normalization is a simple rescale you apply after
    collecting many patches' maps, left to the caller).
    """
    model.eval()
    patch_local_20x = patch_local_20x.clone().requires_grad_(False)

    feat_local, r_local = model.local_branch_activations(patch_local_20x, patch_global_5x)
    r_local.retain_grad()

    pooled = model.gap(r_local).flatten(1)
    # Reconstruct the same concatenation order as forward() but zero out the
    # branches PATHO-CAM doesn't attribute to (global-branch + pre-MMA local),
    # so the gradient signal is isolated to the local attention-block output,
    # matching the paper's "intermediate outputs of the local attention block".
    zeros = torch.zeros_like(pooled)
    concat = torch.cat([pooled, zeros, zeros, zeros], dim=1)
    logits = model.head(concat)
    score = logits[:, class_index].sum()

    grads = torch.autograd.grad(score, r_local, retain_graph=False)[0]  # (B, C, h, w)
    alpha = F.relu(grads).sum(dim=(2, 3))  # Eq. 6: alpha_k, shape (B, C)

    cam = torch.einsum("bc,bchw->bhw", alpha, r_local)  # Eq. 7
    cam = F.relu(cam)

    cam_min = cam.amin(dim=(1, 2), keepdim=True)
    cam_max = cam.amax(dim=(1, 2), keepdim=True)
    cam_norm = (cam - cam_min) / (cam_max - cam_min + 1e-8)
    return cam_norm.detach()


def upsample_cam_to_patch(cam: torch.Tensor, patch_size: int = 256) -> torch.Tensor:
    """Upsamples the (typically low-res) CAM to the patch's pixel resolution
    for overlay visualization."""
    return F.interpolate(cam.unsqueeze(1), size=(patch_size, patch_size), mode="bilinear", align_corners=False).squeeze(1)
