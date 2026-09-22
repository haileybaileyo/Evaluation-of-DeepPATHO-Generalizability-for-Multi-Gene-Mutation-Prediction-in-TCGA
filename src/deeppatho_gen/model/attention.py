"""Multi-Magnification Attention (MMA) module.

Re-implementation of Kosaraju et al. 2025 (DeepPATHO), Methods Eq. 1-5.
The official repo (github.com/datax-lab/DeepPATHO) ships the Inception
backbone but not this attention block or the up/downsampling logic, so this
is our own reconstruction from the paper text -- treat the exact resampling
behavior as a documented approximation, not a byte-for-byte match to the
authors' implementation.

Paper's definitions, for A_G, A_L in R^{c x h x w} (global/local Inception
feature maps of identical shape):
  A_g^{q|k|v} = ReLU(A_g conv1x1)                    (Eq. 1, g in {G, L})
  G = A_G^{k,T} . A_G^{q}   in R^{(hw) x (hw)}         (Eq. 2, self-attention within global)
  L = A_L^{k,T} . A_L^{q}   in R^{(hw) x (hw)}         (Eq. 3, self-attention within local)
  synchronize: crop+upsample G, avgpool+pad L          (paper prose, no closed form given)
  R_L = ReLU(A_L^v (x) G_sync)                          (Eq. 4)
  R_G = ReLU(A_G^v (x) L_sync)                          (Eq. 5)

Physical meaning of the sync step: the global (5x) patch and local (20x)
patch share a center point, but the global patch's field of view is 4x
wider per side (b = 5/20 = 0.25 in the paper's notation). So:
  - the *center* quarter of the global attention field corresponds to the
    full local field of view -> crop G to its center (h*b, w*b) region,
    then bilinearly upsample back to (h, w) so it can gate A_L^v.
  - conversely, the full local attention field only covers the center
    quarter of the global field of view -> average-pool L down to
    (h*b, w*b) and zero-pad it back out to (h, w) so it can gate A_G^v.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _QKV(nn.Module):
    """1x1-conv projections producing query/key/value maps (Eq. 1)."""

    def __init__(self, channels: int):
        super().__init__()
        self.q = nn.Conv2d(channels, channels, kernel_size=1)
        self.k = nn.Conv2d(channels, channels, kernel_size=1)
        self.v = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        return F.relu(self.q(x)), F.relu(self.k(x)), F.relu(self.v(x))


def _self_attention_map(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Eq. 2 / Eq. 3: per-sample self-attention map in R^{(hw) x (hw)}.

    q, k: (B, C, H, W) -> attention: (B, H*W, H*W), softmax-normalized over
    the key dimension (the paper gives the raw bilinear form k^T . q; we add
    a softmax + 1/sqrt(d) scale, standard practice for this to train stably).
    """
    b, c, h, w = q.shape
    q_flat = q.flatten(2)  # (B, C, HW)
    k_flat = k.flatten(2)  # (B, C, HW)
    attn = torch.einsum("bcn,bcm->bnm", k_flat, q_flat) / (c ** 0.5)  # (B, HW_k, HW_q)
    return F.softmax(attn, dim=1)


def _sync_global_to_local(attn_global: torch.Tensor, h: int, w: int, b_ratio: float) -> torch.Tensor:
    """Crop the center (h*b, w*b) region of the global attention field and
    upsample it back to (h, w), so it can gate the local value map."""
    bsz = attn_global.shape[0]
    grid = attn_global.transpose(1, 2).reshape(bsz, h * w, h, w)  # per-query key-map, spatial
    ch, cw = max(1, round(h * b_ratio)), max(1, round(w * b_ratio))
    top, left = (h - ch) // 2, (w - cw) // 2
    cropped = grid[:, :, top:top + ch, left:left + cw]
    upsampled = F.interpolate(cropped, size=(h, w), mode="bilinear", align_corners=False)
    return upsampled.reshape(bsz, h * w, h * w).transpose(1, 2)


def _sync_local_to_global(attn_local: torch.Tensor, h: int, w: int, b_ratio: float) -> torch.Tensor:
    """Average-pool the local attention field down to (h*b, w*b) and
    zero-pad it back into the center of a (h, w) canvas, so it can gate the
    global value map."""
    bsz = attn_local.shape[0]
    grid = attn_local.transpose(1, 2).reshape(bsz, h * w, h, w)
    ph, pw = max(1, round(h * b_ratio)), max(1, round(w * b_ratio))
    pooled = F.adaptive_avg_pool2d(grid, output_size=(ph, pw))
    top, left = (h - ph) // 2, (w - pw) // 2
    canvas = torch.zeros(bsz, h * w, h, w, device=attn_local.device, dtype=attn_local.dtype)
    canvas[:, :, top:top + ph, left:left + pw] = pooled
    return canvas.reshape(bsz, h * w, h * w).transpose(1, 2)


class MultiMagnificationAttention(nn.Module):
    """Synchronizes global (low-mag) and local (high-mag) Inception feature
    maps of identical shape (B, C, H, W). Returns (R_G, R_L), same shape."""

    def __init__(self, channels: int, mag_ratio: float = 5.0 / 20.0):
        super().__init__()
        self.mag_ratio = mag_ratio
        self.qkv_global = _QKV(channels)
        self.qkv_local = _QKV(channels)

    def forward(self, feat_global: torch.Tensor, feat_local: torch.Tensor):
        assert feat_global.shape == feat_local.shape, (
            "global/local feature maps must match in shape; both branches "
            "must use the same backbone stride so a shared center point "
            "makes sense (see module docstring)."
        )
        _, c, h, w = feat_global.shape

        q_g, k_g, v_g = self.qkv_global(feat_global)
        q_l, k_l, v_l = self.qkv_local(feat_local)

        attn_global = _self_attention_map(q_g, k_g)  # Eq. 2
        attn_local = _self_attention_map(q_l, k_l)   # Eq. 3

        gate_for_local = _sync_global_to_local(attn_global, h, w, self.mag_ratio)
        gate_for_global = _sync_local_to_global(attn_local, h, w, self.mag_ratio)

        gate_for_local = gate_for_local.sum(dim=1).reshape(-1, 1, h, w)
        gate_for_global = gate_for_global.sum(dim=1).reshape(-1, 1, h, w)

        r_local = F.relu(v_l * gate_for_local)    # Eq. 4
        r_global = F.relu(v_g * gate_for_global)  # Eq. 5
        return r_global, r_local
