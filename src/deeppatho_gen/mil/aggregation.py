"""Slide-level aggregation of patch-level probabilities.

Implements the three aggregation strategies both source papers actually
specify in enough detail to reproduce exactly:
  - mean_probability : Coudray et al.'s "average of tile probabilities"
  - percent_positive  : Coudray et al.'s "percentage of tiles positively classified"
  - max               : Kosaraju et al.'s MAX pooling MIL, which they found
                         gives the best recall for the ALK screening task
                         (a slide is positive if *any* patch is confidently positive)

HipoMap and CAMIL (the two embedding-based MILs in Kosaraju et al.) are
published as separate methods with their own repos/architectures and are
out of scope to reimplement here; MAX and mean-probability already let us
reproduce every number reported in Table 1 of Coudray et al. for a direct
comparison, which is the immediate goal of this extension.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def aggregate_slide_scores(
    patch_probs: pd.DataFrame,
    slide_col: str = "file_id",
    prob_cols: list[str] | None = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """patch_probs: one row per patch, columns = [slide_col, gene_1, gene_2, ...]
    each holding that patch's sigmoid probability for the gene.

    Returns one row per slide per aggregation method with columns:
    [slide_col, method, gene_1, gene_2, ...]
    """
    if prob_cols is None:
        prob_cols = [c for c in patch_probs.columns if c != slide_col]

    grouped = patch_probs.groupby(slide_col)

    mean_prob = grouped[prob_cols].mean().reset_index()
    mean_prob["method"] = "mean_probability"

    percent_positive = grouped[prob_cols].apply(lambda g: (g >= threshold).mean()).reset_index()
    percent_positive["method"] = "percent_positive"

    max_prob = grouped[prob_cols].max().reset_index()
    max_prob["method"] = "max"

    out = pd.concat([mean_prob, percent_positive, max_prob], ignore_index=True)
    return out[[slide_col, "method"] + prob_cols]


def optimal_threshold_youden(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Point on the ROC curve closest to (0, 1) -- Coudray et al.'s
    definition of "optimal threshold" (used for their 0.4/0.6 cutoff)."""
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    dist = fpr ** 2 + (1 - tpr) ** 2
    return float(thresholds[np.argmin(dist)])


def fdr_threshold(negative_scores: np.ndarray, fdr: float = 0.05) -> float:
    """Kosaraju et al.'s thresholding approach: pick the score cutoff such
    that only `fdr` fraction of known-negative slides would be called
    positive (fit on a held-out negative-only set, not the test set)."""
    return float(np.quantile(negative_scores, 1 - fdr))
