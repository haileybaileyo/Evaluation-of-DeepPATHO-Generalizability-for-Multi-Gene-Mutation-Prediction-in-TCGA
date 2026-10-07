"""Binary classification metrics shared by every EGFR experiment script.

Centralizes what egfr_scratch_train.py / roadmap_egfr_train.py / a UNI
variant all need: Precision/Recall/F1 (positive class) plus ROC-AUC and
PR-AUC, computed the same way at both patch level (this project's actual
training unit) and slide level (patch predictions aggregated per slide via
mil/aggregation.py's mean-probability, matching Coudray et al.'s convention).

AUC alone is explicitly insufficient here: EGFR is ~12% positive, so a
model that always predicts "negative" scores ~0.5 balanced accuracy and can
still score a non-trivial AUC on a small test set while having zero
recall -- exactly what runs/egfr_scratch and runs/egfr_roadmap show
(confusion matrices with 0 predicted positives). Precision/Recall/F1 at a
fixed 0.5 threshold surface that failure mode directly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict:
    """Precision/Recall/F1 (positive class, at `threshold`) + ROC-AUC + PR-AUC.
    NaN for the rank metrics if the sample has only one class present (can't
    define an ROC/PR curve without both classes -- happens on small,
    imbalanced test slides)."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)

    has_both = len(np.unique(y_true)) == 2
    return {
        "n": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if has_both else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_score)) if has_both else float("nan"),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "threshold": threshold,
    }


def find_best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Threshold (of the candidate values actually taken by y_score) that
    maximizes F1 on this sample. Must be called on VALIDATION data only,
    never on test -- picking a threshold on the test set leaks test-set
    information into the reported result.

    Grid is the sample's own scores (plus 0 and 1) rather than a fixed
    linspace: with only a few hundred positive examples, a linspace can
    straddle a score cluster and miss the actual optimum, while candidate
    thresholds drawn from observed scores always land exactly at a
    decision boundary.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    candidates = np.unique(np.concatenate([[0.0, 1.0], y_score]))
    best_t, best_f1 = 0.5, -1.0
    for t in candidates:
        f1 = f1_score(y_true, (y_score >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def patch_level_metrics(patch_df: pd.DataFrame, threshold: float = 0.5) -> dict:
    """patch_df: one row per patch, columns [prob, label]. This is the raw
    training unit -- see docs/LIMITATIONS for why a positive slide's patches
    are not all individually verified positive (label noise)."""
    return binary_metrics(patch_df["label"].values, patch_df["prob"].values, threshold)


def slide_level_metrics(patch_df: pd.DataFrame, threshold: float = 0.5) -> tuple[dict, pd.DataFrame]:
    """Aggregate patch_df (one row per patch: [file_id, prob, label]) to one
    row per slide via mean-probability pooling, then score. Returns
    (metrics_dict, slide_level_dataframe)."""
    slide = patch_df.groupby("file_id").agg(prob=("prob", "mean"), label=("label", "first"))
    m = binary_metrics(slide["label"].values, slide["prob"].values, threshold)
    m["n_slides"] = m.pop("n")
    return m, slide
