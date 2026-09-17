from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

"""Ginis, KS, PSI and threshold checking shared by notebook and monitoring job"""

DEFAULT_THRESHOLDS = {"gini_min": 0.30, "ks_min": 0.20, "psi_max": 0.25}

def gini(y_true_bin: np.ndarray, y_score: np.ndarray) -> float:
    """Gini = 2*AUC - 1 (binary)"""
    return 2.0 * roc_auc_score(y_true_bin, y_score) - 1.0

def ks(y_true_bin: np.ndarray, y_score: np.ndarray) -> float:
    """Kolmogorov-Smirnov: max |TPR - FPR|"""
    fpr, tpr, _ = roc_curve(y_true_bin, y_score)
    return float(np.max(np.abs(tpr - fpr)))

def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two score distributions."""
    breakpoints = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))

    if len(breakpoints) < 3:
        return 0.0  # Not enough unique breakpoints to calculate PSI

    e_counts, _ = np.histogram(expected, bins=breakpoints)
    a_counts, _ = np.histogram(actual, bins=breakpoints)
    e_pct = np.clip(e_counts / max(e_counts.sum(), 1), 1e-6, None)
    a_pct = np.clip(a_counts / max(a_counts.sum(), 1), 1e-6, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))

def classify(g: float, k: float, p: float, thresholds: dict | None = None) -> str:
    """Classify model performance based on Gini, KS, and PSI thresholds"""
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    if g < t["gini_min"]:
        return "GINI_LOW"
    if k < t["ks_min"]:
        return "KS_LOW"
    if p > t["psi_max"]:
        return "PSI_HIGH"
    return "OK"