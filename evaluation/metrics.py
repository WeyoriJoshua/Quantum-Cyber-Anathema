from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)


def _safe_metric(func, *args, default=float("nan"), **kwargs):
    try:
        return float(func(*args, **kwargs))
    except Exception:
        return float(default)


def predict_scores(model: Any, X) -> np.ndarray:
    """Return P(y=1) or a monotonic score when probability is unavailable."""
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X))
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1].astype(float)
        return proba.reshape(-1).astype(float)

    if hasattr(model, "decision_function"):
        score = np.asarray(model.decision_function(X)).reshape(-1)
        score = np.clip(score, -50, 50)
        return (1.0 / (1.0 + np.exp(-score))).astype(float)

    pred = np.asarray(model.predict(X)).reshape(-1)
    return pred.astype(float)


def classification_metrics(
    y_true,
    y_pred,
    y_score=None,
    *,
    inference_seconds: float | None = None,
    n_samples: int | None = None,
) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_pred = np.asarray(y_pred).astype(int).reshape(-1)
    if y_score is None:
        y_score = y_pred.astype(float)
    y_score = np.asarray(y_score, dtype=float).reshape(-1)
    y_score = np.clip(y_score, 1e-7, 1 - 1e-7)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    fnr = fn / (fn + tp) if (fn + tp) else float("nan")

    metrics = {
        "Accuracy": _safe_metric(accuracy_score, y_true, y_pred),
        "Balanced Accuracy": _safe_metric(balanced_accuracy_score, y_true, y_pred),
        "Precision": _safe_metric(precision_score, y_true, y_pred, zero_division=0),
        "Recall": _safe_metric(recall_score, y_true, y_pred, zero_division=0),
        "Specificity": float(specificity),
        "F1": _safe_metric(f1_score, y_true, y_pred, zero_division=0),
        "ROC-AUC": _safe_metric(roc_auc_score, y_true, y_score),
        "PR-AUC": _safe_metric(average_precision_score, y_true, y_score),
        "MCC": _safe_metric(matthews_corrcoef, y_true, y_pred),
        "Cohen Kappa": _safe_metric(cohen_kappa_score, y_true, y_pred),
        "Log Loss": _safe_metric(log_loss, y_true, np.c_[1-y_score, y_score], labels=[0,1]),
        "Brier Score": _safe_metric(brier_score_loss, y_true, y_score),
        "FPR": float(fpr),
        "FNR": float(fnr),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }

    if inference_seconds is not None:
        metrics["Inference Seconds"] = float(inference_seconds)
        n = int(n_samples if n_samples is not None else len(y_true))
        metrics["Throughput Samples/Sec"] = (
            float(n / inference_seconds) if inference_seconds > 0 else float("inf")
        )

    return metrics


def select_threshold(
    y_true,
    y_score,
    *,
    metric: str = "mcc",
    minimum: float = 0.05,
    maximum: float = 0.95,
    steps: int = 181,
):
    """Select a classification threshold from validation data only."""
    import pandas as pd

    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_score = np.asarray(y_score, dtype=float).reshape(-1)
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same length")
    if metric.lower() not in {"mcc", "f1"}:
        raise ValueError("metric must be 'mcc' or 'f1'")

    rows = []
    for threshold in np.linspace(minimum, maximum, steps):
        pred = (y_score >= threshold).astype(int)
        if metric.lower() == "mcc":
            value = matthews_corrcoef(y_true, pred)
        else:
            value = f1_score(y_true, pred, zero_division=0)
        rows.append({"Threshold": float(threshold), "Score": float(value)})

    table = pd.DataFrame(rows)
    best = float(table["Score"].max())
    candidates = table[np.isclose(table["Score"], best)].copy()
    candidates["Distance from 0.5"] = (candidates["Threshold"] - 0.5).abs()
    chosen = candidates.sort_values(["Distance from 0.5", "Threshold"]).iloc[0]
    return float(chosen["Threshold"]), best, table
