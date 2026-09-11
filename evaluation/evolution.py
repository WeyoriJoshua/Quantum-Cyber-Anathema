from __future__ import annotations

import numpy as np


def composite_defense_score(metrics: dict) -> float:
    """
    Primary QCA evolution score.

    Uses three complementary imbalanced-classification metrics:
        F1, MCC, PR-AUC

    MCC is shifted from [-1,1] to [0,1] before averaging.
    """
    f1 = float(metrics["F1"])
    mcc01 = (float(metrics["MCC"]) + 1.0) / 2.0
    prauc = float(metrics["PR-AUC"])
    return float(np.mean([f1, mcc01, prauc]))


def defensive_intelligence_gain(score_t: float, score_0: float) -> float:
    """Operational QCA proxy: improvement relative to the initial defensive score."""
    return float(score_t - score_0)


def recursive_evolution_index(
    score_t: float,
    score_prev: float,
    epsilon: float = 1e-12,
) -> float:
    """Relative cycle-to-cycle defensive improvement."""
    return float((score_t - score_prev) / (abs(score_prev) + epsilon))


def normalized_binary_entropy(probabilities, epsilon: float = 1e-7):
    p = np.asarray(probabilities, dtype=float)
    p = np.clip(p, epsilon, 1.0 - epsilon)
    h = -(p * np.log(p) + (1-p) * np.log(1-p)) / np.log(2.0)
    return h


def empirical_qecr_proxy(
    baseline_probabilities,
    current_probabilities,
    *,
    epsilon: float = 1e-12,
) -> float:
    """
    Empirical information-theoretic proxy for the manuscript's QECR concept.

    It measures reduction in predictive entropy on the same fixed evaluation set.
    It is *not* a claim of thermodynamic entropy conversion.
    """
    h0 = float(np.mean(normalized_binary_entropy(baseline_probabilities)))
    ht = float(np.mean(normalized_binary_entropy(current_probabilities)))
    return float((h0 - ht) / (h0 + epsilon))
