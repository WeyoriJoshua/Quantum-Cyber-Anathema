from __future__ import annotations

try:
    import torch
except Exception:
    torch = None

from .loss import reflexive_signals


def _require_torch():
    if torch is None:
        raise ImportError("PyTorch is required for QCA-v2 training.")


def boundary_hardness(probabilities):
    """Differentiable-free diagnostic hardness around the p=0.5 boundary.

    Returns 1 at p=0.5 and 0 at p in {0, 1}. It is used for memory priority,
    not as a gradient path through the replay-selection mechanism.
    """
    _require_torch()
    p = torch.clamp(probabilities, 0.0, 1.0)
    return torch.clamp(1.0 - 2.0 * torch.abs(p - 0.5), 0.0, 1.0)


def probability_margin_penalty(
    probabilities,
    targets,
    *,
    margin: float = 0.15,
):
    """Hinge-like probability-space separation penalty.

    Positive samples are encouraged toward p >= 0.5 + margin and negative
    samples toward p <= 0.5 - margin. The term is zero once the requested
    separation has been achieved.
    """
    _require_torch()
    if not 0.0 <= margin < 0.5:
        raise ValueError("margin must be in [0, 0.5).")

    p = torch.clamp(probabilities, 1e-7, 1.0 - 1e-7)
    y = targets.to(dtype=p.dtype)
    positive_penalty = torch.relu((0.5 + margin) - p)
    negative_penalty = torch.relu(p - (0.5 - margin))
    return y * positive_penalty + (1.0 - y) * negative_penalty


def qca_v2_reflexive_loss(
    probabilities,
    targets,
    *,
    alpha_error: float = 1.0,
    beta_uncertainty: float = 0.1,
    gamma_malicious: float = 0.25,
    margin_lambda: float = 0.0,
    margin: float = 0.15,
    epsilon: float = 1e-7,
):
    """QCA-v2 loss = reflexively weighted BCE + optional margin separation.

    The existing QCA weighting is preserved exactly as the base objective. The
    v2 margin term is deliberately additive, so a zero margin_lambda recovers
    the original weighted-BCE training objective.
    """
    _require_torch()
    if margin_lambda < 0:
        raise ValueError("margin_lambda must be non-negative.")

    p = torch.clamp(probabilities, epsilon, 1.0 - epsilon)
    y = targets.to(dtype=p.dtype)

    error, uncertainty, importance = reflexive_signals(
        p,
        y,
        alpha_error=alpha_error,
        beta_uncertainty=beta_uncertainty,
        gamma_malicious=gamma_malicious,
        epsilon=epsilon,
    )
    weights = (1.0 + importance).detach()

    per_sample_bce = -(
        y * torch.log(p)
        + (1.0 - y) * torch.log(1.0 - p)
    )
    reflexive_bce = (weights * per_sample_bce).sum() / weights.sum()

    margin_per_sample = probability_margin_penalty(
        p,
        y,
        margin=margin,
    )
    weighted_margin = (weights * margin_per_sample).sum() / weights.sum()
    total = reflexive_bce + float(margin_lambda) * weighted_margin

    hardness = boundary_hardness(p)

    return (
        total,
        reflexive_bce.detach(),
        weighted_margin.detach(),
        error.detach(),
        uncertainty.detach(),
        importance.detach(),
        hardness.detach(),
        weights.detach(),
    )
