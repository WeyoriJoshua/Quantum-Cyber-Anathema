from __future__ import annotations

try:
    import torch
except Exception:
    torch = None


def _require_torch():
    if torch is None:
        raise ImportError("PyTorch is required for QCA training.")


def reflexive_signals(
    probabilities,
    targets,
    *,
    alpha_error: float = 1.0,
    beta_uncertainty: float = 0.5,
    gamma_malicious: float = 0.25,
    epsilon: float = 1e-7,
):
    """
    Compute QCA's operational reflexive signals:
      error       = |y - p|
      uncertainty = normalized binary predictive entropy
      importance  = alpha*error + beta*uncertainty + gamma*y
    """
    _require_torch()

    p = torch.clamp(probabilities, epsilon, 1.0 - epsilon)
    y = targets.to(dtype=p.dtype)

    error = torch.abs(y - p)
    entropy = -(
        p * torch.log(p)
        + (1.0 - p) * torch.log(1.0 - p)
    ) / torch.log(torch.tensor(2.0, dtype=p.dtype, device=p.device))

    importance = (
        alpha_error * error
        + beta_uncertainty * entropy
        + gamma_malicious * y
    )
    return error, entropy, importance


def qca_reflexive_loss(
    probabilities,
    targets,
    *,
    alpha_error: float = 1.0,
    beta_uncertainty: float = 0.5,
    gamma_malicious: float = 0.25,
    epsilon: float = 1e-7,
):
    """
    Weighted BCE operationalizing Quantum Cyber Anathema reflexive learning.

    Weight gradients are detached so the adaptive weighting mechanism changes
    sample influence without creating a second-order shortcut through the weight.
    """
    _require_torch()

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

    loss = (weights * per_sample_bce).sum() / weights.sum()

    return (
        loss,
        error.detach(),
        uncertainty.detach(),
        importance.detach(),
        weights.detach(),
    )
