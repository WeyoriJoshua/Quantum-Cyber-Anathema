from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
except Exception:
    torch = None
    DataLoader = None
    TensorDataset = None

from .pennylane_vqc import PennyLaneVQC, TrainingHistory, evaluate_loss


def _require_torch():
    if torch is None:
        raise ImportError("PyTorch is required for weighted VQC training.")


def balanced_class_weights(y) -> tuple[float, float]:
    """Return inverse-frequency binary class weights fitted on training labels only.

    The weights are normalized so their frequency-weighted average is one:

        w_c = N / (2 * N_c)

    This keeps the overall loss scale comparable to ordinary BCE while giving
    the minority class proportionally greater influence.
    """
    y = np.asarray(y).astype(int).reshape(-1)
    n0 = int(np.sum(y == 0))
    n1 = int(np.sum(y == 1))
    n = len(y)
    if n0 == 0 or n1 == 0:
        raise ValueError("Both binary classes are required to compute class weights.")
    return float(n / (2.0 * n0)), float(n / (2.0 * n1))


def weighted_binary_cross_entropy(
    probabilities,
    targets,
    *,
    weight_negative: float,
    weight_positive: float,
    epsilon: float = 1e-7,
):
    _require_torch()
    if weight_negative <= 0 or weight_positive <= 0:
        raise ValueError("Class weights must be positive.")

    p = torch.clamp(probabilities, epsilon, 1.0 - epsilon)
    y = targets.to(dtype=p.dtype)
    weights = torch.where(
        y >= 0.5,
        torch.as_tensor(weight_positive, dtype=p.dtype, device=p.device),
        torch.as_tensor(weight_negative, dtype=p.dtype, device=p.device),
    )
    per_sample = -(y * torch.log(p) + (1.0 - y) * torch.log(1.0 - p))
    return (weights * per_sample).sum() / weights.sum()


def _loader(X, y, batch_size: int, shuffle: bool):
    _require_torch()
    x = torch.as_tensor(np.asarray(X), dtype=torch.float64)
    target = torch.as_tensor(np.asarray(y), dtype=torch.float64)
    return DataLoader(
        TensorDataset(x, target),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def fit_weighted_vqc(
    model: PennyLaneVQC,
    X_train,
    y_train,
    X_val,
    y_val,
    *,
    epochs: int = 60,
    batch_size: int = 16,
    learning_rate: float = 0.01,
    patience: int = 10,
    checkpoint_path: str | Path | None = None,
    verbose: bool = True,
):
    """Train a class-balanced static VQC control.

    Training uses inverse-frequency weighted BCE fitted from the development
    training partition only. Early stopping deliberately uses the same ordinary
    validation BCE as the unweighted VQC baseline, isolating the effect of the
    training loss rather than changing the stopping criterion as well.
    """
    _require_torch()

    w0, w1 = balanced_class_weights(y_train)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = TrainingHistory([], [], [])
    best_val = float("inf")
    best_state = None
    stale = 0

    train_loader = _loader(X_train, y_train, batch_size, shuffle=True)

    for epoch in range(1, int(epochs) + 1):
        model.train()
        epoch_losses = []
        for xb, yb in train_loader:
            optimizer.zero_grad()
            p = model(xb)
            loss = weighted_binary_cross_entropy(
                p,
                yb,
                weight_negative=w0,
                weight_positive=w1,
            )
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        val_loss = evaluate_loss(model, X_val, y_val, batch_size=batch_size)
        history.epoch.append(epoch)
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)

        if verbose:
            print(
                f"Epoch {epoch:03d} | weighted_train={train_loss:.6f} "
                f"| ordinary_val={val_loss:.6f}"
            )

        if val_loss < best_val - 1e-8:
            best_val = val_loss
            stale = 0
            best_state = {
                k: v.detach().clone()
                for k, v in model.state_dict().items()
            }
            if checkpoint_path is not None:
                path = Path(checkpoint_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, path)
        else:
            stale += 1
            if stale >= int(patience):
                if verbose:
                    print(f"Early stopping after epoch {epoch}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.class_weight_negative_ = float(w0)
    model.class_weight_positive_ = float(w1)
    model.weighted_vqc_best_val_loss_ = float(best_val)
    return history
