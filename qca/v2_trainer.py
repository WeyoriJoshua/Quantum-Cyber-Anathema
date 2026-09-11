from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import TensorDataset, DataLoader
except Exception:
    torch = None
    TensorDataset = None
    DataLoader = None

from evaluation.metrics import classification_metrics
from evaluation.evolution import (
    composite_defense_score,
    defensive_intelligence_gain,
    recursive_evolution_index,
    empirical_qecr_proxy,
)
from models.pennylane_vqc import fit_vqc
from .memory import AdversarialMemory
from .v2_loss import qca_v2_reflexive_loss, boundary_hardness
from .loss import reflexive_signals


def _require_torch():
    if torch is None:
        raise ImportError("PyTorch is required for QCA-v2.")


def _make_loader(X, y, batch_size: int, shuffle: bool = True):
    _require_torch()
    X_t = torch.as_tensor(np.asarray(X), dtype=torch.float64)
    y_t = torch.as_tensor(np.asarray(y), dtype=torch.float64)
    return DataLoader(
        TensorDataset(X_t, y_t),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def _evaluate_probabilities(model, X, *, batch_size: int = 64):
    _require_torch()
    model.eval()
    out = []
    dummy = np.zeros(len(X))
    with torch.no_grad():
        for xb, _ in _make_loader(X, dummy, batch_size, shuffle=False):
            out.append(model(xb).cpu().numpy())
    if not out:
        return np.empty((0,), dtype=float)
    return np.concatenate(out).astype(float)


def predict_qca_v2(model, X, *, batch_size: int = 64):
    return _evaluate_probabilities(model, X, batch_size=batch_size)


def _build_cycle_data(
    X_train,
    y_train,
    memory: AdversarialMemory,
    replay_ratio: float,
    *,
    prioritized_replay: bool = True,
):
    X = np.asarray(X_train, dtype=np.float32)
    y = np.asarray(y_train, dtype=np.int64)

    if len(memory) == 0 or replay_ratio <= 0:
        return X, y, 0

    replay_ratio = min(max(float(replay_ratio), 0.0), 0.95)
    n_replay = int(round(len(X) * replay_ratio / max(1e-12, 1.0 - replay_ratio)))
    X_mem, y_mem = memory.sample(n_replay, prioritized=prioritized_replay)
    if len(y_mem) == 0:
        return X, y, 0

    return (
        np.concatenate([X, X_mem], axis=0),
        np.concatenate([y, y_mem], axis=0),
        int(len(y_mem)),
    )


def _update_memory(
    model,
    X,
    y,
    memory: AdversarialMemory,
    *,
    cycle: int,
    alpha_error: float,
    beta_uncertainty: float,
    gamma_malicious: float,
    replay_priority_mode: str,
    boundary_weight: float,
    boundary_threshold: float,
    batch_size: int,
):
    """Populate memory from DEVELOPMENT TRAINING ONLY.

    `reflexive` reproduces current-QCA sample priority.
    `boundary` prioritizes misclassified and near-boundary examples using
    error + boundary hardness + malicious-class priority.
    """
    _require_torch()
    if replay_priority_mode not in {"reflexive", "boundary"}:
        raise ValueError("replay_priority_mode must be 'reflexive' or 'boundary'.")

    model.eval()
    loader = _make_loader(X, y, batch_size, shuffle=False)

    with torch.no_grad():
        for xb, yb in loader:
            p = model(xb)
            error, uncertainty, importance = reflexive_signals(
                p,
                yb,
                alpha_error=alpha_error,
                beta_uncertainty=beta_uncertainty,
                gamma_malicious=gamma_malicious,
            )
            hard_boundary = boundary_hardness(p)

            xb_np = xb.cpu().numpy()
            y_np = yb.cpu().numpy().astype(int)
            p_np = p.cpu().numpy()
            e_np = error.cpu().numpy()
            u_np = uncertainty.cpu().numpy()
            r_np = importance.cpu().numpy()
            b_np = hard_boundary.cpu().numpy()

            for i in range(len(y_np)):
                pred = int(p_np[i] >= 0.5)
                misclassified = pred != int(y_np[i])

                if replay_priority_mode == "boundary":
                    hard = bool(misclassified or b_np[i] >= boundary_threshold)
                    priority = (
                        alpha_error * float(e_np[i])
                        + boundary_weight * float(b_np[i])
                        + gamma_malicious * float(y_np[i])
                    )
                else:
                    hard = bool(
                        e_np[i] >= 0.50
                        or u_np[i] >= 0.50
                        or (y_np[i] == 1 and p_np[i] < 0.5)
                    )
                    priority = float(r_np[i])

                if hard:
                    memory.add(
                        xb_np[i],
                        y_np[i],
                        priority=float(priority),
                        error=float(e_np[i]),
                        uncertainty=float(u_np[i]),
                        cycle=cycle,
                    )


def fit_qca_v2(
    model,
    X_train,
    y_train,
    X_inner_val,
    y_inner_val,
    *,
    cycles: int = 4,
    epochs_per_cycle: int = 10,
    warmup_epochs: int = 20,
    warmup_patience: int = 8,
    batch_size: int = 16,
    learning_rate: float = 0.01,
    memory_capacity: int = 600,
    replay_ratio: float = 0.20,
    alpha_error: float = 1.0,
    beta_uncertainty: float = 0.1,
    gamma_malicious: float = 0.25,
    margin_lambda: float = 0.0,
    margin: float = 0.15,
    replay_priority_mode: str = "reflexive",
    boundary_weight: float = 1.0,
    boundary_threshold: float = 0.50,
    prioritized_replay: bool = True,
    checkpoint_dir: str | Path | None = None,
    restore_best: bool = True,
    seed: int = 42,
    verbose: bool = True,
):
    """Validation-only QCA-v2 development training.

    Critical safeguard: this function receives only development-training and
    inner-validation arrays. It has no test-set argument. Cycle selection uses
    the inner-validation Defense Score. An outer validation set must be handled
    only by the caller after the best cycle has been restored.
    """
    _require_torch()
    if cycles < 0 or epochs_per_cycle < 1:
        raise ValueError("cycles must be >=0 and epochs_per_cycle must be >=1")
    if warmup_epochs < 0:
        raise ValueError("warmup_epochs must be >=0")
    if not 0.0 <= replay_ratio < 1.0:
        raise ValueError("replay_ratio must be in [0,1).")
    if margin_lambda < 0:
        raise ValueError("margin_lambda must be non-negative.")
    if not 0.0 <= margin < 0.5:
        raise ValueError("margin must be in [0,0.5).")
    if not 0.0 <= boundary_threshold <= 1.0:
        raise ValueError("boundary_threshold must be in [0,1].")
    if boundary_weight < 0:
        raise ValueError("boundary_weight must be non-negative.")

    torch.manual_seed(seed)
    np.random.seed(seed)

    warmup_history = None
    if warmup_epochs > 0:
        if verbose:
            print(f"Warm-up ordinary VQC for up to {warmup_epochs} epochs...")
        warmup_history = fit_vqc(
            model,
            X_train,
            y_train,
            X_inner_val,
            y_inner_val,
            epochs=warmup_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            patience=warmup_patience,
            checkpoint_path=None,
            verbose=verbose,
        )

    memory = AdversarialMemory(capacity=memory_capacity, seed=seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history_rows = []
    baseline_probabilities = None
    baseline_score = None
    previous_score = None
    best_score = -np.inf
    best_cycle = 0
    best_state = None

    for cycle in range(cycles + 1):
        replay_samples = 0
        mean_total_loss = float("nan")
        mean_bce_loss = float("nan")
        mean_margin_loss = float("nan")

        if cycle > 0:
            X_cycle, y_cycle, replay_samples = _build_cycle_data(
                X_train,
                y_train,
                memory,
                replay_ratio,
                prioritized_replay=prioritized_replay,
            )
            loader = _make_loader(
                X_cycle,
                y_cycle,
                batch_size=batch_size,
                shuffle=True,
            )

            cycle_total_losses = []
            cycle_bce_losses = []
            cycle_margin_losses = []

            for epoch in range(1, epochs_per_cycle + 1):
                model.train()
                epoch_losses = []
                for xb, yb in loader:
                    optimizer.zero_grad()
                    p = model(xb)
                    (
                        loss,
                        base_bce,
                        margin_loss,
                        *_rest,
                    ) = qca_v2_reflexive_loss(
                        p,
                        yb,
                        alpha_error=alpha_error,
                        beta_uncertainty=beta_uncertainty,
                        gamma_malicious=gamma_malicious,
                        margin_lambda=margin_lambda,
                        margin=margin,
                    )
                    loss.backward()
                    optimizer.step()

                    cycle_total_losses.append(float(loss.item()))
                    cycle_bce_losses.append(float(base_bce.item()))
                    cycle_margin_losses.append(float(margin_loss.item()))
                    epoch_losses.append(float(loss.item()))

                if verbose:
                    mean_epoch = (
                        float(np.mean(epoch_losses)) if epoch_losses else float("nan")
                    )
                    print(
                        f"Cycle {cycle:02d}/{cycles:02d} "
                        f"Epoch {epoch:03d}/{epochs_per_cycle:03d} "
                        f"Loss={mean_epoch:.6f} Replay={replay_samples}"
                    )

            if cycle_total_losses:
                mean_total_loss = float(np.mean(cycle_total_losses))
                mean_bce_loss = float(np.mean(cycle_bce_losses))
                mean_margin_loss = float(np.mean(cycle_margin_losses))

        t0 = perf_counter()
        val_prob = _evaluate_probabilities(
            model,
            X_inner_val,
            batch_size=max(batch_size, 32),
        )
        infer_seconds = perf_counter() - t0
        val_pred = (val_prob >= 0.5).astype(int)
        metrics = classification_metrics(
            y_inner_val,
            val_pred,
            val_prob,
            inference_seconds=infer_seconds,
            n_samples=len(y_inner_val),
        )
        score = composite_defense_score(metrics)

        if cycle == 0:
            baseline_probabilities = val_prob.copy()
            baseline_score = score
            dig = 0.0
            rei = 0.0
            qecr = 0.0
        else:
            dig = defensive_intelligence_gain(score, baseline_score)
            rei = recursive_evolution_index(score, previous_score)
            qecr = empirical_qecr_proxy(baseline_probabilities, val_prob)

        if np.isfinite(score) and score > best_score:
            best_score = score
            best_cycle = cycle
            best_state = {
                k: v.detach().clone()
                for k, v in model.state_dict().items()
            }

        if checkpoint_dir is not None:
            checkpoint_path = Path(checkpoint_dir)
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            torch.save(
                model.state_dict(),
                checkpoint_path / f"qca_v2_cycle_{cycle:02d}.pt",
            )

        _update_memory(
            model,
            X_train,
            y_train,
            memory,
            cycle=cycle,
            alpha_error=alpha_error,
            beta_uncertainty=beta_uncertainty,
            gamma_malicious=gamma_malicious,
            replay_priority_mode=replay_priority_mode,
            boundary_weight=boundary_weight,
            boundary_threshold=boundary_threshold,
            batch_size=max(batch_size, 32),
        )
        mem = memory.summary()

        history_rows.append(
            {
                "Cycle": cycle,
                **metrics,
                "Defense Score": score,
                "DIG Proxy": dig,
                "REI Proxy": rei,
                "QECR Proxy": qecr,
                "Replay Samples": replay_samples,
                "Memory Size": len(memory),
                "Memory Mean Priority": mem.get("mean_priority", 0.0),
                "Memory Malicious Fraction": mem.get("malicious_fraction", 0.0),
                "Mean Cycle Total Loss": mean_total_loss,
                "Mean Cycle Reflexive BCE": mean_bce_loss,
                "Mean Cycle Margin Penalty": mean_margin_loss,
                "Margin Lambda": float(margin_lambda),
                "Probability Margin": float(margin),
                "Replay Priority Mode": replay_priority_mode,
                "Boundary Weight": float(boundary_weight),
                "Boundary Threshold": float(boundary_threshold),
            }
        )
        previous_score = score

        if verbose:
            print(
                f"[Cycle {cycle}] F1={metrics['F1']:.4f} "
                f"MCC={metrics['MCC']:.4f} "
                f"PR-AUC={metrics['PR-AUC']:.4f} "
                f"DefenseScore={score:.4f} Memory={len(memory)}"
            )

    history = pd.DataFrame(history_rows)
    history["Selected by Inner Validation"] = history["Cycle"].eq(best_cycle)

    if restore_best and best_state is not None:
        model.load_state_dict(best_state)

    model.qca_v2_best_cycle_ = int(best_cycle)
    model.qca_v2_best_inner_validation_score_ = float(best_score)
    model.qca_v2_restored_best_ = bool(restore_best)
    model.qca_v2_margin_lambda_ = float(margin_lambda)
    model.qca_v2_margin_ = float(margin)
    model.qca_v2_replay_priority_mode_ = replay_priority_mode
    model.qca_v2_warmup_history_ = (
        warmup_history.to_dict() if warmup_history is not None else None
    )

    return history, memory
