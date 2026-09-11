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
from .loss import qca_reflexive_loss
from .memory import AdversarialMemory


def _require_torch():
    if torch is None:
        raise ImportError("PyTorch is required for QCA.")


def _make_loader(X, y, batch_size: int, shuffle: bool = True):
    _require_torch()
    X_t = torch.as_tensor(np.asarray(X), dtype=torch.float64)
    y_t = torch.as_tensor(np.asarray(y), dtype=torch.float64)
    return DataLoader(
        TensorDataset(X_t, y_t),
        batch_size=batch_size,
        shuffle=shuffle,
    )


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
    X_mem, y_mem = memory.sample(
        n_replay,
        prioritized=prioritized_replay,
    )
    if len(y_mem) == 0:
        return X, y, 0

    return (
        np.concatenate([X, X_mem], axis=0),
        np.concatenate([y, y_mem], axis=0),
        int(len(y_mem)),
    )


def _evaluate_probabilities(model, X, *, batch_size=64):
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


def predict_qca(model, X, *, batch_size=64):
    return _evaluate_probabilities(model, X, batch_size=batch_size)


def _update_memory_from_dataset(
    model,
    X,
    y,
    memory: AdversarialMemory,
    *,
    cycle: int,
    alpha_error: float,
    beta_uncertainty: float,
    gamma_malicious: float,
    batch_size: int = 64,
    only_hard: bool = True,
    hardness_threshold: float = 0.50,
):
    _require_torch()
    model.eval()

    loader = _make_loader(X, y, batch_size, shuffle=False)

    with torch.no_grad():
        for xb, yb in loader:
            p = model(xb)
            _, errors, uncertainty, importance, _ = qca_reflexive_loss(
                p,
                yb,
                alpha_error=alpha_error,
                beta_uncertainty=beta_uncertainty,
                gamma_malicious=gamma_malicious,
            )

            xb_np = xb.cpu().numpy()
            yb_np = yb.cpu().numpy().astype(int)
            p_np = p.cpu().numpy()
            e_np = errors.cpu().numpy()
            u_np = uncertainty.cpu().numpy()
            r_np = importance.cpu().numpy()

            for i in range(len(yb_np)):
                hard = (
                    e_np[i] >= hardness_threshold
                    or u_np[i] >= hardness_threshold
                    or (yb_np[i] == 1 and p_np[i] < 0.5)
                )
                if (not only_hard) or hard:
                    memory.add(
                        xb_np[i],
                        yb_np[i],
                        priority=float(r_np[i]),
                        error=float(e_np[i]),
                        uncertainty=float(u_np[i]),
                        cycle=cycle,
                    )


def fit_qca(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    *,
    cycles: int = 5,
    epochs_per_cycle: int = 10,
    warmup_epochs: int = 10,
    warmup_patience: int = 4,
    batch_size: int = 16,
    learning_rate: float = 0.01,
    memory_capacity: int = 1000,
    replay_ratio: float = 0.20,
    alpha_error: float = 1.0,
    beta_uncertainty: float = 0.5,
    gamma_malicious: float = 0.25,
    prioritized_replay: bool = True,
    checkpoint_dir: str | Path | None = None,
    restore_best: bool = True,
    seed: int = 42,
    verbose: bool = True,
):
    """Train QCA across recursive hostile-exposure cycles.

    Methodological safeguards:
    * Cycle 0 is a *trained ordinary VQC warm start*, not a random model. This
      prevents trivial apparent gains caused merely by learning from random
      initialization.
    * Adversarial memory is populated only from the training partition.
    * Validation data stay fixed and are never replayed.
    * The best cycle is selected using validation Defense Score; the independent
      test set remains untouched until the caller performs final evaluation.
    """
    _require_torch()
    if cycles < 0 or epochs_per_cycle < 1:
        raise ValueError("cycles must be >=0 and epochs_per_cycle must be >=1")
    if warmup_epochs < 0:
        raise ValueError("warmup_epochs must be >=0")
    if not 0.0 <= replay_ratio < 1.0:
        raise ValueError("replay_ratio must be in [0, 1).")
    for name, value in {
        "alpha_error": alpha_error,
        "beta_uncertainty": beta_uncertainty,
        "gamma_malicious": gamma_malicious,
    }.items():
        if value < 0:
            raise ValueError(f"{name} must be non-negative.")

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
            X_val,
            y_val,
            epochs=warmup_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            patience=warmup_patience,
            checkpoint_path=None,
            verbose=verbose,
        )
    elif verbose:
        print("WARNING: warmup_epochs=0; Cycle 0 is an untrained random baseline.")

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

            for epoch in range(1, epochs_per_cycle + 1):
                model.train()
                losses = []

                for xb, yb in loader:
                    optimizer.zero_grad()
                    p = model(xb)
                    loss, _, _, _, _ = qca_reflexive_loss(
                        p,
                        yb,
                        alpha_error=alpha_error,
                        beta_uncertainty=beta_uncertainty,
                        gamma_malicious=gamma_malicious,
                    )
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.item()))

                if verbose:
                    mean_loss = float(np.mean(losses)) if losses else float("nan")
                    print(
                        f"Cycle {cycle:02d}/{cycles:02d} "
                        f"Epoch {epoch:03d}/{epochs_per_cycle:03d} "
                        f"Loss={mean_loss:.6f} Replay={replay_samples}"
                    )

        t0 = perf_counter()
        val_prob = _evaluate_probabilities(
            model,
            X_val,
            batch_size=max(batch_size, 32),
        )
        infer_seconds = perf_counter() - t0
        val_pred = (val_prob >= 0.5).astype(int)

        metrics = classification_metrics(
            y_val,
            val_pred,
            val_prob,
            inference_seconds=infer_seconds,
            n_samples=len(y_val),
        )
        score = composite_defense_score(metrics)

        if cycle == 0:
            baseline_probabilities = val_prob.copy()
            baseline_score = score
            rei = 0.0
            dig = 0.0
            qecr = 0.0
        else:
            dig = defensive_intelligence_gain(score, baseline_score)
            rei = recursive_evolution_index(score, previous_score)
            qecr = empirical_qecr_proxy(
                baseline_probabilities,
                val_prob,
            )

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
                checkpoint_path / f"qca_cycle_{cycle:02d}.pt",
            )

        _update_memory_from_dataset(
            model,
            X_train,
            y_train,
            memory,
            cycle=cycle,
            alpha_error=alpha_error,
            beta_uncertainty=beta_uncertainty,
            gamma_malicious=gamma_malicious,
            batch_size=max(batch_size, 32),
        )
        mem = memory.summary()

        row = {
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
        }
        history_rows.append(row)
        previous_score = score

        if verbose:
            print(
                f"[Cycle {cycle}] F1={metrics['F1']:.4f} "
                f"MCC={metrics['MCC']:.4f} "
                f"PR-AUC={metrics['PR-AUC']:.4f} "
                f"DefenseScore={score:.4f} "
                f"DIG={dig:.4f} REI={rei:.4f} "
                f"Memory={len(memory)}"
            )

    history = pd.DataFrame(history_rows)
    history["Selected for Test"] = history["Cycle"].eq(best_cycle)

    if restore_best and best_state is not None:
        model.load_state_dict(best_state)

    model.qca_best_cycle_ = int(best_cycle)
    model.qca_best_validation_score_ = float(best_score)
    model.qca_restored_best_ = bool(restore_best)
    model.qca_warmup_history_ = (
        warmup_history.to_dict() if warmup_history is not None else None
    )

    return history, memory
