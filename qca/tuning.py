from __future__ import annotations

import pandas as pd

from experiments.common import set_global_seed
from .model import QuantumCyberAnathema
from .trainer import fit_qca


def tune_uncertainty_beta(
    X_train, y_train, X_val, y_val,
    *,
    candidates=(0.0, 0.10, 0.25, 0.50),
    n_layers: int = 2,
    encoding: str = "angle",
    warmup_epochs: int = 10,
    warmup_patience: int = 4,
    cycles: int = 2,
    epochs_per_cycle: int = 5,
    batch_size: int = 16,
    learning_rate: float = 0.01,
    memory_capacity: int = 500,
    replay_ratio: float = 0.20,
    alpha_error: float = 1.0,
    gamma_malicious: float = 0.25,
    seed: int = 42,
    verbose: bool = False,
):
    """Tune the uncertainty coefficient using validation Defense Score only."""
    rows = []
    for beta in candidates:
        set_global_seed(seed)
        model = QuantumCyberAnathema(
            n_qubits=X_train.shape[1], n_layers=n_layers,
            encoding=encoding, seed=seed,
        )
        history, _ = fit_qca(
            model, X_train, y_train, X_val, y_val,
            warmup_epochs=warmup_epochs,
            warmup_patience=warmup_patience,
            cycles=cycles,
            epochs_per_cycle=epochs_per_cycle,
            batch_size=batch_size,
            learning_rate=learning_rate,
            memory_capacity=memory_capacity,
            replay_ratio=replay_ratio,
            alpha_error=alpha_error,
            beta_uncertainty=float(beta),
            gamma_malicious=gamma_malicious,
            restore_best=True,
            seed=seed,
            verbose=verbose,
        )
        selected = history.loc[history["Selected for Test"]].iloc[0]
        rows.append({
            "Beta": float(beta),
            "Best Cycle": int(selected["Cycle"]),
            "Defense Score": float(selected["Defense Score"]),
            "F1": float(selected["F1"]),
            "MCC": float(selected["MCC"]),
            "PR-AUC": float(selected["PR-AUC"]),
            "ROC-AUC": float(selected["ROC-AUC"]),
        })
    table = pd.DataFrame(rows).sort_values(
        ["Defense Score", "MCC", "PR-AUC"], ascending=False
    ).reset_index(drop=True)
    best_beta = float(table.iloc[0]["Beta"])
    return best_beta, table
