from __future__ import annotations

from pathlib import Path
from time import perf_counter

import pandas as pd

from experiments.common import load_quantum_splits, set_global_seed
from evaluation.metrics import classification_metrics, select_threshold
from .model import QuantumCyberAnathema
from .trainer import fit_qca, predict_qca


def build_ablations(beta_uncertainty: float):
    base = {
        "full_qca": dict(alpha_error=1.0, beta_uncertainty=beta_uncertainty, gamma_malicious=0.25, replay_ratio=0.20),
        "no_error_weighting": dict(alpha_error=0.0, beta_uncertainty=beta_uncertainty, gamma_malicious=0.25, replay_ratio=0.20),
        "no_malicious_priority": dict(alpha_error=1.0, beta_uncertainty=beta_uncertainty, gamma_malicious=0.0, replay_ratio=0.20),
        "no_memory_replay": dict(alpha_error=1.0, beta_uncertainty=beta_uncertainty, gamma_malicious=0.25, replay_ratio=0.0),
    }
    if beta_uncertainty > 0:
        base["no_uncertainty"] = dict(alpha_error=1.0, beta_uncertainty=0.0, gamma_malicious=0.25, replay_ratio=0.20)
    else:
        base["with_uncertainty_0.25"] = dict(alpha_error=1.0, beta_uncertainty=0.25, gamma_malicious=0.25, replay_ratio=0.20)
    return base


ABLATIONS = build_ablations(0.5)


def run_ablation(
    data_path: str | Path,
    out_dir: str | Path,
    *,
    beta_uncertainty: float,
    layers: int = 2,
    cycles: int = 4,
    epochs_per_cycle: int = 10,
    warmup_epochs: int = 20,
    warmup_patience: int = 8,
    batch_size: int = 16,
    learning_rate: float = 0.01,
    memory_capacity: int = 600,
    seed: int = 42,
    threshold_metric: str = "mcc",
    verbose: bool = False,
) -> pd.DataFrame:
    """Run matched QCA ablations and report the validation-selected best cycle."""
    data = load_quantum_splits(data_path)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    summary_rows, all_histories = [], []
    for name, params in build_ablations(beta_uncertainty).items():
        print(f"\n===== {name} =====")
        set_global_seed(seed)
        model = QuantumCyberAnathema(
            n_qubits=X_train.shape[1], n_layers=layers, encoding="angle", seed=seed
        )
        t0 = perf_counter()
        history, memory = fit_qca(
            model, X_train, y_train, X_val, y_val,
            cycles=cycles,
            epochs_per_cycle=epochs_per_cycle,
            warmup_epochs=warmup_epochs,
            warmup_patience=warmup_patience,
            batch_size=batch_size,
            learning_rate=learning_rate,
            memory_capacity=memory_capacity,
            prioritized_replay=True,
            restore_best=True,
            seed=seed,
            verbose=verbose,
            **params,
        )
        train_seconds = perf_counter() - t0
        selected = history.loc[history["Selected for Test"]].iloc[0]

        val_prob = predict_qca(model, X_val, batch_size=max(batch_size, 32))
        threshold, threshold_score, threshold_table = select_threshold(
            y_val, val_prob, metric=threshold_metric
        )
        threshold_table.to_csv(out / f"{name}_threshold_search.csv", index=False)

        t1 = perf_counter()
        test_prob = predict_qca(model, X_test, batch_size=max(batch_size, 32))
        infer_seconds = perf_counter() - t1
        test_pred = (test_prob >= threshold).astype(int)
        test_metrics = classification_metrics(
            y_test, test_pred, test_prob,
            inference_seconds=infer_seconds, n_samples=len(y_test),
        )

        row = {
            "Ablation": name,
            "Selected Cycle": int(selected["Cycle"]),
            "Validation Defense Score": float(selected["Defense Score"]),
            "Validation DIG": float(selected["DIG Proxy"]),
            "Validation REI": float(selected["REI Proxy"]),
            "Validation QECR": float(selected["QECR Proxy"]),
            "Threshold": float(threshold),
            f"Validation Threshold {threshold_metric.upper()}": float(threshold_score),
            "Train Seconds": float(train_seconds),
            "Memory Size": int(len(memory)),
            **{f"Test {k}": v for k, v in test_metrics.items()},
        }
        summary_rows.append(row)

        history = history.copy()
        history.insert(0, "Ablation", name)
        history.to_csv(out / f"{name}_history.csv", index=False)
        all_histories.append(history)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "ablation_summary_corrected.csv", index=False)
    pd.concat(all_histories, ignore_index=True).to_csv(out / "ablation_all_cycles_corrected.csv", index=False)
    return summary
