from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
import json
import pandas as pd
import torch

from project_config import CONFIG
from experiments.common import load_quantum_splits, save_json, set_global_seed
from evaluation.metrics import classification_metrics, select_threshold
from evaluation.plots import save_classification_plots, save_evolution_plot
from qca.model import QuantumCyberAnathema
from qca.trainer import fit_qca, predict_qca
from qca.tuning import tune_uncertainty_beta


def parse_args():
    p = argparse.ArgumentParser(description="Train Quantum Cyber Anathema with validation-controlled reflexive learning.")
    p.add_argument("--data", default=str(CONFIG.benchmark_splits_path))
    p.add_argument("--out", default=str(CONFIG.project_root / "results" / "qca" / CONFIG.experiment_tag))
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--encoding", choices=["angle", "data_reuploading"], default="angle")
    p.add_argument("--cycles", type=int, default=CONFIG.qca_cycles)
    p.add_argument("--epochs-per-cycle", type=int, default=CONFIG.qca_epochs_per_cycle)
    p.add_argument("--warmup-epochs", type=int, default=CONFIG.qca_warmup_epochs)
    p.add_argument("--warmup-patience", type=int, default=CONFIG.qca_warmup_patience)
    p.add_argument("--batch-size", type=int, default=CONFIG.batch_size)
    p.add_argument("--lr", type=float, default=CONFIG.learning_rate)
    p.add_argument("--memory-capacity", type=int, default=600)
    p.add_argument("--replay-ratio", type=float, default=0.20)
    p.add_argument("--alpha-error", type=float, default=1.0)
    p.add_argument("--gamma-malicious", type=float, default=0.25)
    p.add_argument("--beta-uncertainty", type=float, default=None,
                   help="If omitted, tune beta on validation using 0,0.1,0.25,0.5.")
    p.add_argument("--skip-beta-tuning", action="store_true")
    p.add_argument("--threshold-metric", choices=["mcc", "f1"], default="mcc")
    p.add_argument("--seed", type=int, default=CONFIG.seed)
    return p.parse_args()


def main():
    args = parse_args()
    set_global_seed(args.seed)
    data = load_quantum_splits(args.data)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    if args.beta_uncertainty is not None:
        best_beta = float(args.beta_uncertainty)
        beta_table = pd.DataFrame([{"Beta": best_beta, "Selection": "CLI-fixed"}])
    elif args.skip_beta_tuning:
        best_beta = 0.0
        beta_table = pd.DataFrame([{"Beta": best_beta, "Selection": "default-no-tuning"}])
    else:
        print("Tuning beta_uncertainty on validation data...")
        best_beta, beta_table = tune_uncertainty_beta(
            data["X_train"], data["y_train"], data["X_val"], data["y_val"],
            n_layers=args.layers, encoding=args.encoding,
            batch_size=args.batch_size, learning_rate=args.lr,
            memory_capacity=min(args.memory_capacity, 500),
            replay_ratio=args.replay_ratio,
            alpha_error=args.alpha_error,
            gamma_malicious=args.gamma_malicious,
            seed=args.seed, verbose=False,
        )
    beta_table.to_csv(out / "beta_tuning.csv", index=False)
    print(f"Selected beta_uncertainty={best_beta}")

    model = QuantumCyberAnathema(
        n_qubits=data["X_train"].shape[1], n_layers=args.layers,
        encoding=args.encoding, seed=args.seed,
    )

    t0 = perf_counter()
    history, memory = fit_qca(
        model,
        data["X_train"], data["y_train"], data["X_val"], data["y_val"],
        cycles=args.cycles,
        epochs_per_cycle=args.epochs_per_cycle,
        warmup_epochs=args.warmup_epochs,
        warmup_patience=args.warmup_patience,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        memory_capacity=args.memory_capacity,
        replay_ratio=args.replay_ratio,
        alpha_error=args.alpha_error,
        beta_uncertainty=best_beta,
        gamma_malicious=args.gamma_malicious,
        checkpoint_dir=out / "checkpoints",
        restore_best=True,
        seed=args.seed,
    )
    train_seconds = perf_counter() - t0

    val_prob = predict_qca(model, data["X_val"], batch_size=max(args.batch_size, 32))
    threshold, val_threshold_score, threshold_table = select_threshold(
        data["y_val"], val_prob, metric=args.threshold_metric
    )
    threshold_table.to_csv(out / "threshold_search.csv", index=False)

    t1 = perf_counter()
    test_prob = predict_qca(model, data["X_test"], batch_size=max(args.batch_size, 32))
    infer_seconds = perf_counter() - t1
    test_pred = (test_prob >= threshold).astype(int)
    test_metrics = classification_metrics(
        data["y_test"], test_pred, test_prob,
        inference_seconds=infer_seconds, n_samples=len(data["y_test"]),
    )
    test_metrics.update({
        "Train Seconds": float(train_seconds),
        "Selected Validation Cycle": int(model.qca_best_cycle_),
        "Selected Validation Defense Score": float(model.qca_best_validation_score_),
        "Beta Uncertainty": float(best_beta),
        "Threshold": float(threshold),
        f"Validation Threshold {args.threshold_metric.upper()}": float(val_threshold_score),
        "Maximum Optimization Budget": int(args.warmup_epochs + args.cycles * args.epochs_per_cycle),
    })

    torch.save(model.state_dict(), out / "selected_qca.pt")
    history.to_csv(out / "evolution_history.csv", index=False)
    save_json(test_metrics, out / "test_metrics.json")
    pd.DataFrame([test_metrics]).to_csv(out / "matched_test_metrics.csv", index=False)
    save_json(memory.summary(), out / "memory_summary.json")
    selected_config = {
        "n_qubits": data["X_train"].shape[1],
        "n_layers": args.layers,
        "encoding": args.encoding,
        "warmup_epochs": args.warmup_epochs,
        "warmup_patience": args.warmup_patience,
        "cycles": args.cycles,
        "epochs_per_cycle": args.epochs_per_cycle,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "memory_capacity": args.memory_capacity,
        "replay_ratio": args.replay_ratio,
        "alpha_error": args.alpha_error,
        "beta_uncertainty": best_beta,
        "gamma_malicious": args.gamma_malicious,
        "threshold": threshold,
        "selected_cycle": int(model.qca_best_cycle_),
        "restored_best_before_test": bool(model.qca_restored_best_),
    }
    save_json(selected_config, out / "selected_qca_config.json")

    save_evolution_plot(history, out / "qca_evolution.png")
    save_classification_plots(data["y_test"], test_prob, model_name="Quantum Cyber Anathema", out_dir=out)

    print("\nFINAL MATCHED TEST METRICS")
    print(json.dumps(test_metrics, indent=2))
    print("\nMEMORY")
    print(json.dumps(memory.summary(), indent=2))
    print("\nEVOLUTION")
    print(history.to_string(index=False))


if __name__ == "__main__":
    main()
