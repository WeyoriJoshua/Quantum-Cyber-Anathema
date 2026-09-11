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
from evaluation.plots import save_classification_plots
from models.pennylane_vqc import PennyLaneVQC, fit_vqc, predict_vqc


def parse_args():
    p = argparse.ArgumentParser(description="Train the budget-matched standard PennyLane VQC control.")
    p.add_argument("--data", default=str(CONFIG.benchmark_splits_path))
    p.add_argument("--out", default=str(CONFIG.project_root / "results" / "vqc" / CONFIG.experiment_tag))
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--encoding", choices=["angle", "data_reuploading"], default="angle")
    p.add_argument("--epochs", type=int, default=CONFIG.vqc_epochs)
    p.add_argument("--batch-size", type=int, default=CONFIG.batch_size)
    p.add_argument("--lr", type=float, default=CONFIG.learning_rate)
    p.add_argument("--patience", type=int, default=CONFIG.vqc_patience)
    p.add_argument("--threshold-metric", choices=["mcc", "f1"], default="mcc")
    p.add_argument("--shots", type=int, default=None)
    p.add_argument("--seed", type=int, default=CONFIG.seed)
    return p.parse_args()


def main():
    args = parse_args()
    set_global_seed(args.seed)
    data = load_quantum_splits(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = PennyLaneVQC(
        n_qubits=data["X_train"].shape[1],
        n_layers=args.layers,
        encoding=args.encoding,
        shots=args.shots,
        seed=args.seed,
    )

    t0 = perf_counter()
    history = fit_vqc(
        model,
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"],
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        checkpoint_path=out / "best_vqc.pt",
    )
    train_seconds = perf_counter() - t0
    history_df = pd.DataFrame(history.to_dict())

    val_prob = predict_vqc(model, data["X_val"], batch_size=max(args.batch_size, 32))
    threshold, val_selection_score, threshold_table = select_threshold(
        data["y_val"], val_prob, metric=args.threshold_metric
    )
    val_pred = (val_prob >= threshold).astype(int)
    val_metrics = classification_metrics(data["y_val"], val_pred, val_prob)
    val_metrics.update({
        "Threshold": threshold,
        f"Threshold Selection {args.threshold_metric.upper()}": val_selection_score,
        "Epochs Executed": len(history.epoch),
        "Train Seconds": float(train_seconds),
    })

    t1 = perf_counter()
    test_prob = predict_vqc(model, data["X_test"], batch_size=max(args.batch_size, 32))
    infer_seconds = perf_counter() - t1
    test_pred = (test_prob >= threshold).astype(int)
    test_metrics = classification_metrics(
        data["y_test"], test_pred, test_prob,
        inference_seconds=infer_seconds, n_samples=len(data["y_test"]),
    )
    test_metrics.update({
        "Model": f"PennyLane VQC - {args.encoding}",
        "Threshold": threshold,
        "Train Seconds": float(train_seconds),
        "Epochs Executed": len(history.epoch),
    })

    torch.save(model.state_dict(), out / "final_vqc.pt")
    save_json(history.to_dict(), out / "training_history.json")
    history_df.to_csv(out / "training_history.csv", index=False)
    threshold_table.to_csv(out / "threshold_search.csv", index=False)
    save_json(val_metrics, out / "validation_metrics.json")
    save_json(test_metrics, out / "test_metrics.json")
    pd.DataFrame([test_metrics]).to_csv(out / "matched_test_metrics.csv", index=False)
    save_classification_plots(
        data["y_test"], test_prob,
        model_name=f"PennyLane VQC ({args.encoding})", out_dir=out,
    )

    print("\nVALIDATION")
    print(json.dumps(val_metrics, indent=2))
    print("\nFINAL MATCHED TEST")
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
