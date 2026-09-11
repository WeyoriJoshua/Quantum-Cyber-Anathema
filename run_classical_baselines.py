from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
import joblib

from project_config import CONFIG
from experiments.common import load_quantum_splits, save_metrics_table, set_global_seed
from evaluation.metrics import classification_metrics, predict_scores, select_threshold
from models.classical import build_classical_models, fit_classical_models


def parse_args():
    p = argparse.ArgumentParser(description="Train classical controls on the matched benchmark.")
    p.add_argument("--data", default=str(CONFIG.benchmark_splits_path))
    p.add_argument("--out", default=str(CONFIG.project_root / "results" / "classical" / CONFIG.experiment_tag))
    p.add_argument("--seed", type=int, default=CONFIG.seed)
    p.add_argument("--threshold-metric", choices=["mcc", "f1"], default="mcc")
    return p.parse_args()


def main():
    args = parse_args()
    set_global_seed(args.seed)
    data = load_quantum_splits(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    fitted = fit_classical_models(
        build_classical_models(args.seed), data["X_train"], data["y_train"]
    )
    validation_rows, test_rows = [], []

    for name, item in fitted.items():
        model = item["model"]
        val_score = predict_scores(model, data["X_val"])
        threshold, selection_score, threshold_table = select_threshold(
            data["y_val"], val_score, metric=args.threshold_metric
        )
        val_pred = (val_score >= threshold).astype(int)
        val_metrics = classification_metrics(data["y_val"], val_pred, val_score)
        validation_rows.append({
            "Model": name,
            "Threshold": threshold,
            f"Threshold Selection {args.threshold_metric.upper()}": selection_score,
            "Train Seconds": item["train_seconds"],
            **val_metrics,
        })
        threshold_table.to_csv(out / f"{name}_threshold_search.csv", index=False)

        t0 = perf_counter()
        test_score = predict_scores(model, data["X_test"])
        infer_seconds = perf_counter() - t0
        test_pred = (test_score >= threshold).astype(int)
        test_metrics = classification_metrics(
            data["y_test"], test_pred, test_score,
            inference_seconds=infer_seconds, n_samples=len(data["y_test"]),
        )
        test_rows.append({
            "Model": name,
            "Threshold": threshold,
            "Train Seconds": item["train_seconds"],
            **test_metrics,
        })
        joblib.dump(model, out / f"{name}.joblib")

    val_table = save_metrics_table(validation_rows, out / "matched_validation_metrics.csv")
    test_table = save_metrics_table(test_rows, out / "matched_test_metrics.csv")
    test_table.to_csv(out / "classical_baseline_metrics.csv", index=False)

    print("\nVALIDATION METRICS")
    print(val_table.to_string(index=False))
    print("\nFINAL MATCHED TEST METRICS")
    print(test_table.to_string(index=False))


if __name__ == "__main__":
    main()
