from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
import json
import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

from project_config import CONFIG
from experiments.common import (
    load_quantum_splits,
    save_json,
    set_global_seed,
    stratified_subsample_arrays,
)
from evaluation.metrics import classification_metrics, predict_scores, select_threshold
from models.qiskit_baselines import build_qiskit_vqc, build_qsvc


def parse_args():
    p = argparse.ArgumentParser(
        description="Train Qiskit VQC / fidelity quantum-kernel SVM controls."
    )
    p.add_argument("--data", default=str(CONFIG.benchmark_splits_path))
    p.add_argument(
        "--out",
        default=str(
            CONFIG.project_root / "results" / "qiskit" / CONFIG.experiment_tag
        ),
    )
    p.add_argument("--model", choices=["vqc", "qsvc", "both"], default="both")
    p.add_argument("--feature-map", choices=["z", "zz"], default="zz")
    p.add_argument("--maxiter", type=int, default=CONFIG.qiskit_vqc_maxiter)
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--max-val", type=int, default=None)
    p.add_argument("--max-test", type=int, default=None)
    p.add_argument("--threshold-metric", choices=["mcc", "f1"], default="mcc")
    p.add_argument("--seed", type=int, default=CONFIG.seed)
    return p.parse_args()


def main():
    args = parse_args()
    set_global_seed(args.seed)
    data = load_quantum_splits(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    X_train, y_train = stratified_subsample_arrays(
        data["X_train"], data["y_train"], args.max_train, seed=args.seed
    )
    X_val, y_val = stratified_subsample_arrays(
        data["X_val"], data["y_val"], args.max_val, seed=args.seed + 1
    )
    X_test, y_test = stratified_subsample_arrays(
        data["X_test"], data["y_test"], args.max_test, seed=args.seed + 2
    )

    model_names = ["vqc", "qsvc"] if args.model == "both" else [args.model]
    validation_rows, test_rows, all_metrics = [], [], {}

    for name in model_names:
        label = "VQC" if name == "vqc" else "QuantumKernelSVM"
        print(
            f"\nTraining Qiskit {label} with {args.feature_map.upper()} feature map..."
        )

        if name == "vqc":
            model = build_qiskit_vqc(
                X_train.shape[1],
                feature_map=args.feature_map,
                maxiter=args.maxiter,
                seed=args.seed,
            )
        else:
            base = build_qsvc(
                X_train.shape[1],
                feature_map=args.feature_map,
                seed=args.seed,
            )
            model = CalibratedClassifierCV(
                estimator=base,
                method="sigmoid",
                cv=StratifiedKFold(
                    n_splits=3,
                    shuffle=True,
                    random_state=args.seed,
                ),
                ensemble=False,
            )

        t0 = perf_counter()
        model.fit(X_train, y_train)
        train_seconds = perf_counter() - t0

        val_score = predict_scores(model, X_val)
        threshold, val_selection_score, threshold_table = select_threshold(
            y_val, val_score, metric=args.threshold_metric
        )
        val_pred = (val_score >= threshold).astype(int)
        val_metrics = classification_metrics(y_val, val_pred, val_score)

        validation_rows.append(
            {
                "Model": label,
                "Feature Map": args.feature_map.upper(),
                "Threshold": threshold,
                f"Threshold Selection {args.threshold_metric.upper()}": val_selection_score,
                "Train Seconds": float(train_seconds),
                **val_metrics,
            }
        )
        threshold_table.to_csv(
            out / f"{name}_threshold_search.csv", index=False
        )

        t1 = perf_counter()
        test_score = predict_scores(model, X_test)
        infer_seconds = perf_counter() - t1
        test_pred = (test_score >= threshold).astype(int)

        test_metrics = classification_metrics(
            y_test,
            test_pred,
            test_score,
            inference_seconds=infer_seconds,
            n_samples=len(y_test),
        )
        test_metrics.update(
            {
                "Model": label,
                "Feature Map": args.feature_map.upper(),
                "Threshold": threshold,
                "Train Seconds": float(train_seconds),
                "Training Samples": len(y_train),
                "Validation Samples": len(y_val),
                "Test Samples": len(y_test),
            }
        )
        test_rows.append(test_metrics)
        all_metrics[name] = test_metrics

        # VQC may serialize; the callable quantum-kernel SVM is intentionally
        # not required to serialize for the publication result package.
        if name == "vqc":
            try:
                model.to_dill(str(out / f"{name}.dill"))
            except Exception:
                try:
                    joblib.dump(model, out / f"{name}.joblib")
                except Exception:
                    pass

    pd.DataFrame(validation_rows).to_csv(
        out / "matched_validation_metrics.csv", index=False
    )
    pd.DataFrame(test_rows).to_csv(
        out / "matched_test_metrics.csv", index=False
    )
    save_json(all_metrics, out / "qiskit_metrics.json")
    print(json.dumps(all_metrics, indent=2))


if __name__ == "__main__":
    main()
