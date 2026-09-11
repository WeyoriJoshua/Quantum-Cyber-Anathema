from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

from project_config import CONFIG
from experiments.common import load_quantum_splits, set_global_seed
from evaluation.metrics import classification_metrics, predict_scores, select_threshold
from models.pennylane_vqc import PennyLaneVQC, fit_vqc, predict_vqc
from models.qiskit_baselines import build_qiskit_vqc, build_qsvc


def parse_args():
    p = argparse.ArgumentParser(
        description="Validation-driven quantum encoding / feature-map ablation."
    )
    p.add_argument("--data", default=str(CONFIG.benchmark_splits_path))
    p.add_argument(
        "--out",
        default=str(
            CONFIG.project_root
            / "results"
            / "feature_map_ablation"
            / CONFIG.experiment_tag
        ),
    )
    p.add_argument("--pl-epochs", type=int, default=CONFIG.vqc_epochs)
    p.add_argument("--pl-patience", type=int, default=CONFIG.vqc_patience)
    p.add_argument("--qiskit-maxiter", type=int, default=CONFIG.qiskit_vqc_maxiter)
    p.add_argument("--seed", type=int, default=CONFIG.seed)
    return p.parse_args()


def calibrated_qkernel_svm(n_features: int, fmap: str, seed: int):
    base = build_qsvc(n_features, feature_map=fmap, seed=seed)
    return CalibratedClassifierCV(
        estimator=base,
        method="sigmoid",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=seed),
        ensemble=False,
    )


def main():
    args = parse_args()
    data = load_quantum_splits(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    Xtr, ytr = data["X_train"], data["y_train"]
    Xv, yv = data["X_val"], data["y_val"]
    Xt, yt = data["X_test"], data["y_test"]

    pl_val_rows, pl_models = [], {}
    for encoding in ["angle", "data_reuploading"]:
        set_global_seed(args.seed)
        model = PennyLaneVQC(Xtr.shape[1], n_layers=2, encoding=encoding, seed=args.seed)
        history = fit_vqc(
            model, Xtr, ytr, Xv, yv,
            epochs=args.pl_epochs,
            batch_size=CONFIG.batch_size,
            learning_rate=CONFIG.learning_rate,
            patience=args.pl_patience,
            verbose=False,
        )
        pv = predict_vqc(model, Xv, batch_size=32)
        threshold, _, _ = select_threshold(yv, pv, metric="mcc")
        m = classification_metrics(yv, (pv >= threshold).astype(int), pv)
        pl_val_rows.append({
            "Framework": "PennyLane", "Model": "VQC", "Encoding": encoding,
            "Threshold": threshold, "Epochs Executed": len(history.epoch), **m,
        })
        pl_models[encoding] = (model, threshold)

    pl_val = pd.DataFrame(pl_val_rows).sort_values("PR-AUC", ascending=False)
    best_encoding = str(pl_val.iloc[0]["Encoding"])
    best_model, best_threshold = pl_models[best_encoding]
    pt = predict_vqc(best_model, Xt, batch_size=32)
    pl_test = pd.DataFrame([{
        "Framework": "PennyLane", "Model": "VQC", "Encoding": best_encoding,
        "Threshold": best_threshold,
        **classification_metrics(yt, (pt >= best_threshold).astype(int), pt),
    }])

    q_val_rows, q_models = [], {}
    for algorithm in ["VQC", "QSVC"]:
        for fmap in ["z", "zz"]:
            set_global_seed(args.seed)
            if algorithm == "VQC":
                model = build_qiskit_vqc(
                    Xtr.shape[1], feature_map=fmap,
                    maxiter=args.qiskit_maxiter, seed=args.seed,
                )
            else:
                model = calibrated_qkernel_svm(Xtr.shape[1], fmap, args.seed)

            t0 = perf_counter()
            model.fit(Xtr, ytr)
            train_seconds = perf_counter() - t0
            pv = predict_scores(model, Xv)
            threshold, _, _ = select_threshold(yv, pv, metric="mcc")
            m = classification_metrics(yv, (pv >= threshold).astype(int), pv)
            q_val_rows.append({
                "Framework": "Qiskit", "Model": algorithm,
                "Encoding": fmap.upper(), "Threshold": threshold,
                "Train Seconds": train_seconds, **m,
            })
            q_models[(algorithm, fmap)] = (model, threshold)

    q_val = pd.DataFrame(q_val_rows)
    q_test_rows = []
    for algorithm in ["VQC", "QSVC"]:
        selected = q_val[q_val["Model"] == algorithm].sort_values("PR-AUC", ascending=False).iloc[0]
        fmap = str(selected["Encoding"]).lower()
        model, threshold = q_models[(algorithm, fmap)]
        pt = predict_scores(model, Xt)
        q_test_rows.append({
            "Framework": "Qiskit", "Model": algorithm,
            "Encoding": fmap.upper(), "Threshold": threshold,
            **classification_metrics(yt, (pt >= threshold).astype(int), pt),
        })

    q_test = pd.DataFrame(q_test_rows)
    pl_val.to_csv(out / "pennylane_validation_ablation.csv", index=False)
    pl_test.to_csv(out / "pennylane_selected_test.csv", index=False)
    q_val.to_csv(out / "qiskit_validation_ablation.csv", index=False)
    q_test.to_csv(out / "qiskit_selected_test.csv", index=False)

    print("\nPENNYLANE VALIDATION\n", pl_val.to_string(index=False))
    print("\nPENNYLANE SELECTED TEST\n", pl_test.to_string(index=False))
    print("\nQISKIT VALIDATION\n", q_val.to_string(index=False))
    print("\nQISKIT SELECTED TEST\n", q_test.to_string(index=False))


if __name__ == "__main__":
    main()
