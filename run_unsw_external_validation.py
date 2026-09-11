from __future__ import annotations

import argparse
import itertools
import json
import math
import traceback
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy import stats

from project_config import CONFIG
from experiments.common import load_quantum_splits, save_json, set_global_seed
from evaluation.metrics import classification_metrics, predict_scores, select_threshold
from models.classical import build_classical_models, fit_classical_models
from models.pennylane_vqc import PennyLaneVQC, fit_vqc, predict_vqc
from models.weighted_vqc import fit_weighted_vqc
from qca.model import QuantumCyberAnathema
from qca.trainer import fit_qca, predict_qca
from qca.v2_trainer import fit_qca_v2, predict_qca_v2


MODELS = (
    "LogisticRegression",
    "RandomForest",
    "XGBoost",
    "StaticVQC",
    "WeightedVQC",
    "QCA_Current",
    "QCA_V2_Frozen",
)

SUMMARY_METRICS = (
    "Accuracy", "Balanced Accuracy", "Precision", "Recall", "Specificity",
    "F1", "ROC-AUC", "PR-AUC", "MCC", "Cohen Kappa", "Log Loss",
    "Brier Score", "FPR", "FNR", "Train Seconds", "Test Inference Seconds",
)

PAIRED_METRICS = (
    "F1", "MCC", "PR-AUC", "ROC-AUC", "Recall", "Precision",
    "Balanced Accuracy", "Specificity",
)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Five-seed UNSW-NB15 external validation of the frozen QCA-v2 mechanism. "
            "Training/validation are sampled only from the official training split; "
            "the external test benchmark comes only from the official testing split."
        )
    )
    p.add_argument("--data", default=str(CONFIG.unsw_benchmark_splits_path))
    p.add_argument(
        "--out",
        default=str(
            CONFIG.project_root / "results" / "external_validation" / CONFIG.unsw_external_tag
        ),
    )
    p.add_argument("--seeds", nargs="+", type=int, default=list(CONFIG.final_seeds))
    p.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    p.add_argument("--threshold-metric", choices=["mcc", "f1"], default="mcc")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _counts(y):
    values, counts = np.unique(np.asarray(y).astype(int), return_counts=True)
    return {str(int(k)): int(v) for k, v in zip(values, counts)}


def _load_frozen_qca_config() -> dict:
    path = (
        CONFIG.project_root / "results" / "qca" / CONFIG.experiment_tag / "selected_qca_config.json"
    )
    if not path.exists():
        raise FileNotFoundError(
            "CICIDS2017 frozen QCA configuration is required for external validation: " + str(path)
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_data(data: dict):
    expected = {
        "train": int(CONFIG.unsw_train_size),
        "validation": int(CONFIG.unsw_val_size),
        "test": int(CONFIG.unsw_test_size),
    }
    actual = {
        "train": len(data["y_train"]),
        "validation": len(data["y_val"]),
        "test": len(data["y_test"]),
    }
    if actual != expected:
        raise ValueError(f"Expected frozen UNSW benchmark {expected}, got {actual}.")
    if data["X_train"].shape[1] != int(CONFIG.n_qubits):
        raise ValueError(
            f"Expected {CONFIG.n_qubits} quantum dimensions, got {data['X_train'].shape[1]}."
        )
    for split in ("train", "val", "test"):
        y = np.asarray(data[f"y_{split}"]).astype(int)
        if set(np.unique(y)) != {0, 1}:
            raise ValueError(f"{split}: both binary classes are required.")


def _prediction_frame(seed, model_name, split, y_true, y_score, threshold):
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_score = np.asarray(y_score, dtype=float).reshape(-1)
    return pd.DataFrame(
        {
            "Seed": int(seed),
            "Model": model_name,
            "Split": split,
            "Sample Index": np.arange(len(y_true), dtype=int),
            "y_true": y_true,
            "y_score": y_score,
            "y_pred": (y_score >= float(threshold)).astype(int),
            "Threshold": float(threshold),
        }
    )


def _save_confusion(metrics: dict, path: Path):
    pd.DataFrame(
        [[int(metrics["TN"]), int(metrics["FP"])], [int(metrics["FN"]), int(metrics["TP"]) ]],
        index=["True 0", "True 1"],
        columns=["Pred 0", "Pred 1"],
    ).to_csv(path)


def _completed(model_dir: Path) -> bool:
    return (
        (model_dir / "metrics.json").exists()
        and (model_dir / "validation_predictions.csv").exists()
        and (model_dir / "test_predictions.csv").exists()
    )


def _load_metrics(model_dir: Path) -> dict:
    return json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))


def _evaluate_and_save(
    *, model, score_fn, seed, model_name, model_dir, data, threshold_metric,
    train_seconds, extra=None,
):
    model_dir.mkdir(parents=True, exist_ok=True)

    val_score = np.asarray(score_fn(model, data["X_val"]), dtype=float).reshape(-1)
    threshold, val_best, search = select_threshold(
        data["y_val"], val_score, metric=threshold_metric
    )

    # External test is first accessed here, after model fitting, cycle/early-stop
    # selection and threshold selection are complete.
    t0 = perf_counter()
    test_score = np.asarray(score_fn(model, data["X_test"]), dtype=float).reshape(-1)
    test_seconds = perf_counter() - t0
    test_pred = (test_score >= threshold).astype(int)

    metrics = classification_metrics(
        data["y_test"], test_pred, test_score,
        inference_seconds=test_seconds, n_samples=len(data["y_test"]),
    )
    row = {
        "Seed": int(seed),
        "Model": model_name,
        "Threshold": float(threshold),
        f"Validation Threshold {threshold_metric.upper()}": float(val_best),
        "Train Seconds": float(train_seconds),
        "Test Inference Seconds": float(test_seconds),
        **metrics,
    }
    if extra:
        row.update(extra)

    search.to_csv(model_dir / "validation_threshold_search.csv", index=False)
    _prediction_frame(seed, model_name, "validation", data["y_val"], val_score, threshold).to_csv(
        model_dir / "validation_predictions.csv", index=False
    )
    _prediction_frame(seed, model_name, "external_test", data["y_test"], test_score, threshold).to_csv(
        model_dir / "test_predictions.csv", index=False
    )
    _save_confusion(row, model_dir / "test_confusion_matrix.csv")
    save_json(row, model_dir / "metrics.json")
    return row


def _new_qca_model(seed: int, frozen: dict):
    return QuantumCyberAnathema(
        n_qubits=int(frozen.get("n_qubits", CONFIG.n_qubits)),
        n_layers=int(frozen.get("n_layers", 2)),
        encoding=str(frozen.get("encoding", "angle")),
        seed=int(seed),
    )


def _run_classical(seed, model_name, data, out, threshold_metric, resume):
    model_dir = out / f"seed_{seed}" / model_name
    if resume and _completed(model_dir):
        print(f"[resume] seed={seed} {model_name}")
        return _load_metrics(model_dir)

    models = build_classical_models(seed)
    if model_name not in models:
        raise RuntimeError(f"{model_name} is unavailable in this environment.")

    print(f"[train] seed={seed} {model_name}")
    set_global_seed(seed)
    fitted = fit_classical_model²È="24µ±¥ÀÀˆè}•á…Ñ}Í¥¹}™±¥Á}ÁÙ…±Õ”¡¤°(€€€€€€€€€€€€‰]¥±½á½¸ÀˆèÝ¥±½á½¹}À°(€€€€€€€ô¤(€€€É•ÑÕÉ¸Á¹…Ñ…É…µ”¡É½ÝÌ¤(()‘•˜}…É•…Ñ•}ÁÉ•‘¥Ñ¥½¹Ì¡½ÕÐèA…Ñ ¤è(€€€Ù…°°Ñ•ÍÐ€ômt°mt(€€€™½ÈÀ¥¸Í½ÉÑ•¡½ÕÐ¹±½ˆ ‰Í••‘|¨¼¨½Ù…±¥‘…Ñ¥½¹}ÁÉ•‘¥Ñ¥½¹Ì¹ÍØˆ¤¤è(€€€€€€€Ù…°¹…ÁÁ•¹¡Á¹É•…‘}ÍØ¡À¤¤(€€€™½ÈÀ¥¸Í½ÉÑ•¡½ÕÐ¹±½ˆ ‰Í••‘|¨¼¨½Ñ•ÍÑ}ÁÉ•‘¥Ñ¥½¹Ì¹ÍØˆ¤¤è(€€€€€€€Ñ•ÍÐ¹…ÁÁ•¹¡Á¹É•…‘}ÍØ¡À¤¤(€€€¥˜Ù…°è(€€€€€€€Á¹½¹…Ð¡Ù…°°¥¹½É•}¥¹‘•àõQÉÕ”¤¹Ñ½}ÍØ¡½ÕÐ€¼€‰…±±}Ù…±¥‘…Ñ¥½¹}ÁÉ•‘¥Ñ¥½¹Ì¹ÍØˆ°¥¹‘•àõ…±Í”¤(€€€¥˜Ñ•ÍÐè(€€€€€€€Á¹½¹…Ð¡Ñ•ÍÐ°¥¹½É•}¥¹‘•àõQÉÕ”¤¹Ñ½}ÍØ¡½ÕÐ€¼€‰…±±}•áÑ•É¹…±}Ñ•ÍÑ}ÁÉ•‘¥Ñ¥½¹Ì¹ÍØˆ°¥¹‘•àõ…±Í”¤(()‘•˜}µ…¹¥™•ÍÐ¡½ÕÐèA…Ñ °…ÉÌ°‘…Ñ„è‘¥Ð°™É½é•¸è‘¥Ð¤è(€€€ÁÉ•ÁÉ½•ÍÍ}É•Á½ÉÐ€ô=9%¹Õ¹ÍÝ}ÁÉ•ÁÉ½•ÍÍ•‘}‘¥È€¼€‰ÁÉ•ÁÉ½•ÍÍ¥¹}É•Á½ÉÐ¹©Í½¸ˆ(€€€ÁÉ•ÁÉ½•ÍÍ¥¹œ€ô9½¹”(€€€¥˜ÁÉ•ÁÉ½•ÍÍ}É•Á½ÉÐ¹•á¥ÍÑÌ ¤è(€€€€€€€ÁÉ•ÁÉ½•ÍÍ¥¹œ€ô©Í½¸¹±½…‘Ì¡ÁÉ•ÁÉ½•ÍÍ}É•Á½ÉÐ¹É•…‘}Ñ•áÐ¡•¹½‘¥¹œô‰ÕÑ˜´àˆ¤¤(€€€µ…¹¥™•ÍÐ€ôì(€€€€€€€€‰‘…Ñ…Í•Ðˆè€‰U9M\µ9ÄÔˆ°(€€€€€€€€‰Ù…±¥‘…Ñ¥½¹}É½±”ˆè€‰•áÑ•É¹…°½¹™¥Éµ…Ñ¥½¸…™Ñ•È%%LÈÀÄÜEµØÈ‘•Ù•±½Áµ•¹Ðˆ°(€€€€€€€€‰‰•¹¡µ…É­}Á…Ñ ˆèÍÑÈ¡A…Ñ ¡…ÉÌ¹‘…Ñ„¤¹É•Í½±Ù” ¤¤°(€€€€€€€€‰Í••‘Ìˆèm¥¹Ð¡Ì¤™½ÈÌ¥¸…ÉÌ¹Í••‘Ít°(€€€€€€€€‰µ½‘•±Ìˆè±¥ÍÐ¡…ÉÌ¹µ½‘•±Ì¤°(€€€€€€€€‰‰•¹¡µ…É­}Í¥é•Ìˆèì(€€€€€€€€€€€€‰ÑÉ…¥¸ˆè±•¸¡‘…Ñ…l‰å}ÑÉ…¥¸‰t¤°(€€€€€€€€€€€€‰Ù…±¥‘…Ñ¥½¸ˆè±•¸¡‘…Ñ…l‰å}Ù…°‰t¤°(€€€€€€€€€€€€‰•áÑ•É¹…±}Ñ•ÍÐˆè±•¸¡‘…Ñ…l‰å}Ñ•ÍÐ‰t¤°(€€€€€€€ô°(€€€€€€€€‰±…ÍÍ}½Õ¹ÑÌˆèì(€€€€€€€€€€€€‰ÑÉ…¥¸ˆè}½Õ¹ÑÌ¡‘…Ñ…l‰å}ÑÉ…¥¸‰t¤°(€€€€€€€€€€€€‰Ù…±¥‘…Ñ¥½¸ˆè}½Õ¹ÑÌ¡‘…Ñ…l‰å}Ù…°‰t¤°(€€€€€€€€€€€€‰•áÑ•É¹…±}Ñ•ÍÐˆè}½Õ¹ÑÌ¡‘…Ñ…l‰å}Ñ•ÍÐ‰t¤°(€€€€€€€ô°(€€€€€€€€‰¹}ÅÕ‰¥ÑÌˆè¥¹Ð¡‘…Ñ…l‰a}ÑÉ…¥¸‰t¹Í¡…Á•lÅt¤°(€€€€€€€€‰Ñ¡É•Í¡½±‘}Í•±•Ñ¥½¸ˆè˜‰U9M\Ù…±¥‘…Ñ¥½¸½¹±ä°µ•ÑÉ¥Œõí…ÉÌ¹Ñ¡É•Í¡½±‘}µ•ÑÉ¥Œ¹ÕÁÁ•È ¥ôˆ°(€€€€€€€€‰•áÑ•É¹…±}Ñ•ÍÑ}Á½±¥äˆè€ (€€€€€€€€€€€€‰=™™¥¥…°U9M\Ñ•ÍÑ¥¹œÍÁ±¥ÐÍÕÁÁ±¥•ÌÑ¡”•áÑ•É¹…°Ñ•ÍÐ‰•¹¡µ…É¬ì¥Ð¥Ì¹½ÐÕÍ•€ˆ(€€€€€€€€€€€€‰™½ÈÁÉ•ÁÉ½•ÍÍ¥¹œ™¥Ð°µ½‘•°™¥ÑÑ¥¹œ°•…É±äÍÑ½ÁÁ¥¹œ½å±”Í•±•Ñ¥½¸°½ÈÑ¡É•Í¡½±Í•±•Ñ¥½¸¸ˆ(€€€€€€€€¤°(€€€€€€€€‰™É½é•¹}Å…}‰…Í•}½¹™¥}™É½µ}¥¥‘Ìˆè™É½é•¸°(€€€€€€€€‰™É½é•¹}Å…}ØÈˆèì(€€€€€€€€€€€€‰Í•±•Ñ•‘}½¸ˆè€‰%%LÈÀÄÜ™¥Ù”µÍ••Ù…±¥‘…Ñ¥½¸µ½¹±ä‘•Ù•±½Áµ•¹Ðˆ°(€€€€€€€€€€€€‰Ù…É¥…¹Ðˆè=9%¹Õ¹ÍÝ}Å…}ØÉ}Ù…É¥…¹Ð°(€€€€€€€€€€€€‰µ…É¥¹}±…µ‰‘„ˆè=9%¹Õ¹ÍÝ}Å…}ØÉ}µ…É¥¹}±…µ‰‘„°(€€€€€€€€€€€€‰ÁÉ½‰…‰¥±¥Ñå}µ…É¥¸ˆè=9%¹Õ¹ÍÝ}Å…}ØÉ}ÁÉ½‰…‰¥±¥Ñå}µ…É¥¸°(€€€€€€€€€€€€‰É•Á±…å}ÁÉ¥½É¥Ñå}µ½‘”ˆè€‰‰½Õ¹‘…Éäˆ°(€€€€€€€€€€€€‰‰½Õ¹‘…Éå}Ý•¥¡Ðˆè=9%¹Õ¹ÍÝ}Å…}ØÉ}‰½Õ¹‘…Éå}Ý•¥¡Ð°(€€€€€€€€€€€€‰‰½Õ¹‘…Éå}Ñ¡É•Í¡½±ˆè=9%¹Õ¹ÍÝ}Å…}ØÉ}‰½Õ¹‘…Éå}Ñ¡É•Í¡½±°(€€€€€€€ô°(€€€€€€€€‰ÁÉ•ÁÉ½•ÍÍ¥¹}É•Á½ÉÐˆèÁÉ•ÁÉ½•ÍÍ¥¹œ°(€€€€€€€€‰¥¹Ñ•ÉÁÉ•Ñ…Ñ¥½¹}¹½Ñ”ˆè€ (€€€€€€€€€€€€‰5•…¸½M¼äÔ”$…É½ÍÌÍ••‘ÌÅÕ…¹Ñ¥™äÍÑ½¡…ÍÑ¥ŒÑÉ…¥¹¥¹œÙ…É¥…‰¥±¥Ñä½¸½¹”™¥á••áÑ•É¹…°‰•¹¡µ…É¬°€ˆ(€€€€€€€€€€€€‰¹½ÐÕ¹•ÉÑ…¥¹Ñä½Ù•È¹•Ü‘…Ñ…Í•ÐÉ•Í…µÁ±•Ì¸ˆ(€€€€€€€€¤°(€€€ô(€€€Í…Ù•}©Í½¸¡µ…¹¥™•ÍÐ°½ÕÐ€¼€‰•áÑ•É¹…±}Ù…±¥‘…Ñ¥½¹}µ…¹¥™•ÍÐ¹©Í½¸ˆ¤(()‘•˜µ…¥¸ ¤è(€€€…ÉÌ€ôÁ…ÉÍ•}…ÉÌ ¤(€€€‘…Ñ„€ô±½…‘}ÅÕ…¹ÑÕµ}ÍÁ±¥ÑÌ¡…ÉÌ¹‘…Ñ„¤(€€€}Ù…±¥‘…Ñ•}‘…Ñ„¡‘…Ñ„¤(€€€™É½é•¸€ô}±½…‘}™É½é•¹}Å…}½¹™¥œ ¤(€€€½ÕÐ€ôA…Ñ ¡…ÉÌ¹½ÕÐ¤(€€€½ÕÐ¹µ­‘¥È¡Á…É•¹ÑÌõQÉÕ”°•á¥ÍÑ}½¬õQÉÕ”¤(€€€}µ…¹¥™•ÍÐ¡½ÕÐ°…ÉÌ°‘…Ñ„°™É½é•¸¤((€€€ÁÉ¥¹Ð ‰q¸ôôôU9M\µ9ÄÔaQI90Y1%Q%=8€ôôôˆ¤(€€€ÁÉ¥¹Ð ‰	•¹¡µ…É¬èˆ°‘…Ñ…l‰a}ÑÉ…¥¸‰t¹Í¡…Á”°‘…Ñ…l‰a}Ù…°‰t¹Í¡…Á”°‘…Ñ…l‰a}Ñ•ÍÐ‰t¹Í¡…Á”¤(€€€ÁÉ¥¹Ð ‰±…ÍÌ½Õ¹ÑÌÑÉ…¥¸èˆ°}½Õ¹ÑÌ¡‘…Ñ…l‰å}ÑÉ…¥¸‰t¤¤(€€€ÁÉ¥¹Ð ‰±…ÍÌ½Õ¹ÑÌÙ…°€€èˆ°}½Õ¹ÑÌ¡‘…Ñ…l‰å}Ù…°‰t¤¤(€€€ÁÉ¥¹Ð ‰±…ÍÌ½Õ¹ÑÌÑ•ÍÐ€èˆ°}½Õ¹ÑÌ¡‘…Ñ…l‰å}Ñ•ÍÐ‰t¤¤(€€€ÁÉ¥¹Ð ‰M••‘Ìèˆ°…ÉÌ¹Í••‘Ì¤(€€€ÁÉ¥¹Ð ‰5½‘•±Ìèˆ°…ÉÌ¹µ½‘•±Ì¤(€€€ÁÉ¥¹Ð ‰EµØÈ™É½é•¸Ù…É¥…¹Ðèˆ°=9%¹Õ¹ÍÝ}Å…}ØÉ}Ù…É¥…¹Ð¤(€€€ÁÉ¥¹Ð ‰EµØÈµ…É¥¸±…µ‰‘„èˆ°=9%¹Õ¹ÍÝ}Å…}ØÉ}µ…É¥¹}±…µ‰‘„¤(€€€ÁÉ¥¹Ð ‰EµØÈÉ•Á±…äµ½‘”è‰½Õ¹‘…Éäˆ¤(€€€ÁÉ¥¹Ð ‰áÑ•É¹…°Ñ•ÍÐÕÍ•™½ÈÑÕ¹¥¹œè…±Í”ˆ¤(€€€ÁÉ¥¹Ð ‰=ÕÑÁÕÐèˆ°½ÕÐ¤((€€€¥˜…ÉÌ¹‘Éå}ÉÕ¸è(€€€€€€€ÁÉ¥¹Ð ‰q¹ÉäÉÕ¸½µÁ±•Ñ”¸9¼µ½‘•±ÌÝ•É”ÑÉ…¥¹•¸ˆ¤(€€€€€€€É•ÑÕÉ¸((€€€™…¥±ÕÉ•Ì€ômt((€€€™½ÈÍ••¥¸…ÉÌ¹Í••‘Ìè(€€€€€€€ÁÉ¥¹Ð¡˜‰q¹q¸ôôôôôôôôôôôôôôôôMíÍ••‘ô€ôôôôôôôôôôôôôôôôˆ¤(€€€€€€€™½Èµ½‘•±}¹…µ”¥¸…ÉÌ¹µ½‘•±Ìè(€€€€€€€€€€€ÑÉäè(€€€€€€€€€€€€€€€¥˜µ½‘•±}¹…µ”¥¸ì‰1½¥ÍÑ¥I•É•ÍÍ¥½¸ˆ°€‰I…¹‘½µ½É•ÍÐˆ°€‰a	½½ÍÐ‰ôè(€€€€€€€€€€€€€€€€€€€}ÉÕ¹}±…ÍÍ¥…°¡Í••°µ½‘•±}¹…µ”°‘…Ñ„°½ÕÐ°…ÉÌ¹Ñ¡É•Í¡½±‘}µ•ÑÉ¥Œ°…ÉÌ¹É•ÍÕµ”¤(€€€€€€€€€€€€€€€•±¥˜µ½‘•±}¹…µ”€ôô€‰MÑ…Ñ¥YEˆè(€€€€€€€€€€€€€€€€€€€}ÉÕ¹}ÍÑ…Ñ¥}ÙÅŒ¡Í••°™É½é•¸°‘…Ñ„°½ÕÐ°…ÉÌ¹Ñ¡É•Í¡½±‘}µ•ÑÉ¥Œ°…ÉÌ¹É•ÍÕµ”¤(€€€€€€€€€€€€€€€•±¥˜µ½‘•±}¹…µ”€ôô€‰]•¥¡Ñ•‘YEˆè(€€€€€€€€€€€€€€€€€€€}ÉÕ¹}Ý•¥¡Ñ•‘}ÙÅŒ¡Í••°™É½é•¸°‘…Ñ„°½ÕÐ°…ÉÌ¹Ñ¡É•Í¡½±‘}µ•ÑÉ¥Œ°…ÉÌ¹É•ÍÕµ”¤(€€€€€€€€€€€€€€€•±¥˜µ½‘•±}¹…µ”€ôô€‰E}ÕÉÉ•¹Ðˆè(€€€€€€€€€€€€€€€€€€€}ÉÕ¹}Å…}ÕÉÉ•¹Ð¡Í••°™É½é•¸°‘…Ñ„°½ÕÐ°…ÉÌ¹Ñ¡É•Í¡½±‘}µ•ÑÉ¥Œ°…ÉÌ¹É•ÍÕµ”¤(€€€€€€€€€€€€€€€•±¥˜µ½‘•±}¹…µ”€ôô€‰E}XÉ}É½é•¸ˆè(€€€€€€€€€€€€€€€€€€€}ÉÕ¹}Å…}ØÈ¡Í••°™É½é•¸°‘…Ñ„°½ÕÐ°…ÉÌ¹Ñ¡É•Í¡½±‘}µ•ÑÉ¥Œ°…ÉÌ¹É•ÍÕµ”¤(€€€€€€€€€€€•á•ÁÐá•ÁÑ¥½¸…Ì•áŒè(€€€€€€€€€€€€€€€™…¥±ÕÉ•Ì¹…ÁÁ•¹¡ì(€€€€€€€€€€€€€€€€€€€€‰M••ˆè¥¹Ð¡Í••¤°€‰5½‘•°ˆèµ½‘•±}¹…µ”°(€€€€€€€€€€€€€€€€€€€€‰ÉÉ½ÈˆèÉ•ÁÈ¡•áŒ¤°€‰QÉ…•‰…¬ˆèÑÉ…•‰…¬¹™½Éµ…Ñ}•áŒ ¤°(€€€€€€€€€€€€€€€ô¤(€€€€€€€€€€€€€€€¥˜…ÉÌ¹™…¥±}™…ÍÐè(€€€€€€€€€€€€€€€€€€€É…¥Í”((€€€€€€€€ŒÁ•ÉÍ¥ÍÐÁ…ÉÑ¥…°…É•…Ñ•Ì…™Ñ•È•Ù•ÉäÍ••(€€€€€€€É½ÝÌ€ômt(€€€€€€€™½ÈÀ¥¸Í½ÉÑ•¡½ÕÐ¹±½ˆ ‰Í••‘|¨¼¨½µ•ÑÉ¥Ì¹©Í½¸ˆ¤¤è(€€€€€€€€€€€ÑÉäè(€€€€€€€€€€€€€€€É½ÝÌ¹…ÁÁ•¹¡©Í½¸¹±½…‘Ì¡À¹É•…‘}Ñ•áÐ¡•¹½‘¥¹œô‰ÕÑ˜´àˆ¤¤¤(€€€€€€€€€€€•á•ÁÐá•ÁÑ¥½¸è(€€€€€€€€€€€€€€€Á…ÍÌ(€€€€€€€¥˜É½ÝÌè(€€€€€€€€€€€Á¹…Ñ…É…µ”¡É½ÝÌ¤¹‘É½Á}‘ÕÁ±¥…Ñ•Ì¡l‰M••ˆ°€‰5½‘•°‰t°­••Àô‰±…ÍÐˆ¤¹Í½ÉÑ}Ù…±Õ•Ì (€€€€€€€€€€€€€€€l‰M••ˆ°€‰5½‘•°‰t(€€€€€€€€€€€€¤¹Ñ½}ÍØ¡½ÕÐ€¼€‰…±±}Í••‘}•áÑ•É¹…±}Ñ•ÍÑ}µ•ÑÉ¥Ì¹ÍØˆ°¥¹‘•àõ…±Í”¤(€€€€€€€¥˜™…¥±ÕÉ•Ìè(€€€€€€€€€€€Á¹…Ñ…É…µ”¡™…¥±ÕÉ•Ì¤¹Ñ½}ÍØ¡½ÕÐ€¼€‰™…¥±ÕÉ•Ì¹ÍØˆ°¥¹‘•àõ…±Í”¤(€€€€€€€}…É•…Ñ•}ÁÉ•‘¥Ñ¥½¹Ì¡½ÕÐ¤((€€€É½ÝÌ€ômt(€€€™½ÈÀ¥¸Í½ÉÑ•¡½ÕÐ¹±½ˆ ‰Í••‘|¨¼¨½µ•ÑÉ¥Ì¹©Í½¸ˆ¤¤è(€€€€€€€É½ÝÌ¹…ÁÁ•¹¡©Í½¸¹±½…‘Ì¡À¹É•…‘}Ñ•áÐ¡•¹½‘¥¹œô‰ÕÑ˜´àˆ¤¤¤(€€€µ•ÑÉ¥Ì€ôÁ¹…Ñ…É…µ”¡É½ÝÌ¤¹‘É½Á}‘ÕÁ±¥…Ñ•Ì¡l‰M••ˆ°€‰5½‘•°‰t°­••Àô‰±…ÍÐˆ¤¹Í½ÉÑ}Ù…±Õ•Ì (€€€€€€€l‰M••ˆ°€‰5½‘•°‰t(€€€€¤(€€€µ•ÑÉ¥Ì¹Ñ½}ÍØ¡½ÕÐ€¼€‰…±±}Í••‘}•áÑ•É¹…±}Ñ•ÍÑ}µ•ÑÉ¥Ì¹ÍØˆ°¥¹‘•àõ…±Í”¤((€€€ÍÕµµ…Éä€ôÍÕµµ…É¥é•}µ•ÑÉ¥Ì¡µ•ÑÉ¥Ì¤(€€€ÍÕµµ…Éä¹Ñ½}ÍØ¡½ÕÐ€¼€‰ÍÕµµ…Éå}µ•…¹}Í‘|äÕ$¹ÍØˆ°¥¹‘•àõ…±Í”¤((€€€½µÁ…É¥Í½¹Ì€ômt(€€€™½ÈÉ•™•É•¹”¥¸€ ‰MÑ…Ñ¥YEˆ°€‰]•¥¡Ñ•‘YEˆ°€‰E}ÕÉÉ•¹Ðˆ¤è(€€€€€€€Ñ…‰±”€ôÁ…¥É•‘}½µÁ…É¥Í½¸¡µ•ÑÉ¥Ì°€‰E}XÉ}É½é•¸ˆ°É•™•É•¹”¤(€€€€€€€¥˜¹½ÐÑ…‰±”¹•µÁÑäè(€€€€€€€€€€€½µÁ…É¥Í½¹Ì¹…ÁÁ•¹¡Ñ…‰±”¤(€€€¥˜½µÁ…É¥Í½¹Ìè(€€€€€€€Á¹½¹…Ð¡½µÁ…É¥Í½¹Ì°¥¹½É•}¥¹‘•àõQÉÕ”¤¹Ñ½}ÍØ (€€€€€€€€€€€½ÕÐ€¼€‰E}XÉ}•áÑ•É¹…±}Á…¥É•‘}½µÁ…É¥Í½¹Ì¹ÍØˆ°¥¹‘•àõ…±Í”(€€€€€€€€¤((€€€€Œ•ÍÉ¥ÁÑ¥Ù”É…¹­¥¹œ½¹±äì•áÑ•É¹…°Ñ•ÍÐµÕÍÐ¹½Ð‰”ÕÍ•Ñ¼É•ÑÕ¹”½ÈÉ•Í•±•ÐEµØÈ¸(€€€É…¹¬€ô€ (€€€€€€€µ•ÑÉ¥Ì¹É½ÕÁ‰ä ‰5½‘•°ˆ°…Í}¥¹‘•àõ…±Í”¤(€€€€€€€€¹…œ¡ì‰AHµUˆè€‰µ•…¸ˆ°€‰5ˆè€‰µ•…¸ˆ°€‰Äˆè€‰µ•…¸ˆ°€‰I=µUˆè€‰µ•…¸ˆ°€‰I•…±°ˆè€‰µ•…¸‰ô¤(€€€€€€€€¹Í½ÉÑ}Ù…±Õ•Ì¡l‰AHµUˆ°€‰5‰t°…Í•¹‘¥¹œõ…±Í”¤(€€€€¤(€€€É…¹¬¹¥¹Í•ÉÐ À°€‰•ÍÉ¥ÁÑ¥Ù”I…¹¬ˆ°¹À¹…É…¹” Ä°±•¸¡É…¹¬¤€¬€Ä¤¤(€€€É…¹¬¹Ñ½}ÍØ¡½ÕÐ€¼€‰•áÑ•É¹…±}Ñ•ÍÑ}‘•ÍÉ¥ÁÑ¥Ù•}É…¹­¥¹œ¹ÍØˆ°¥¹‘•àõ…±Í”¤((€€€}…É•…Ñ•}ÁÉ•‘¥Ñ¥½¹Ì¡½ÕÐ¤((€€€ÁÉ¥¹Ð ‰q¸ôôôaQI90Y1%Q%=8=5A1Q€¼UII9Q1dY%1	1€ôôôˆ¤(€€€ÁÉ¥¹Ð ‰½µÁ±•Ñ•µ½‘•°µÍ••É½ÝÌèˆ°±•¸¡µ•ÑÉ¥Ì¤¤(€€€ÁÉ¥¹Ð ‰M…Ù•èˆ°½ÕÐ€¼€‰…±±}Í••‘}•áÑ•É¹…±}Ñ•ÍÑ}µ•ÑÉ¥Ì¹ÍØˆ¤(€€€ÁÉ¥¹Ð ‰M…Ù•èˆ°½ÕÐ€¼€‰ÍÕµµ…Éå}µ•…¹}Í‘|äÕ$¹ÍØˆ¤(€€€ÁÉ¥¹Ð ‰M…Ù•èˆ°½ÕÐ€¼€‰E}XÉ}•áÑ•É¹…±}Á…¥É•‘}½µÁ…É¥Í½¹Ì¹ÍØˆ¤(€€€ÁÉ¥¹Ð ‰M…Ù•èˆ°½ÕÐ€¼€‰…±±}•áÑ•É¹…±}Ñ•ÍÑ}ÁÉ•‘¥Ñ¥½¹Ì¹ÍØˆ¤(€€€¥˜™…¥±ÕÉ•Ìè(€€€€€€€ÁÉ¥¹Ð ‰M½µ”ÉÕ¹Ì™…¥±•ì¥¹ÍÁ•Ðèˆ°½ÕÐ€¼€‰™…¥±ÕÉ•Ì¹ÍØˆ¤(()¥˜}}¹…µ•}|€ôô€‰}}µ…¥¹}|ˆè(€€€µ…¥¸ ¤(