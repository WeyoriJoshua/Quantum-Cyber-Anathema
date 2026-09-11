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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

from project_config import CONFIG
from experiments.common import save_json, set_global_seed
from evaluation.metrics import classification_metrics, select_threshold
from models.pennylane_vqc import PennyLaneVQC, fit_vqc, predict_vqc
from models.weighted_vqc import fit_weighted_vqc
from qca.model import QuantumCyberAnathema
from qca.trainer import fit_qca, predict_qca
from qca.v2_trainer import fit_qca_v2, predict_qca_v2


VARIANTS = (
    "StaticVQC",
    "WeightedVQC",
    "QCA_Current",
    "QCA_Margin",
    "QCA_HardReplay",
    "QCA_MarginHardReplay",
)

SUMMARY_METRICS = (
    "Accuracy",
    "Balanced Accuracy",
    "Precision",
    "Recall",
    "Specificity",
    "F1",
    "ROC-AUC",
    "PR-AUC",
    "MCC",
    "Cohen Kappa",
    "Log Loss",
    "Brier Score",
    "FPR",
    "FNR",
    "Score Mean Gap",
    "KS Statistic",
    "Train Seconds",
    "Outer Validation Inference Seconds",
)

PAIRED_METRICS = (
    "F1",
    "MCC",
    "PR-AUC",
    "ROC-AUC",
    "Recall",
    "Precision",
    "Balanced Accuracy",
    "Score Mean Gap",
    "KS Statistic",
)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "QCA-v2 validation-only development. The frozen 300-sample test "
            "partition is never loaded or evaluated by this script."
        )
    )
    p.add_argument(
        "--data",
        default=str(CONFIG.benchmark_splits_path),
        help="Frozen 5K benchmark NPZ. Only X_train/y_train/X_val/y_val are accessed.",
    )
    p.add_argument(
        "--out",
        default=str(
            CONFIG.project_root
            / "results"
            / "qca_v2_development"
            / CONFIG.qca_v2_development_tag
        ),
    )
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(CONFIG.final_seeds),
        help="Model/training seeds. The development split is fixed across seeds.",
    )
    p.add_argument(
        "--variants",
        nargs="+",
        choices=list(VARIANTS),
        default=list(VARIANTS),
    )
    p.add_argument(
        "--inner-val-fraction",
        type=float,
        default=CONFIG.qca_v2_inner_val_fraction,
        help="Fraction of the existing 1000-sample training partition reserved for inner validation.",
    )
    p.add_argument(
        "--split-seed",
        type=int,
        default=CONFIG.qca_v2_split_seed,
        help="Fixed inner split seed; independent of model/training seeds.",
    )
    p.add_argument("--threshold-metric", choices=["mcc", "f1"], default="mcc")
    p.add_argument("--margin-lambda", type=float, default=CONFIG.qca_v2_margin_lambda)
    p.add_argument("--margin", type=float, default=CONFIG.qca_v2_probability_margin)
    p.add_argument("--boundary-weight", type=float, default=CONFIG.qca_v2_boundary_weight)
    p.add_argument("--boundary-threshold", type=float, default=CONFIG.qca_v2_boundary_threshold)
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _load_development_partitions(path: str | Path):
    """Load only development train and outer-validation arrays.

    Deliberately does not access X_test/y_test even though the frozen NPZ also
    contains them. This keeps QCA-v2 development separated from the previously
    inspected confirmatory test partition.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as data:
        required = {"X_train", "y_train", "X_val", "y_val"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise KeyError(f"Missing development arrays: {missing}")

        X_train = np.asarray(data["X_train"], dtype=np.float32).copy()
        y_train = np.asarray(data["y_train"], dtype=np.int64).copy()
        X_outer = np.asarray(data["X_val"], dtype=np.float32).copy()
        y_outer = np.asarray(data["y_val"], dtype=np.int64).copy()

    return X_train, y_train, X_outer, y_outer


def _fixed_inner_split(X, y, *, fraction: float, split_seed: int):
    if not 0.05 <= fraction <= 0.40:
        raise ValueError("inner-val-fraction must be between 0.05 and 0.40.")

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=float(fraction),
        random_state=int(split_seed),
    )
    train_idx, inner_idx = next(splitter.split(X, y))
    return (
        np.asarray(train_idx, dtype=np.int64),
        np.asarray(inner_idx, dtype=np.int64),
    )


def _class_counts(y):
    y = np.asarray(y).astype(int)
    return {"0": int(np.sum(y == 0)), "1": int(np.sum(y == 1))}


def _frozen_qca_config() -> dict:
    path = (
        CONFIG.project_root
        / "results"
        / "qca"
        / CONFIG.experiment_tag
        / "selected_qca_config.json"
    )
    if not path.exists():
        raise FileNotFoundError(
            "Frozen seed-42 QCA configuration is required for fair QCA-v2 development: "
            f"{path}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _score_distribution_metrics(y_true, y_score) -> dict[str, float]:
    y = np.asarray(y_true).astype(int).reshape(-1)
    s = np.asarray(y_score, dtype=float).reshape(-1)
    benign = s[y == 0]
    attack = s[y == 1]

    if len(benign) == 0 or len(attack) == 0:
        return {
            "Benign Mean Score": float("nan"),
            "Attack Mean Score": float("nan"),
            "Score Mean Gap": float("nan"),
            "Benign Median Score": float("nan"),
            "Attack Median Score": float("nan"),
            "KS Statistic": float("nan"),
            "KS p-value": float("nan"),
        }

    ks = stats.ks_2samp(attack, benign, alternative="two-sided", method="auto")
    return {
        "Benign Mean Score": float(np.mean(benign)),
        "Attack Mean Score": float(np.mean(attack)),
        "Score Mean Gap": float(np.mean(attack) - np.mean(benign)),
        "Benign Median Score": float(np.median(benign)),
        "Attack Median Score": float(np.median(attack)),
        "KS Statistic": float(ks.statistic),
        "KS p-value": float(ks.pvalue),
    }


def _prediction_frame(
    *,
    seed: int,
    variant: str,
    split: str,
    y_true,
    y_score,
    threshold: float,
):
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_score = np.asarray(y_score, dtype=float).reshape(-1)
    return pd.DataFrame(
        {
            "Seed": int(seed),
            "Variant": variant,
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
        [
            [int(metrics["TN"]), int(metrics["FP"])],
            [int(metrics["FN"]), int(metrics["TP"])],
        ],
        index=["True 0", "True 1"],
        columns=["Pred 0", "Pred 1"],
    ).to_csv(path)


def _completed(model_dir: Path) -> bool:
    return (
        (model_dir / "metrics.json").exists()
        and (model_dir / "inner_validation_predictions.csv").exists()
        and (model_dir / "outer_validation_predictions.csv").exists()
    )


def _evaluate_and_save(
    *,
    model,
    score_fn,
    seed: int,
    variant: str,
    model_dir: Path,
    X_inner,
    y_inner,
    X_outer,
    y_outer,
    threshold_metric: str,
    train_seconds: float,
    extra: dict | None = None,
):
    model_dir.mkdir(parents=True, exist_ok=True)

    inner_score = np.asarray(score_fn(model, X_inner), dtype=float).reshape(-1)
    threshold, threshold_score, threshold_table = select_threshold(
        y_inner,
        inner_score,
        metric=threshold_metric,
    )

    t0 = perf_counter()
    outer_score = np.asarray(score_fn(model, X_outer), dtype=float).reshape(-1)
    outer_infer = perf_counter() - t0
    outer_pred = (outer_score >= threshold).astype(int)

    outer_metrics = classification_metrics(
        y_outer,
        outer_pred,
        outer_score,
        inference_seconds=outer_infer,
        n_samples=len(y_outer),
    )
    distribution = _score_distribution_metrics(y_outer, outer_score)

    row = {
        "Seed": int(seed),
        "Variant": variant,
        "Threshold": float(threshold),
        f"Inner Validation Threshold {threshold_metric.upper()}": float(threshold_score),
        "Train Seconds": float(train_seconds),
        "Outer Validation Inference Seconds": float(outer_infer),
        **outer_metrics,
        **distribution,
    }
    if extra:
        row.update(extra)

    threshold_table.to_csv(model_dir / "inner_threshold_search.csv", index=False)
    _prediction_frame(
        seed=seed,
        variant=variant,
        split="inner_validation",
        y_true=y_inner,
        y_score=inner_score,
        threshold=threshold,
    ).to_csv(model_dir / "inner_validation_predictions.csv", index=False)
    _prediction_frame(
        seed=seed,
        variant=variant,
        split="outer_validation",
        y_true=y_outer,
        y_score=outer_score,
        threshold=threshold,
    ).to_csv(model_dir / "outer_validation_predictions.csv", index=False)
    _save_confusion(row, model_dir / "outer_validation_confusion_matrix.csv")
    save_json(row, model_dir / "metrics.json")
    return row


def _new_model(seed: int, frozen: dict):
    return QuantumCyberAnathema(
        n_qubits=int(frozen.get("n_qubits", CONFIG.n_qubits)),
        n_layers=int(frozen.get("n_layers", 2)),
        encoding=str(frozen.get("encoding", "angle")),
        seed=int(seed),
    )


def _run_static_vqc(
    *, seed, frozen, arrays, out, threshold_metric, resume
):
    variant = "StaticVQC"
    model_dir = out / f"seed_{seed}" / variant
    if resume and _completed(model_dir):
        print(f"[resume] seed={seed} {variant}")
        return json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))

    X_dev, y_dev, X_inner, y_inner, X_outer, y_outer = arrays
    set_global_seed(seed)
    model = PennyLaneVQC(
        X_dev.shape[1],
        n_layers=int(frozen.get("n_layers", 2)),
        encoding=str(frozen.get("encoding", "angle")),
        seed=seed,
    )

    print(f"[train] seed={seed} {variant}")
    t0 = perf_counter()
    history = fit_vqc(
        model,
        X_dev,
        y_dev,
        X_inner,
        y_inner,
        epochs=int(CONFIG.vqc_epochs),
        batch_size=int(CONFIG.batch_size),
        learning_rate=float(CONFIG.learning_rate),
        patience=int(CONFIG.vqc_patience),
        checkpoint_path=model_dir / "best_vqc.pt",
        verbose=False,
    )
    train_seconds = perf_counter() - t0
    model_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history.to_dict()).to_csv(model_dir / "training_history.csv", index=False)

    return _evaluate_and_save(
        model=model,
        score_fn=lambda m, X: predict_vqc(m, X, batch_size=32),
        seed=seed,
        variant=variant,
        model_dir=model_dir,
        X_inner=X_inner,
        y_inner=y_inner,
        X_outer=X_outer,
        y_outer=y_outer,
        threshold_metric=threshold_metric,
        train_seconds=train_seconds,
        extra={
            "Encoding": str(frozen.get("encoding", "angle")),
            "Layers": int(frozen.get("n_layers", 2)),
            "Epochs Executed": int(len(history.epoch)),
            "Maximum Epoch Budget": int(CONFIG.vqc_epochs),
        },
    )


def _run_weighted_vqc(
    *, seed, frozen, arrays, out, threshold_metric, resume
):
    variant = "WeightedVQC"
    model_dir = out / f"seed_{seed}" / variant
    if resume and _completed(model_dir):
        print(f"[resume] seed={seed} {variant}")
        return json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))

    X_dev, y_dev, X_inner, y_inner, X_outer, y_outer = arrays
    set_global_seed(seed)
    model = PennyLaneVQC(
        X_dev.shape[1],
        n_layers=int(frozen.get("n_layers", 2)),
        encoding=str(frozen.get("encoding", "angle")),
        seed=seed,
    )

    print(f"[train] seed={seed} {variant}")
    t0 = perf_counter()
    history = fit_weighted_vqc(
        model,
        X_dev,
        y_dev,
        X_inner,
        y_inner,
        epochs=int(CONFIG.vqc_epochs),
        batch_size=int(CONFIG.batch_size),
        learning_rate=float(CONFIG.learning_rate),
        patience=int(CONFIG.vqc_patience),
        checkpoint_path=model_dir / "best_weighted_vqc.pt",
        verbose=False,
    )
    train_seconds = perf_counter() - t0
    model_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history.to_dict()).to_csv(model_dir / "training_history.csv", index=False)

    return _evaluate_and_save(
        model=model,
        score_fn=lambda m, X: predict_vqc(m, X, batch_size=32),
        seed=seed,
        variant=variant,
        model_dir=model_dir,
        X_inner=X_inner,
        y_inner=y_inner,
        X_outer=X_outer,
        y_outer=y_outer,
        threshold_metric=threshold_metric,
        train_seconds=train_seconds,
        extra={
            "Encoding": str(frozen.get("encoding", "angle")),
            "Layers": int(frozen.get("n_layers", 2)),
            "Class Weight Negative": float(model.class_weight_negative_),
            "Class Weight Positive": float(model.class_weight_positive_),
            "Epochs Executed": int(len(history.epoch)),
            "Maximum Epoch Budget": int(CONFIG.vqc_epochs),
        },
    )


def _run_qca_current(
    *, seed, frozen, arrays, out, threshold_metric, resume
):
    variant = "QCA_Current"
    model_dir = out / f"seed_{seed}" / variant
    if resume and _completed(model_dir):
        print(f"[resume] seed={seed} {variant}")
        return json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))

    X_dev, y_dev, X_inner, y_inner, X_outer, y_outer = arrays
    set_global_seed(seed)
    model = _new_model(seed, frozen)

    print(f"[train] seed={seed} {variant}")
    t0 = perf_counter()
    history, memory = fit_qca(
        model,
        X_dev,
        y_dev,
        X_inner,
        y_inner,
        cycles=int(frozen.get("cycles", CONFIG.qca_cycles)),
        epochs_per_cycle=int(
            frozen.get("epochs_per_cycle", CONFIG.qca_epochs_per_cycle)
        ),
        warmup_epochs=int(
            frozen.get("warmup_epochs", CONFIG.qca_warmup_epochs)
        ),
        warmup_patience=int(
            frozen.get("warmup_patience", CONFIG.qca_warmup_patience)
        ),
        batch_size=int(frozen.get("batch_size", CONFIG.batch_size)),
        learning_rate=float(frozen.get("learning_rate", CONFIG.learning_rate)),
        memory_capacity=int(frozen.get("memory_capacity", 600)),
        replay_ratio=float(frozen.get("replay_ratio", 0.20)),
        alpha_error=float(frozen.get("alpha_error", 1.0)),
        beta_uncertainty=float(frozen.get("beta_uncertainty", 0.1)),
        gamma_malicious=float(frozen.get("gamma_malicious", 0.25)),
        prioritized_replay=True,
        checkpoint_dir=model_dir / "checkpoints",
        restore_best=True,
        seed=seed,
        verbose=False,
    )
    train_seconds = perf_counter() - t0
    model_dir.mkdir(parents=True, exist_ok=True)
    if "Selected for Test" in history.columns:
        history = history.rename(
            columns={"Selected for Test": "Selected by Inner Validation"}
        )
    history.to_csv(model_dir / "evolution_history.csv", index=False)
    save_json(memory.summary(), model_dir / "memory_summary.json")

    return _evaluate_and_save(
        model=model,
        score_fn=lambda m, X: predict_qca(m, X, batch_size=32),
        seed=seed,
        variant=variant,
        model_dir=model_dir,
        X_inner=X_inner,
        y_inner=y_inner,
        X_outer=X_outer,
        y_outer=y_outer,
        threshold_metric=threshold_metric,
        train_seconds=train_seconds,
        extra={
            "Selected Inner Validation Cycle": int(model.qca_best_cycle_),
            "Selected Inner Validation Defense Score": float(
                model.qca_best_validation_score_
            ),
            "Beta Uncertainty": float(frozen.get("beta_uncertainty", 0.1)),
            "Margin Lambda": 0.0,
            "Replay Priority Mode": "reflexive",
            "Memory Size": int(len(memory)),
            "Maximum Optimization Budget": int(
                frozen.get("warmup_epochs", CONFIG.qca_warmup_epochs)
                + frozen.get("cycles", CONFIG.qca_cycles)
                * frozen.get("epochs_per_cycle", CONFIG.qca_epochs_per_cycle)
            ),
        },
    )


def _run_qca_v2_variant(
    *,
    variant,
    seed,
    frozen,
    arrays,
    out,
    threshold_metric,
    margin_lambda,
    margin,
    replay_priority_mode,
    boundary_weight,
    boundary_threshold,
    resume,
):
    model_dir = out / f"seed_{seed}" / variant
    if resume and _completed(model_dir):
        print(f"[resume] seed={seed} {variant}")
        return json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))

    X_dev, y_dev, X_inner, y_inner, X_outer, y_outer = arrays
    set_global_seed(seed)
    model = _new_model(seed, frozen)

    print(
        f"[train] seed={seed} {variant} "
        f"margin_lambda={margin_lambda} replay={replay_priority_mode}"
    )
    t0 = perf_counter()
    history, memory = fit_qca_v2(
        model,
        X_dev,
        y_dev,
        X_inner,
        y_inner,
        cycles=int(frozen.get("cycles", CONFIG.qca_cycles)),
        epochs_per_cycle=int(
            frozen.get("epochs_per_cycle", CONFIG.qca_epochs_per_cycle)
        ),
        warmup_epochs=int(
            frozen.get("warmup_epochs", CONFIG.qca_warmup_epochs)
        ),
        warmup_patience=int(
            frozen.get("warmup_patience", CONFIG.qca_warmup_patience)
        ),
        batch_size=int(frozen.get("batch_size", CONFIG.batch_size)),
        learning_rate=float(frozen.get("learning_rate", CONFIG.learning_rate)),
        memory_capacity=int(frozen.get("memory_capacity", 600)),
        replay_ratio=float(frozen.get("replay_ratio", 0.20)),
        alpha_error=float(frozen.get("alpha_error", 1.0)),
        beta_uncertainty=float(frozen.get("beta_uncertainty", 0.1)),
        gamma_malicious=float(frozen.get("gamma_malicious", 0.25)),
        margin_lambda=float(margin_lambda),
        margin=float(margin),
        replay_priority_mode=replay_priority_mode,
        boundary_weight=float(boundary_weight),
        boundary_threshold=float(boundary_threshold),
        prioritized_replay=True,
        checkpoint_dir=model_dir / "checkpoints",
        restore_best=True,
        seed=seed,
        verbose=False,
    )
    train_seconds = perf_counter() - t0
    model_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(model_dir / "evolution_history.csv", index=False)
    save_json(memory.summary(), model_dir / "memory_summary.json")

    return _evaluate_and_save(
        model=model,
        score_fn=lambda m, X: predict_qca_v2(m, X, batch_size=32),
        seed=seed,
        variant=variant,
        model_dir=model_dir,
        X_inner=X_inner,
        y_inner=y_inner,
        X_outer=X_outer,
        y_outer=y_outer,
        threshold_metric=threshold_metric,
        train_seconds=train_seconds,
        extra={
            "Selected Inner Validation Cycle": int(model.qca_v2_best_cycle_),
            "Selected Inner Validation Defense Score": float(
                model.qca_v2_best_inner_validation_score_
            ),
            "Beta Uncertainty": float(frozen.get("beta_uncertainty", 0.1)),
            "Margin Lambda": float(margin_lambda),
            "Probability Margin": float(margin),
            "Replay Priority Mode": replay_priority_mode,
            "Boundary Weight": float(boundary_weight),
            "Boundary Threshold": float(boundary_threshold),
            "Memory Size": int(len(memory)),
            "Maximum Optimization Budget": int(
                frozen.get("warmup_epochs", CONFIG.qca_warmup_epochs)
                + frozen.get("cycles", CONFIG.qca_cycles)
                * frozen.get("epochs_per_cycle", CONFIG.qca_epochs_per_cycle)
            ),
        },
    )


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in metrics.groupby("Variant", sort=True):
        for metric in SUMMARY_METRICS:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(float)
            if len(values) == 0:
                continue
            n = len(values)
            mean = float(np.mean(values))
            sd = float(np.std(values, ddof=1)) if n > 1 else float("nan")
            if n > 1 and np.isfinite(sd):
                tcrit = float(stats.t.ppf(0.975, df=n - 1))
                half = tcrit * sd / math.sqrt(n)
                lo, hi = mean - half, mean + half
            else:
                lo = hi = float("nan")
            rows.append(
                {
                    "Variant": variant,
                    "Metric": metric,
                    "N Seeds": n,
                    "Mean": mean,
                    "SD": sd,
                    "95% CI Lower": float(lo),
                    "95% CI Upper": float(hi),
                }
            )
    return pd.DataFrame(rows)


def _exact_sign_flip_pvalue(differences) -> float:
    d = np.asarray(differences, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return float("nan")
    if np.allclose(d, 0.0):
        return 1.0
    observed = abs(float(np.mean(d)))
    permuted = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(d)):
        permuted.append(abs(float(np.mean(d * np.asarray(signs)))))
    permuted = np.asarray(permuted)
    return float(np.mean(permuted >= observed - 1e-15))


def _holm_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p, np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(p))
    if len(valid) == 0:
        return adjusted
    order = valid[np.argsort(p[valid])]
    m = len(order)
    running = 0.0
    for rank, idx in enumerate(order):
        candidate = (m - rank) * p[idx]
        running = max(running, candidate)
        adjusted[idx] = min(1.0, running)
    return adjusted


def paired_comparisons(metrics: pd.DataFrame, reference: str) -> pd.DataFrame:
    rows = []
    ref = metrics[metrics["Variant"] == reference]
    if ref.empty:
        return pd.DataFrame()

    for variant in sorted(metrics["Variant"].unique()):
        if variant == reference:
            continue
        cand = metrics[metrics["Variant"] == variant]
        for metric in PAIRED_METRICS:
            if metric not in metrics.columns:
                continue
            merged = cand[["Seed", metric]].merge(
                ref[["Seed", metric]],
                on="Seed",
                suffixes=("_Candidate", "_Reference"),
            ).sort_values("Seed")
            if merged.empty:
                continue

            c = merged[f"{metric}_Candidate"].to_numpy(float)
            r = merged[f"{metric}_Reference"].to_numpy(float)
            d = c - r
            n = len(d)
            mean_diff = float(np.mean(d))
            sd_diff = float(np.std(d, ddof=1)) if n > 1 else float("nan")
            if n > 1 and np.isfinite(sd_diff):
                tcrit = float(stats.t.ppf(0.975, df=n - 1))
                half = tcrit * sd_diff / math.sqrt(n)
                lo, hi = mean_diff - half, mean_diff + half
            else:
                lo = hi = float("nan")

            try:
                wilcoxon_p = 1.0 if np.allclose(d, 0.0) else float(
                    stats.wilcoxon(
                        c,
                        r,
                        alternative="two-sided",
                        zero_method="pratt",
                        method="auto",
                    ).pvalue
                )
            except Exception:
                wilcoxon_p = float("nan")

            rows.append(
                {
                    "Reference": reference,
                    "Candidate": variant,
                    "Metric": metric,
                    "N Paired Seeds": n,
                    "Candidate Mean": float(np.mean(c)),
                    "Reference Mean": float(np.mean(r)),
                    "Mean Difference Candidate-Reference": mean_diff,
                    "SD Difference": sd_diff,
                    "95% CI Difference Lower": float(lo),
                    "95% CI Difference Upper": float(hi),
                    "Candidate Wins": int(np.sum(d > 0)),
                    "Ties": int(np.sum(np.isclose(d, 0.0))),
                    "Candidate Losses": int(np.sum(d < 0)),
                    "Exact Sign-Flip p": _exact_sign_flip_pvalue(d),
                    "Wilcoxon p": wilcoxon_p,
                }
            )

    result = pd.DataFrame(rows)
    if not result.empty:
        result["Holm-adjusted Sign-Flip p"] = _holm_adjust(
            result["Exact Sign-Flip p"].to_numpy(float)
        )
        result["Holm-adjusted Wilcoxon p"] = _holm_adjust(
            result["Wilcoxon p"].to_numpy(float)
        )
    return result


def _aggregate_predictions(out: Path):
    inner, outer = [], []
    for p in sorted(out.glob("seed_*/*/inner_validation_predictions.csv")):
        inner.append(pd.read_csv(p))
    for p in sorted(out.glob("seed_*/*/outer_validation_predictions.csv")):
        outer.append(pd.read_csv(p))
    if inner:
        pd.concat(inner, ignore_index=True).to_csv(
            out / "all_inner_validation_predictions.csv", index=False
        )
    if outer:
        pd.concat(outer, ignore_index=True).to_csv(
            out / "all_outer_validation_predictions.csv", index=False
        )


def _prediction_behavior_vs_static(out: Path) -> pd.DataFrame:
    path = out / "all_outer_validation_predictions.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    rows = []
    for seed in sorted(df["Seed"].unique()):
        s = df[df["Seed"] == seed]
        ref = s[s["Variant"] == "StaticVQC"][
            ["Sample Index", "y_true", "y_score", "y_pred"]
        ].rename(
            columns={
                "y_score": "y_score_static",
                "y_pred": "y_pred_static",
            }
        )
        if ref.empty:
            continue
        for variant in sorted(s["Variant"].unique()):
            if variant == "StaticVQC":
                continue
            cand = s[s["Variant"] == variant][
                ["Sample Index", "y_true", "y_score", "y_pred"]
            ].rename(
                columns={
                    "y_score": "y_score_candidate",
                    "y_pred": "y_pred_candidate",
                }
            )
            merged = cand.merge(
                ref,
                on=["Sample Index", "y_true"],
                how="inner",
            )
            if merged.empty:
                continue
            disagree = merged["y_pred_candidate"] != merged["y_pred_static"]
            static_correct = merged["y_pred_static"] == merged["y_true"]
            cand_correct = merged["y_pred_candidate"] == merged["y_true"]
            corr = np.corrcoef(
                merged["y_score_candidate"],
                merged["y_score_static"],
            )[0, 1]
            rows.append(
                {
                    "Seed": int(seed),
                    "Variant": variant,
                    "N Outer Validation Samples": int(len(merged)),
                    "Decision Disagreement Rate vs Static": float(np.mean(disagree)),
                    "Changed Static Error to Correct": int(
                        np.sum(disagree & (~static_correct) & cand_correct)
                    ),
                    "Changed Static Correct to Error": int(
                        np.sum(disagree & static_correct & (~cand_correct))
                    ),
                    "Score Correlation vs Static": float(corr),
                    "Mean Absolute Score Difference vs Static": float(
                        np.mean(
                            np.abs(
                                merged["y_score_candidate"]
                                - merged["y_score_static"]
                            )
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def _variant_ranking(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    pivot = summary.pivot_table(
        index="Variant",
        columns="Metric",
        values="Mean",
        aggfunc="first",
    ).reset_index()
    keep = [
        c
        for c in [
            "Variant",
            "PR-AUC",
            "MCC",
            "F1",
            "ROC-AUC",
            "Score Mean Gap",
            "KS Statistic",
        ]
        if c in pivot.columns
    ]
    ranking = pivot[keep].copy()
    sort_cols = [c for c in ["PR-AUC", "MCC", "Score Mean Gap"] if c in ranking.columns]
    if sort_cols:
        ranking = ranking.sort_values(sort_cols, ascending=False)
    ranking.insert(0, "Development Rank", np.arange(1, len(ranking) + 1))
    return ranking


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_outer, y_outer = _load_development_partitions(args.data)
    if X_train.shape != (CONFIG.benchmark_train, CONFIG.n_qubits):
        raise ValueError(
            "Expected frozen 1000x4 development training partition; "
            f"got {X_train.shape}."
        )
    if X_outer.shape != (CONFIG.benchmark_val, CONFIG.n_qubits):
        raise ValueError(
            "Expected frozen 300x4 outer validation partition; "
            f"got {X_outer.shape}."
        )

    train_idx, inner_idx = _fixed_inner_split(
        X_train,
        y_train,
        fraction=args.inner_val_fraction,
        split_seed=args.split_seed,
    )
    X_dev, y_dev = X_train[train_idx], y_train[train_idx]
    X_inner, y_inner = X_train[inner_idx], y_train[inner_idx]
    arrays = (X_dev, y_dev, X_inner, y_inner, X_outer, y_outer)

    np.savez_compressed(
        out / "fixed_development_split_indices.npz",
        development_train_indices=train_idx,
        inner_validation_indices=inner_idx,
    )

    frozen = _frozen_qca_config()

    manifest = {
        "dataset": CONFIG.dataset_name,
        "source_experiment_tag": CONFIG.experiment_tag,
        "development_tag": out.name,
        "data_path": str(Path(args.data).resolve()),
        "test_partition_accessed": False,
        "development_train_source": "original matched benchmark X_train/y_train only",
        "outer_validation_source": "original matched benchmark X_val/y_val",
        "development_train_size": int(len(y_dev)),
        "inner_validation_size": int(len(y_inner)),
        "outer_validation_size": int(len(y_outer)),
        "class_counts": {
            "development_train": _class_counts(y_dev),
            "inner_validation": _class_counts(y_inner),
            "outer_validation": _class_counts(y_outer),
        },
        "fixed_inner_split_seed": int(args.split_seed),
        "inner_validation_fraction": float(args.inner_val_fraction),
        "model_seeds": [int(s) for s in args.seeds],
        "variants": list(args.variants),
        "n_qubits": int(CONFIG.n_qubits),
        "static_vqc_maximum_epochs": int(CONFIG.vqc_epochs),
        "qca_maximum_budget": int(
            frozen.get("warmup_epochs", CONFIG.qca_warmup_epochs)
            + frozen.get("cycles", CONFIG.qca_cycles)
            * frozen.get("epochs_per_cycle", CONFIG.qca_epochs_per_cycle)
        ),
        "frozen_qca_configuration": frozen,
        "qca_v2_parameters": {
            "margin_lambda": float(args.margin_lambda),
            "probability_margin": float(args.margin),
            "boundary_weight": float(args.boundary_weight),
            "boundary_threshold": float(args.boundary_threshold),
        },
        "threshold_policy": (
            f"threshold selected on inner validation by {args.threshold_metric.upper()} "
            "and frozen before outer-validation evaluation"
        ),
        "cycle_selection_policy": "inner-validation Defense Score only",
        "scientific_role": (
            "development/mechanism analysis only; outer validation may rank variants, "
            "but the previously inspected 300-sample test partition is not used"
        ),
    }
    save_json(manifest, out / "development_manifest.json")

    print("\n=== QCA-v2 VALIDATION-ONLY DEVELOPMENT ===")
    print("Data:", Path(args.data))
    print("Original training partition:", X_train.shape)
    print("Development train:", X_dev.shape, _class_counts(y_dev))
    print("Inner validation:", X_inner.shape, _class_counts(y_inner))
    print("Outer validation:", X_outer.shape, _class_counts(y_outer))
    print("TEST PARTITION ACCESSED: False")
    print("Model seeds:", args.seeds)
    print("Variants:", args.variants)
    print("Margin lambda:", args.margin_lambda)
    print("Probability margin:", args.margin)
    print("Boundary weight:", args.boundary_weight)
    print("Boundary threshold:", args.boundary_threshold)
    print("Output:", out)

    if args.dry_run:
        print("\nDry run complete. No models were trained.")
        return

    rows = []
    failures = []

    for seed in args.seeds:
        print(f"\n\n================ SEED {seed} ================")
        for variant in args.variants:
            try:
                if variant == "StaticVQC":
                    row = _run_static_vqc(
                        seed=seed,
                        frozen=frozen,
                        arrays=arrays,
                        out=out,
                        threshold_metric=args.threshold_metric,
                        resume=args.resume,
                    )
                elif variant == "WeightedVQC":
                    row = _run_weighted_vqc(
                        seed=seed,
                        frozen=frozen,
                        arrays=arrays,
                        out=out,
                        threshold_metric=args.threshold_metric,
                        resume=args.resume,
                    )
                elif variant == "QCA_Current":
                    row = _run_qca_current(
                        seed=seed,
                        frozen=frozen,
                        arrays=arrays,
                        out=out,
                        threshold_metric=args.threshold_metric,
                        resume=args.resume,
                    )
                elif variant == "QCA_Margin":
                    row = _run_qca_v2_variant(
                        variant=variant,
                        seed=seed,
                        frozen=frozen,
                        arrays=arrays,
                        out=out,
                        threshold_metric=args.threshold_metric,
                        margin_lambda=args.margin_lambda,
                        margin=args.margin,
                        replay_priority_mode="reflexive",
                        boundary_weight=args.boundary_weight,
                        boundary_threshold=args.boundary_threshold,
                        resume=args.resume,
                    )
                elif variant == "QCA_HardReplay":
                    row = _run_qca_v2_variant(
                        variant=variant,
                        seed=seed,
                        frozen=frozen,
                        arrays=arrays,
                        out=out,
                        threshold_metric=args.threshold_metric,
                        margin_lambda=0.0,
                        margin=args.margin,
                        replay_priority_mode="boundary",
                        boundary_weight=args.boundary_weight,
                        boundary_threshold=args.boundary_threshold,
                        resume=args.resume,
                    )
                elif variant == "QCA_MarginHardReplay":
                    row = _run_qca_v2_variant(
                        variant=variant,
                        seed=seed,
                        frozen=frozen,
                        arrays=arrays,
                        out=out,
                        threshold_metric=args.threshold_metric,
                        margin_lambda=args.margin_lambda,
                        margin=args.margin,
                        replay_priority_mode="boundary",
                        boundary_weight=args.boundary_weight,
                        boundary_threshold=args.boundary_threshold,
                        resume=args.resume,
                    )
                else:
                    raise ValueError(variant)
                rows.append(row)
            except Exception as exc:
                failures.append(
                    {
                        "Seed": int(seed),
                        "Variant": variant,
                        "Error": repr(exc),
                        "Traceback": traceback.format_exc(),
                    }
                )
                print(f"[FAILED] seed={seed} {variant}: {exc!r}")
                if args.fail_fast:
                    raise

        if rows:
            pd.DataFrame(rows).drop_duplicates(
                subset=["Seed", "Variant"], keep="last"
            ).sort_values(["Seed", "Variant"]).to_csv(
                out / "all_seed_outer_validation_metrics.csv", index=False
            )
        if failures:
            pd.DataFrame(failures).to_csv(out / "failures.csv", index=False)
        _aggregate_predictions(out)

    completed_rows = []
    for path in sorted(out.glob("seed_*/*/metrics.json")):
        try:
            completed_rows.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            pass

    metrics = pd.DataFrame(completed_rows)
    if metrics.empty:
        raise RuntimeError("No completed QCA-v2 development runs were found.")
    metrics = metrics.drop_duplicates(
        subset=["Seed", "Variant"], keep="last"
    ).sort_values(["Seed", "Variant"])
    metrics.to_csv(out / "all_seed_outer_validation_metrics.csv", index=False)

    summary = summarize_metrics(metrics)
    summary.to_csv(out / "summary_mean_sd_95CI.csv", index=False)

    paired_static = paired_comparisons(metrics, "StaticVQC")
    paired_static.to_csv(out / "paired_comparisons_vs_StaticVQC.csv", index=False)

    paired_weighted = paired_comparisons(metrics, "WeightedVQC")
    paired_weighted.to_csv(out / "paired_comparisons_vs_WeightedVQC.csv", index=False)

    ranking = _variant_ranking(summary)
    ranking.to_csv(out / "outer_validation_variant_ranking.csv", index=False)

    _aggregate_predictions(out)
    behavior = _prediction_behavior_vs_static(out)
    behavior.to_csv(out / "prediction_behavior_vs_StaticVQC.csv", index=False)

    print("\n=== QCA-v2 DEVELOPMENT COMPLETE / CURRENTLY AVAILABLE ===")
    print("Completed variant-seed rows:", len(metrics))
    print("Saved:", out / "all_seed_outer_validation_metrics.csv")
    print("Saved:", out / "summary_mean_sd_95CI.csv")
    print("Saved:", out / "paired_comparisons_vs_StaticVQC.csv")
    print("Saved:", out / "paired_comparisons_vs_WeightedVQC.csv")
    print("Saved:", out / "prediction_behavior_vs_StaticVQC.csv")
    print("Saved:", out / "outer_validation_variant_ranking.csv")
    print("Saved:", out / "all_outer_validation_predictions.csv")
    print("\nThe frozen test partition was not loaded by this development runner.")
    if failures:
        print("Some runs failed; inspect:", out / "failures.csv")


if __name__ == "__main__":
    main()
