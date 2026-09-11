from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from evaluation.metrics import classification_metrics, predict_scores


def _optional_xgboost(random_state: int = 42):
    try:
        from xgboost import XGBClassifier
    except Exception:
        return None

    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )


def build_classical_models(random_state: int = 42) -> dict[str, Any]:
    models: dict[str, Any] = {
        "LogisticRegression": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    xgb = _optional_xgboost(random_state)
    if xgb is not None:
        models["XGBoost"] = xgb

    return models


@dataclass
class ClassicalResult:
    name: str
    model: Any
    metrics: dict
    train_seconds: float
    inference_seconds: float


def train_evaluate_classical(
    models: dict[str, Any],
    X_train,
    y_train,
    X_eval,
    y_eval,
) -> list[ClassicalResult]:
    results: list[ClassicalResult] = []

    for name, estimator in models.items():
        model = clone(estimator)

        # Match the imbalance-aware treatment used by the other classical
        # controls. XGBoost does not have class_weight, so derive
        # scale_pos_weight strictly from the training partition.
        if name == "XGBoost" and hasattr(model, "set_params"):
            y_arr = np.asarray(y_train).astype(int).reshape(-1)
            n_pos = int(np.sum(y_arr == 1))
            n_neg = int(np.sum(y_arr == 0))
            if n_pos > 0:
                model.set_params(scale_pos_weight=n_neg / n_pos)

        t0 = perf_counter()
        model.fit(X_train, y_train)
        train_seconds = perf_counter() - t0

        t1 = perf_counter()
        scores = predict_scores(model, X_eval)
        pred = (scores >= 0.5).astype(int)
        inference_seconds = perf_counter() - t1

        metrics = classification_metrics(
            y_eval,
            pred,
            scores,
            inference_seconds=inference_seconds,
            n_samples=len(y_eval),
        )
        metrics["Train Seconds"] = float(train_seconds)

        results.append(
            ClassicalResult(
                name=name,
                model=model,
                metrics=metrics,
                train_seconds=train_seconds,
                inference_seconds=inference_seconds,
            )
        )

    return results


def fit_classical_models(models: dict[str, Any], X_train, y_train):
    """Fit each classical control once and return model/timing pairs."""
    fitted = {}
    for name, estimator in models.items():
        model = clone(estimator)
        if name == "XGBoost" and hasattr(model, "set_params"):
            y_arr = np.asarray(y_train).astype(int).reshape(-1)
            n_pos = int(np.sum(y_arr == 1))
            n_neg = int(np.sum(y_arr == 0))
            if n_pos > 0:
                model.set_params(scale_pos_weight=n_neg / n_pos)
        t0 = perf_counter()
        model.fit(X_train, y_train)
        fitted[name] = {
            "model": model,
            "train_seconds": float(perf_counter() - t0),
        }
    return fitted
