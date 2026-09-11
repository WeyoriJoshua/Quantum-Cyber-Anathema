from __future__ import annotations

from pathlib import Path
import json
import random

import numpy as np
import pandas as pd


def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    try:
        from qiskit_machine_learning.utils import algorithm_globals
        algorithm_globals.random_seed = seed
    except Exception:
        pass


def load_quantum_splits(path: str | Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Quantum split file not found: {path.resolve()}\n"
            "Run run_preprocessing.py first or correct the experiment path."
        )

    with np.load(path, allow_pickle=False) as data:
        required = {
            "X_train", "y_train", "X_val", "y_val", "X_test", "y_test"
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise KeyError(f"Missing arrays in {path}: {missing}")
        result = {key: np.asarray(data[key]) for key in required}

    for split in ("train", "val", "test"):
        X = result[f"X_{split}"]
        y = result[f"y_{split}"].reshape(-1)
        if X.ndim != 2:
            raise ValueError(f"X_{split} must be 2D, got {X.shape}")
        if len(X) != len(y):
            raise ValueError(f"X_{split}/y_{split} length mismatch")
        if not np.isfinite(X).all():
            raise ValueError(f"X_{split} contains NaN or infinity")
        classes = set(np.unique(y).tolist())
        if not classes.issubset({0, 1}):
            raise ValueError(f"y_{split} must contain binary labels 0/1, got {classes}")
        result[f"y_{split}"] = y.astype(np.int64)

    n_features = {
        result["X_train"].shape[1],
        result["X_val"].shape[1],
        result["X_test"].shape[1],
    }
    if len(n_features) != 1:
        raise ValueError("Train/validation/test feature dimensions do not match.")
    return result


def stratified_subsample_arrays(X, y, n: int | None, *, seed: int = 42):
    """Reproducible class-proportional subset for expensive quantum experiments."""
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    if n is None or n >= len(y):
        return X, y
    if n < 2:
        raise ValueError("Subset size must be >=2 for binary classification.")

    classes, counts = np.unique(y, return_counts=True)
    if len(classes) != 2:
        raise ValueError("Expected two classes before stratified subsampling.")

    ideal = counts / counts.sum() * n
    alloc = np.maximum(np.floor(ideal).astype(int), 1)
    alloc = np.minimum(alloc, counts)
    while alloc.sum() > n:
        candidates = np.flatnonzero(alloc > 1)
        j = candidates[np.argmax(alloc[candidates] - ideal[candidates])]
        alloc[j] -= 1
    remainder = n - int(alloc.sum())
    for j in np.argsort(-(ideal - np.floor(ideal))):
        if remainder <= 0:
            break
        room = int(counts[j] - alloc[j])
        if room:
            take = min(room, remainder)
            alloc[j] += take
            remainder -= take

    rng = np.random.default_rng(seed)
    idx_parts = []
    for cls, n_cls in zip(classes, alloc):
        idx = np.flatnonzero(y == cls)
        idx_parts.append(rng.choice(idx, size=int(n_cls), replace=False))
    idx = np.concatenate(idx_parts)
    rng.shuffle(idx)
    return X[idx], y[idx]


def create_matched_benchmark_splits(
    data: dict,
    *,
    train_size: int = 600,
    val_size: int = 200,
    test_size: int = 200,
    seed: int = 42,
):
    """Create one reproducible class-proportional benchmark for every model."""
    X_train, y_train = stratified_subsample_arrays(
        data["X_train"], data["y_train"], train_size, seed=seed
    )
    X_val, y_val = stratified_subsample_arrays(
        data["X_val"], data["y_val"], val_size, seed=seed + 1
    )
    X_test, y_test = stratified_subsample_arrays(
        data["X_test"], data["y_test"], test_size, seed=seed + 2
    )
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }


def save_quantum_splits(data: dict, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X_train=np.asarray(data["X_train"]),
        y_train=np.asarray(data["y_train"], dtype=np.int64),
        X_val=np.asarray(data["X_val"]),
        y_val=np.asarray(data["y_val"], dtype=np.int64),
        X_test=np.asarray(data["X_test"]),
        y_test=np.asarray(data["y_test"], dtype=np.int64),
    )
    return path


def save_json(obj, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")


def save_metrics_table(rows: list[dict], path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df
