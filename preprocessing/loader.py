from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence
import re

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetSpec:
    """Schema-tolerant configuration for a benchmark intrusion dataset."""

    name: str
    aliases: tuple[str, ...]
    label_candidates: tuple[str, ...]
    benign_values: tuple[str, ...]
    file_patterns: tuple[str, ...] = ("*.csv", "*.parquet", "*.pq")
    always_drop_candidates: tuple[str, ...] = field(default_factory=tuple)


DATASET_SPECS: dict[str, DatasetSpec] = {
    "UNSW-NB15": DatasetSpec(
        name="UNSW-NB15",
        aliases=("unsw", "unsw_nb15", "unsw-nb15"),
        label_candidates=("label", "Label", "target", "class"),
        benign_values=("0", "benign", "normal", "normal traffic"),
        always_drop_candidates=("attack_cat", "id"),
    ),
    "CICIDS2017": DatasetSpec(
        name="CICIDS2017",
        aliases=("cicids2017", "cic_ids_2017", "ids2017"),
        label_candidates=("Label", "label", "target", "class"),
        benign_values=("benign", "normal", "0"),
        always_drop_candidates=("Flow ID", "Src IP", "Dst IP", "Timestamp"),
    ),
    "CSE-CIC-IDS2018": DatasetSpec(
        name="CSE-CIC-IDS2018",
        aliases=("cse-cic-ids2018", "csecicids2018", "ids2018"),
        label_candidates=("Label", "label", "target", "class"),
        benign_values=("benign", "normal", "0"),
        always_drop_candidates=("Flow ID", "Src IP", "Dst IP", "Timestamp"),
    ),
    "BoT-IoT": DatasetSpec(
        name="BoT-IoT",
        aliases=("bot-iot", "botiot", "bot_iot"),
        label_candidates=("label", "Label", "target", "class"),
        benign_values=("0", "benign", "normal"),
        always_drop_candidates=("category", "subcategory", "pkSeqID"),
    ),
    "TON_IoT": DatasetSpec(
        name="TON_IoT",
        aliases=("ton_iot", "ton-iot", "toniot"),
        label_candidates=("label", "Label", "target", "class"),
        benign_values=("0", "benign", "normal"),
        always_drop_candidates=("type",),
    ),
}


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def resolve_dataset_spec(name: str) -> DatasetSpec:
    needle = _normalize_name(name)
    for spec in DATASET_SPECS.values():
        candidates = (spec.name,) + spec.aliases
        if needle in {_normalize_name(x) for x in candidates}:
            return spec
    valid = ", ".join(DATASET_SPECS)
    raise ValueError(f"Unknown dataset {name!r}. Valid names: {valid}")


def _read_one(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path}")


def _collect_files(dataset_path: Path, patterns: Sequence[str]) -> list[Path]:
    if dataset_path.is_file():
        return [dataset_path]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(dataset_path.rglob(pattern))
    files = sorted({p.resolve() for p in files if p.is_file()})
    if not files:
        raise FileNotFoundError(
            f"No supported dataset files found under {dataset_path}. "
            f"Expected one of: {patterns}"
        )
    return files


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [
        re.sub(r"\s+", " ", str(c).replace("\ufeff", "").strip())
        for c in out.columns
    ]
    return out


def infer_label_column(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    exact = {str(c): str(c) for c in df.columns}
    lower = {str(c).lower(): str(c) for c in df.columns}

    for candidate in candidates:
        if candidate in exact:
            return exact[candidate]
        if candidate.lower() in lower:
            return lower[candidate.lower()]

    semantic = {"label", "target", "class", "binary_label", "attack_label"}
    for c in df.columns:
        if _normalize_name(c) in {_normalize_name(x) for x in semantic}:
            return str(c)

    raise ValueError(
        "Could not infer the label column safely. "
        f"Available columns include: {list(df.columns)[:25]}. "
        "Pass --label-column explicitly."
    )


def standardize_binary_label(
    labels: pd.Series,
    *,
    benign_values: Iterable[str] = ("0", "benign", "normal"),
) -> pd.Series:
    """Convert labels to QCA's canonical target: 0=benign, 1=malicious."""
    y = labels.copy()

    numeric = pd.to_numeric(y, errors="coerce")
    non_null_original = y.notna().sum()
    if numeric.notna().sum() == non_null_original and non_null_original > 0:
        unique = set(numeric.dropna().astype(float).unique().tolist())
        if unique.issubset({0.0, 1.0}):
            return numeric.astype("Int64").astype(int)
        raise ValueError(
            "Numeric target is not binary {0,1}. "
            "Do not silently collapse numeric classes; supply an explicit mapping."
        )

    normalized = (
        y.astype("string")
        .str.strip()
        .str.lower()
        .replace({"": pd.NA, "nan": pd.NA, "none": pd.NA})
    )

    benign = {str(v).strip().lower() for v in benign_values}
    missing_mask = normalized.isna()
    if missing_mask.any():
        raise ValueError(
            f"Target contains {int(missing_mask.sum())} missing labels. "
            "Resolve them before training."
        )

    benign_mask = normalized.isin(benign)
    if benign_mask.sum() == 0:
        observed = normalized.value_counts().head(15).to_dict()
        raise ValueError(
            "No benign label matched the configured benign values. "
            f"Observed target values: {observed}. "
            f"Configured benign values: {sorted(benign)}"
        )

    return (~benign_mask).astype(int)


def _stratified_sample_indices(
    y: pd.Series,
    n_samples: int,
    *,
    random_state: int = 42,
) -> np.ndarray:
    y_arr = np.asarray(y).reshape(-1)
    n_total = len(y_arr)
    if n_samples >= n_total:
        return np.arange(n_total, dtype=int)
    if n_samples < 2:
        raise ValueError("max_rows must be >= 2 for binary classification.")

    classes, counts = np.unique(y_arr, return_counts=True)
    if len(classes) < 2:
        raise ValueError("Binary dataset must contain both benign and malicious samples.")
    if n_samples < len(classes):
        raise ValueError(
            f"max_rows={n_samples} is too small to represent {len(classes)} classes."
        )

    ideal = counts / counts.sum() * n_samples
    alloc = np.floor(ideal).astype(int)
    alloc = np.maximum(alloc, 1)
    alloc = np.minimum(alloc, counts)

    while alloc.sum() > n_samples:
        candidates = np.where(alloc > 1)[0]
        if len(candidates) == 0:
            break
        j = candidates[np.argmax(alloc[candidates] - ideal[candidates])]
        alloc[j] -= 1

    remainder = n_samples - int(alloc.sum())
    fractional = ideal - np.floor(ideal)
    order = np.argsort(-fractional)
    for j in order:
        if remainder <= 0:
            break
        room = counts[j] - alloc[j]
        if room > 0:
            take = min(int(room), remainder)
            alloc[j] += take
            remainder -= take

    rng = np.random.default_rng(random_state)
    selected: list[np.ndarray] = []
    for cls, n_cls in zip(classes, alloc):
        idx = np.flatnonzero(y_arr == cls)
        selected.append(rng.choice(idx, size=int(n_cls), replace=False))
    out = np.concatenate(selected)
    rng.shuffle(out)
    return out.astype(int)


def load_dataset(
    dataset: str,
    path: str | Path,
    *,
    label_column: str | None = None,
    max_rows: int | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Load a supported intrusion benchmark from a file or directory."""
    spec = resolve_dataset_spec(dataset)
    dataset_path = Path(path).expanduser().resolve()
    files = _collect_files(dataset_path, spec.file_patterns)

    frames = []
    for f in files:
        frame = canonicalize_columns(_read_one(f))
        frame["__source_file__"] = f.name
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True, sort=False)
    rows_loaded = int(len(df))

    label_column = label_column or infer_label_column(df, spec.label_candidates)
    if label_column not in df.columns:
        raise KeyError(f"Label column {label_column!r} not found.")

    y = standardize_binary_label(
        df[label_column],
        benign_values=spec.benign_values,
    ).reset_index(drop=True)

    sampled = False
    if max_rows is not None and len(df) > max_rows:
        idx = _stratified_sample_indices(
            y,
            int(max_rows),
            random_state=random_state,
        )
        df = df.iloc[idx].reset_index(drop=True)
        y = y.iloc[idx].reset_index(drop=True)
        sampled = True

    X = df.drop(columns=[label_column]).copy()

    metadata = {
        "dataset": spec.name,
        "path": str(dataset_path),
        "files": [str(f) for f in files],
        "rows_loaded_before_sampling": rows_loaded,
        "rows": int(len(df)),
        "stratified_subsample_applied": sampled,
        "max_rows": None if max_rows is None else int(max_rows),
        "columns_before_target_drop": int(df.shape[1]),
        "label_column": label_column,
        "class_counts": {
            str(k): int(v) for k, v in y.value_counts().sort_index().items()
        },
        "always_drop_candidates": list(spec.always_drop_candidates),
    }
    return X, y, metadata
