from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from project_config import CONFIG
from experiments.common import save_quantum_splits
from preprocessing.quantum_pipeline import QuantumTabularPreprocessor


DROP_COLUMNS = ("id", "attack_cat")
TARGET_COLUMN = "label"
EXPECTED_COLUMNS = {
    "id", "dur", "proto", "service", "state", "spkts", "dpkts", "sbytes",
    "dbytes", "rate", "sttl", "dttl", "sload", "dload", "sloss", "dloss",
    "sinpkt", "dinpkt", "sjit", "djit", "swin", "stcpb", "dtcpb", "dwin",
    "tcprtt", "synack", "ackdat", "smean", "dmean", "trans_depth",
    "response_body_len", "ct_srv_src", "ct_state_ttl", "ct_dst_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "is_ftp_login",
    "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst",
    "is_sm_ips_ports", "attack_cat", "label",
}


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Prepare a leakage-resistant matched UNSW-NB15 external-validation benchmark. "
            "The official training file supplies train/validation data; the official testing "
            "file supplies the untouched external test data."
        )
    )
    p.add_argument("--training", default=str(CONFIG.unsw_training_path))
    p.add_argument("--testing", default=str(CONFIG.unsw_testing_path))
    p.add_argument("--out", default=str(CONFIG.unsw_preprocessed_dir))
    p.add_argument("--train-size", type=int, default=CONFIG.unsw_train_size)
    p.add_argument("--val-size", type=int, default=CONFIG.unsw_val_size)
    p.add_argument("--test-size", type=int, default=CONFIG.unsw_test_size)
    p.add_argument("--qubits", type=int, default=CONFIG.n_qubits)
    p.add_argument("--seed", type=int, default=CONFIG.unsw_benchmark_seed)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _counts(y) -> dict[str, int]:
    values, counts = np.unique(np.asarray(y).astype(int), return_counts=True)
    return {str(int(k)): int(v) for k, v in zip(values, counts)}


def _load(path: Path, split: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    missing = sorted(EXPECTED_COLUMNS.difference(df.columns))
    if missing:
        raise ValueError(f"{split}: missing expected UNSW-NB15 columns: {missing}")

    y = pd.to_numeric(df[TARGET_COLUMN], errors="raise").astype(int)
    if set(y.unique()) != {0, 1}:
        raise ValueError(f"{split}: label must contain exactly {{0,1}}")

    df = df.copy()
    df["__source_row__"] = np.arange(len(df), dtype=np.int64)
    return df


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c not in {TARGET_COLUMN, *DROP_COLUMNS, "__source_row__"}
    ]


def _clean_split(df: pd.DataFrame, *, split: str) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    feature_cols = _feature_columns(df)
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
    if len(numeric_cols):
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    hashes = pd.util.hash_pandas_object(df[feature_cols], index=False).astype("uint64")
    tmp = pd.DataFrame({"h": hashes.to_numpy(), "y": df[TARGET_COLUMN].to_numpy()})
    label_counts = tmp.groupby("h", sort=False)["y"].nunique()
    conflict_hashes = set(label_counts[label_counts > 1].index.tolist())

    conflict_mask = hashes.isin(conflict_hashes)
    conflict_rows = int(conflict_mask.sum())
    if conflict_rows:
        df = df.loc[~conflict_mask].copy()

    pair_subset = feature_cols + [TARGET_COLUMN]
    duplicate_pair_mask = df.duplicated(subset=pair_subset, keep="first")
    duplicate_pairs = int(duplicate_pair_mask.sum())
    if duplicate_pairs:
        df = df.loc[~duplicate_pair_mask].copy()

    df = df.reset_index(drop=True)
    report = {
        "split": split,
        "rows_after": int(len(df)),
        "class_counts_after": _counts(df[TARGET_COLUMN]),
        "conflicting_feature_rows_removed": conflict_rows,
        "duplicate_feature_label_pairs_removed": duplicate_pairs,
        "feature_columns": feature_cols,
    }
    return df, report


def _remove_cross_split_overlap(train_df: pd.DataFrame, test_df: pd.DataFrame):
    feature_cols = _feature_columns(train_df)
    if feature_cols != _feature_columns(test_df):
        raise ValueError("Training/testing feature schemas do not match after fixed drops.")

    train_hashes = pd.util.hash_pandas_object(train_df[feature_cols], index=False).astype("uint64")
    test_hashes = pd.util.hash_pandas_object(test_df[feature_cols], index=False).astype("uint64")
    train_set = set(train_hashes.tolist())
    overlap_mask = test_hashes.isin(train_set)
    removed = int(overlap_mask.sum())
    decontaminated = test_df.loc[~overlap_mask].reset_index(drop=True)
    return decontaminated, {
        "test_rows_removed_for_feature_overlap_with_official_training": removed,
        "test_rows_after_cross_split_decontamination": int(len(decontaminated)),
        "test_class_counts_after_cross_split_decontamination": _counts(decontaminated[TARGET_COLUMN]),
    }


def _stratified_indices(y, n: int, seed: int) -> np.ndarray:
    y = np.asarray(y).astype(int)
    if n > len(y):
        raise ValueError(f"Requested {n} samples from only {len(y)} rows.")
    if n == len(y):
        return np.arange(len(y), dtype=np.int64)
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=int(n), random_state=int(seed))
    chosen, _ = next(splitter.split(np.zeros((len(y), 1)), y))
    return np.asarray(chosen, dtype=np.int64)


def _assert_disjoint_feature_rows(named_frames: dict[str, pd.DataFrame]):
    hashes = {}
    for name, df in named_frames.items():
        feature_cols = _feature_columns(df)
        hashes[name] = set(
            pd.util.hash_pandas_object(df[feature_cols], index=False).astype("uint64").tolist()
        )
    names = list(hashes)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = hashes[names[i]].intersection(hashes[names[j]])
            if overlap:
                raise RuntimeError(
                    f"Feature-row overlap remains between {names[i]} and {names[j]}: {len(overlap)} hash groups"
                )


def main():
    args = parse_args()
    train_path = Path(args.training).expanduser().resolve()
    test_path = Path(args.testing).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()

    raw_train = _load(train_path, "official_training")
    raw_test = _load(test_path, "official_testing")

    if list(raw_train.drop(columns=["__source_row__"]).columns) != list(raw_test.drop(columns=["__source_row__"]).columns):
        raise ValueError("Official training/testing CSV schemas differ.")

    train_clean, train_report = _clean_split(raw_train, split="official_training")
    test_clean, test_report = _clean_split(raw_test, split="official_testing")
    test_clean, cross_report = _remove_cross_split_overlap(train_clean, test_clean)

    need_train = int(args.train_size) + int(args.val_size)
    train_pool_idx = _stratified_indices(train_clean[TARGET_COLUMN].to_numpy(), need_train, args.seed)
    dev_pool = train_clean.iloc[train_pool_idx].reset_index(drop=True)

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=int(args.train_size),
        test_size=int(args.val_size),
        random_state=int(args.seed),
    )
    train_idx, val_idx = next(
        splitter.split(np.zeros((len(dev_pool), 1)), dev_pool[TARGET_COLUMN].to_numpy())
    )
    benchmark_train = dev_pool.iloc[train_idx].reset_index(drop=True)
    benchmark_val = dev_pool.iloc[val_idx].reset_index(drop=True)

    test_idx = _stratified_indices(test_clean[TARGET_COLUMN].to_numpy(), int(args.test_size), args.seed + 1)
    benchmark_test = test_clean.iloc[test_idx].reset_index(drop=True)

    _assert_disjoint_feature_rows({
        "benchmark_train": benchmark_train,
        "benchmark_validation": benchmark_val,
        "benchmark_test": benchmark_test,
    })

    feature_cols = _feature_columns(benchmark_train)
    X_train = benchmark_train[feature_cols].copy()
    y_train = benchmark_train[TARGET_COLUMN].astype(int).to_numpy()
    X_val = benchmark_val[feature_cols].copy()
    y_val = benchmark_val[TARGET_COLUMN].astype(int).to_numpy()
    X_test = benchmark_test[feature_cols].copy()
    y_test = benchmark_test[TARGET_COLUMN].astype(int).to_numpy()

    report = {
        "dataset": "UNSW-NB15",
        "purpose": "External validation of frozen QCA-v2 selected on CICIDS2017",
        "raw_files": {
            "training": str(train_path),
            "testing": str(test_path),
            "training_sha256": _sha256(train_path),
            "testing_sha256": _sha256(test_path),
        },
        "raw_shapes": {
            "training": [int(raw_train.shape[0]), int(raw_train.shape[1] - 1)],
            "testing": [int(raw_test.shape[0]), int(raw_test.shape[1] - 1)],
        },
        "raw_class_counts": {
            "training": _counts(raw_train[TARGET_COLUMN]),
            "testing": _counts(raw_test[TARGET_COLUMN]),
        },
        "fixed_drops": list(DROP_COLUMNS),
        "drop_rationale": {
            "id": "record identifier; not a predictive network feature",
            "attack_cat": "attack-family annotation adjacent to the binary target; excluded to prevent target leakage",
        },
        "within_split_cleaning": {"training": train_report, "testing": test_report},
        "cross_split_decontamination": cross_report,
        "benchmark": {
            "seed": int(args.seed),
            "sizes": {"train": int(len(y_train)), "validation": int(len(y_val)), "test": int(len(y_test))},
            "class_counts": {"train": _counts(y_train), "validation": _counts(y_val), "test": _counts(y_test)},
            "official_source": {
                "train": "UNSW_NB15_training-set.csv",
                "validation": "UNSW_NB15_training-set.csv",
                "test": "UNSW_NB15_testing-set.csv",
            },
            "feature_overlap_between_benchmark_partitions": 0,
        },
        "methodological_guards": {
            "official_test_never_used_for_training_or_validation": True,
            "attack_cat_excluded_from_features": True,
            "id_excluded_from_features": True,
            "within_split_duplicate_pairs_removed_before_sampling": True,
            "contradictory_exact_feature_groups_removed_before_sampling": True,
            "official_test_rows_overlapping_training_features_removed_before_sampling": True,
            "preprocessing_fit_on_benchmark_train_only": True,
            "validation_transform_only": True,
            "test_transform_only": True,
        },
    }

    print("\n=== UNSW-NB15 EXTERNAL BENCHMARK PLAN ===")
    print("Raw training:", raw_train.shape[0], _counts(raw_train[TARGET_COLUMN]))
    print("Raw testing :", raw_test.shape[0], _counts(raw_test[TARGET_COLUMN]))
    print("Clean training:", len(train_clean), _counts(train_clean[TARGET_COLUMN]))
    print("Decontaminated testing:", len(test_clean), _counts(test_clean[TARGET_COLUMN]))
    print("Benchmark train:", len(y_train), _counts(y_train))
    print("Benchmark val  :", len(y_val), _counts(y_val))
    print("Benchmark test :", len(y_test), _counts(y_test))
    print("Features before encoding:", len(feature_cols))
    print("Qubits / PCA dimensions:", args.qubits)
    print("Test used for fitting: False")

    if args.dry_run:
        print("\nDry run complete. No preprocessing artifacts were written.")
        return

    if out.exists() and any(out.iterdir()) and not args.force:
        raise FileExistsError(f"Output directory is not empty: {out}. Use --force only if intentionally rebuilding.")
    out.mkdir(parents=True, exist_ok=True)

    preprocessor = QuantumTabularPreprocessor(n_qubits=int(args.qubits), random_state=int(args.seed))
    X_train_q = preprocessor.fit_transform(X_train)
    X_val_q = preprocessor.transform(X_val)
    X_test_q = preprocessor.transform(X_test)

    benchmark = {
        "X_train": X_train_q,
        "y_train": y_train.astype(np.int64),
        "X_val": X_val_q,
        "y_val": y_val.astype(np.int64),
        "X_test": X_test_q,
        "y_test": y_test.astype(np.int64),
    }
    save_quantum_splits(benchmark, out / "benchmark_splits.npz")
    preprocessor.save(out / "quantum_preprocessor.joblib")
    preprocessor.save_metadata(out / "quantum_preprocessor_metadata.json")

    report["quantum_preprocessing"] = preprocessor.metadata_.to_dict()
    report["quantum_shapes"] = {
        "train": list(X_train_q.shape),
        "validation": list(X_val_q.shape),
        "test": list(X_test_q.shape),
    }
    report["angle_range_observed"] = {
        "train": [float(np.min(X_train_q)), float(np.max(X_train_q))],
        "validation": [float(np.min(X_val_q)), float(np.max(X_val_q))],
        "test": [float(np.min(X_test_q)), float(np.max(X_test_q))],
    }

    provenance = []
    for split_name, frame in [("train", benchmark_train), ("validation", benchmark_val), ("test", benchmark_test)]:
        for i, row in frame[["__source_row__", "id", TARGET_COLUMN]].iterrows():
            provenance.append({
                "Benchmark Split": split_name,
                "Benchmark Index": int(i),
                "Official Source Row": int(row["__source_row__"]),
                "Official id": int(row["id"]),
                "Label": int(row[TARGET_COLUMN]),
            })
    pd.DataFrame(provenance).to_csv(out / "benchmark_provenance.csv", index=False)

    (out / "preprocessing_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("\n=== PREPROCESSING COMPLETE ===")
    print("Saved:", out / "benchmark_splits.npz")
    print("PCA explained variance sum:", preprocessor.metadata_.explained_variance_ratio_sum)
    print("Encoded feature count:", preprocessor.metadata_.encoded_feature_count)
    print("Quantum shapes:", report["quantum_shapes"])


if __name__ == "__main__":
    main()
