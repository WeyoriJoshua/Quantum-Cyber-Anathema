from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from project_config import CONFIG
from experiments.common import create_matched_benchmark_splits, save_quantum_splits
from preprocessing.audit import audit_dataframe
from preprocessing.clean import clean_dataset
from preprocessing.loader import load_dataset
from preprocessing.quantum_pipeline import QuantumTabularPreprocessor
from preprocessing.split import stratified_train_val_test_split


def parse_args():
    parser = argparse.ArgumentParser(
        description="Leakage-safe preprocessing for Quantum Cyber Anathema."
    )
    parser.add_argument("--dataset", default=CONFIG.dataset_name)
    parser.add_argument("--path", default=str(CONFIG.raw_dataset_path))
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--qubits", type=int, default=CONFIG.n_qubits)
    parser.add_argument("--max-rows", type=int, default=CONFIG.max_rows)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--validation-size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=CONFIG.seed)
    parser.add_argument("--benchmark-train", type=int, default=CONFIG.benchmark_train)
    parser.add_argument("--benchmark-val", type=int, default=CONFIG.benchmark_val)
    parser.add_argument("--benchmark-test", type=int, default=CONFIG.benchmark_test)
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Keep exact feature+label duplicate rows (not recommended).",
    )
    parser.add_argument(
        "--keep-conflicts",
        action="store_true",
        help="Keep feature-identical rows with contradictory labels (not recommended).",
    )
    parser.add_argument("--out", default=str(CONFIG.preprocessed_dir))
    return parser.parse_args()


def _counts(y):
    values, counts = np.unique(np.asarray(y).astype(int), return_counts=True)
    return {str(int(k)): int(v) for k, v in zip(values, counts)}


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    X_raw, y_raw, load_meta = load_dataset(
        args.dataset,
        args.path,
        label_column=args.label_column,
        max_rows=args.max_rows,
        random_state=args.seed,
    )
    audit_before = audit_dataframe(X_raw)

    X, y, clean_meta = clean_dataset(
        X_raw,
        y_raw,
        dataset_drop_candidates=load_meta["always_drop_candidates"],
        drop_duplicate_pairs=not args.keep_duplicates,
        drop_conflicting_feature_rows=not args.keep_conflicts,
    )
    audit_after = audit_dataframe(X)
    class_counts_after = y.value_counts().sort_index().to_dict()
    if len(class_counts_after) != 2:
        raise ValueError("Cleaning left fewer than two classes.")

    splits = stratified_train_val_test_split(
        X,
        y,
        test_size=args.test_size,
        validation_size=args.validation_size,
        random_state=args.seed,
    )

    preprocessor = QuantumTabularPreprocessor(
        n_qubits=args.qubits,
        random_state=args.seed,
    )
    X_train_q = preprocessor.fit_transform(splits.X_train)
    X_val_q = preprocessor.transform(splits.X_val)
    X_test_q = preprocessor.transform(splits.X_test)

    full_data = {
        "X_train": X_train_q,
        "y_train": splits.y_train.to_numpy(dtype=np.int64),
        "X_val": X_val_q,
        "y_val": splits.y_val.to_numpy(dtype=np.int64),
        "X_test": X_test_q,
        "y_test": splits.y_test.to_numpy(dtype=np.int64),
    }
    save_quantum_splits(full_data, out / "quantum_splits.npz")

    benchmark = create_matched_benchmark_splits(
        full_data,
        train_size=args.benchmark_train,
        val_size=args.benchmark_val,
        test_size=args.benchmark_test,
        seed=args.seed,
    )
    save_quantum_splits(benchmark, out / "benchmark_splits.npz")

    preprocessor.save(out / "quantum_preprocessor.joblib")
    preprocessor.save_metadata(out / "quantum_preprocessor_metadata.json")

    report = {
        "load": load_meta,
        "audit_before": audit_before.to_dict(),
        "cleaning": clean_meta,
        "class_counts_after_cleaning": {str(k): int(v) for k, v in class_counts_after.items()},
        "audit_after": audit_after.to_dict(),
        "split_sizes": {
            "train": len(splits.y_train),
            "validation": len(splits.y_val),
            "test": len(splits.y_test),
        },
        "split_class_counts": {
            "train": _counts(full_data["y_train"]),
            "validation": _counts(full_data["y_val"]),
            "test": _counts(full_data["y_test"]),
        },
        "quantum_shape": {
            "train": list(X_train_q.shape),
            "validation": list(X_val_q.shape),
            "test": list(X_test_q.shape),
        },
        "matched_benchmark": {
            "path": str((out / "benchmark_splits.npz").resolve()),
            "sizes": {
                "train": len(benchmark["y_train"]),
                "validation": len(benchmark["y_val"]),
                "test": len(benchmark["y_test"]),
            },
            "class_counts": {
                "train": _counts(benchmark["y_train"]),
                "validation": _counts(benchmark["y_val"]),
                "test": _counts(benchmark["y_test"]),
            },
            "purpose": "Identical headline comparison partition for classical, VQC, QSVC and QCA models.",
        },
        "quantum_preprocessing": preprocessor.metadata_.to_dict(),
        "methodological_guards": {
            "fitted_preprocessing_training_only": True,
            "duplicates_controlled_before_split": not args.keep_duplicates,
            "conflicting_feature_labels_removed_before_split": not args.keep_conflicts,
            "test_partition_not_used_for_preprocessing_fit": True,
            "benchmark_sampled_within_existing_partitions": True,
        },
    }

    (out / "preprocessing_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, default=str))
    print(f"\nSaved preprocessing artifacts to: {out.resolve()}")
    print(f"Matched benchmark: {(out / 'benchmark_splits.npz').resolve()}")


if __name__ == "__main__":
    main()
