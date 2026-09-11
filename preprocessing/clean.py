from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .audit import recommended_drop_columns


def clean_features(
    X: pd.DataFrame,
    *,
    dataset_drop_candidates: Iterable[str] = (),
) -> tuple[pd.DataFrame, dict]:
    """Deterministic feature cleanup that does not fit data-dependent transforms.

    This stage removes provenance/identifier/target-adjacent/constant columns and
    converts +/- infinity to missing values. Imputation, scaling, encoding and PCA
    are intentionally deferred until *after* train/validation/test splitting.

    Duplicate handling is deliberately excluded here because dropping rows from X
    alone would desynchronise X and y. Use :func:`clean_dataset` when labels exist.
    """
    out = X.copy()

    if "__source_file__" in out.columns:
        out = out.drop(columns=["__source_file__"])

    numeric_cols = out.select_dtypes(include=[np.number]).columns
    if len(numeric_cols):
        out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)

    drops = recommended_drop_columns(
        out,
        dataset_drop_candidates=dataset_drop_candidates,
    )
    out = out.drop(columns=drops, errors="ignore")
    out = out.reset_index(drop=True)

    report = {
        "dropped_columns": drops,
        "rows": int(len(out)),
        "remaining_columns": int(out.shape[1]),
    }
    return out, report


def _feature_hashes(X: pd.DataFrame) -> pd.Series:
    """Stable row hashes used only to identify exact duplicate feature rows."""
    return pd.util.hash_pandas_object(X, index=False).astype("uint64")


def clean_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    dataset_drop_candidates: Iterable[str] = (),
    drop_duplicate_pairs: bool = True,
    drop_conflicting_feature_rows: bool = True,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Clean features while preserving exact X/y row alignment.

    Publication-safety rules:
    * identical feature+label duplicates can be removed before splitting so the
      same observation cannot appear in both train and test partitions;
    * feature-identical rows carrying contradictory labels are identified and,
      by default, all rows in those conflicting groups are removed rather than
      silently choosing one label.

    These operations happen before the split and do not fit any learned
    preprocessing statistics.
    """
    X_clean, feature_report = clean_features(
        X,
        dataset_drop_candidates=dataset_drop_candidates,
    )
    y_clean = pd.Series(y).reset_index(drop=True).astype(int)

    if len(X_clean) != len(y_clean):
        raise ValueError(
            f"X/y length mismatch after cleanup: {len(X_clean)} != {len(y_clean)}"
        )

    before = len(X_clean)
    hashes = _feature_hashes(X_clean)
    label_frame = pd.DataFrame({"__hash__": hashes, "__target__": y_clean})

    target_counts = label_frame.groupby("__hash__", sort=False)["__target__"].nunique()
    candidate_conflict_hashes = set(target_counts[target_counts > 1].index.tolist())

    conflicting_mask = pd.Series(False, index=X_clean.index)
    if candidate_conflict_hashes:
        candidate_idx = hashes[hashes.isin(candidate_conflict_hashes)].index
        for h in candidate_conflict_hashes:
            idx = hashes[hashes == h].index
            bucket = X_clean.loc[idx]
            joined = bucket.copy()
            joined["__target__"] = y_clean.loc[idx].to_numpy()
            feature_cols = list(bucket.columns)
            counts = joined.groupby(feature_cols, dropna=False)["__target__"].nunique()
            bad_keys = counts[counts > 1]
            if len(bad_keys):
                for row_idx in idx:
                    row = X_clean.loc[row_idx]
                    same = pd.Series(True, index=idx)
                    for col in feature_cols:
                        value = row[col]
                        if pd.isna(value):
                            same &= X_clean.loc[idx, col].isna()
                        else:
                            same &= X_clean.loc[idx, col].eq(value)
                    exact_idx = same[same].index
                    if y_clean.loc[exact_idx].nunique() > 1:
                        conflicting_mask.loc[exact_idx] = True

    conflicting_rows = int(conflicting_mask.sum())
    conflicting_groups = 0
    if conflicting_rows:
        conflicting_groups = int(
            pd.DataFrame(
                {
                    "h": _feature_hashes(X_clean.loc[conflicting_mask]),
                    "y": y_clean.loc[conflicting_mask].to_numpy(),
                }
            )["h"].nunique()
        )

    if drop_conflicting_feature_rows and conflicting_rows:
        keep = ~conflicting_mask
        X_clean = X_clean.loc[keep].reset_index(drop=True)
        y_clean = y_clean.loc[keep].reset_index(drop=True)

    pair_duplicates_removed = 0
    if drop_duplicate_pairs:
        combined = X_clean.copy()
        combined["__target__"] = y_clean.to_numpy()
        duplicate_pair_mask = combined.duplicated(keep="first")
        pair_duplicates_removed = int(duplicate_pair_mask.sum())
        if pair_duplicates_removed:
            keep = ~duplicate_pair_mask
            X_clean = X_clean.loc[keep].reset_index(drop=True)
            y_clean = y_clean.loc[keep].reset_index(drop=True)

    report = {
        **feature_report,
        "rows_before_duplicate_control": int(before),
        "conflicting_feature_rows_detected": conflicting_rows,
        "conflicting_feature_groups_detected": conflicting_groups,
        "conflicting_feature_rows_removed": (
            conflicting_rows if drop_conflicting_feature_rows else 0
        ),
        "duplicate_feature_label_pairs_removed": pair_duplicates_removed,
        "rows_after_duplicate_control": int(len(X_clean)),
        "drop_duplicate_pairs": bool(drop_duplicate_pairs),
        "drop_conflicting_feature_rows": bool(drop_conflicting_feature_rows),
    }
    return X_clean, y_clean, report
