from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable
import re

import numpy as np
import pandas as pd


IDENTIFIER_PATTERNS = (
    r"^id$",
    r".*_id$",
    r"^flow id$",
    r"^flowid$",
    r"^src ip$",
    r"^source ip$",
    r"^dst ip$",
    r"^destination ip$",
    r"^timestamp$",
    r"^time stamp$",
)

TARGET_ADJACENT_PATTERNS = (
    r"attack[_ ]?cat",
    r"attack[_ ]?category",
    r"subcategory",
    r"attack[_ ]?type",
    r"^type$",
)


@dataclass
class LeakageAudit:
    rows: int
    columns: int
    duplicate_rows: int
    infinite_cells: int
    missing_cells: int
    constant_columns: list[str]
    all_missing_columns: list[str]
    identifier_candidates: list[str]
    target_adjacent_candidates: list[str]
    high_cardinality_object_columns: list[str]
    near_unique_columns: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _matches_any(column: str, patterns: Iterable[str]) -> bool:
    text = str(column).strip().lower()
    return any(re.search(pattern, text, flags=re.I) for pattern in patterns)


def audit_dataframe(
    X: pd.DataFrame,
    *,
    high_cardinality_threshold: int = 100,
    near_unique_ratio: float = 0.98,
) -> LeakageAudit:
    work = X.copy()
    numeric = work.select_dtypes(include=[np.number])

    if numeric.shape[1]:
        infinite_cells = int(np.isinf(numeric.to_numpy(dtype=float, copy=False)).sum())
    else:
        infinite_cells = 0

    nunique = work.nunique(dropna=False)
    constant_columns = nunique[nunique <= 1].index.astype(str).tolist()
    all_missing_columns = work.columns[work.isna().all()].astype(str).tolist()

    identifier_candidates = [
        str(c) for c in work.columns if _matches_any(str(c), IDENTIFIER_PATTERNS)
    ]
    target_adjacent_candidates = [
        str(c) for c in work.columns if _matches_any(str(c), TARGET_ADJACENT_PATTERNS)
    ]

    object_cols = work.select_dtypes(include=["object", "string", "category"]).columns
    high_cardinality = [
        str(c)
        for c in object_cols
        if work[c].nunique(dropna=True) > high_cardinality_threshold
    ]

    n = max(len(work), 1)
    near_unique = [
        str(c)
        for c in work.columns
        if work[c].nunique(dropna=True) / n >= near_unique_ratio
    ]

    return LeakageAudit(
        rows=int(len(work)),
        columns=int(work.shape[1]),
        duplicate_rows=int(work.duplicated().sum()),
        infinite_cells=infinite_cells,
        missing_cells=int(work.isna().sum().sum()),
        constant_columns=constant_columns,
        all_missing_columns=all_missing_columns,
        identifier_candidates=identifier_candidates,
        target_adjacent_candidates=target_adjacent_candidates,
        high_cardinality_object_columns=high_cardinality,
        near_unique_columns=near_unique,
    )


def recommended_drop_columns(
    X: pd.DataFrame,
    *,
    dataset_drop_candidates: Iterable[str] = (),
) -> list[str]:
    """
    Conservative auto-drop list:
      - obvious identifiers
      - target-adjacent attack-family/type columns
      - constant/all-missing columns
      - dataset-specific known candidates if present

    High-cardinality features are *reported* but are not silently dropped.
    """
    audit = audit_dataframe(X)
    lower_to_actual = {str(c).strip().lower(): str(c) for c in X.columns}

    dataset_present = []
    for candidate in dataset_drop_candidates:
        actual = lower_to_actual.get(str(candidate).strip().lower())
        if actual is not None:
            dataset_present.append(actual)

    columns = (
        audit.constant_columns
        + audit.all_missing_columns
        + audit.identifier_candidates
        + audit.target_adjacent_candidates
        + dataset_present
    )
    return sorted(set(c for c in columns if c in X.columns))
