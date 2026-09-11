from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class DatasetSplits:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def stratified_train_val_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.20,
    validation_size: float = 0.20,
    random_state: int = 42,
) -> DatasetSplits:
    """Create stratified train/validation/test splits.

    `validation_size` is a fraction of the complete dataset, not the
    remaining training partition. Defaults produce 60/20/20.
    """
    if not (0 < test_size < 1 and 0 < validation_size < 1):
        raise ValueError("test_size and validation_size must be in (0,1).")
    if test_size + validation_size >= 1:
        raise ValueError("test_size + validation_size must be < 1.")

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    val_relative = validation_size / (1.0 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_relative,
        random_state=random_state,
        stratify=y_trainval,
    )

    return DatasetSplits(
        X_train=X_train.reset_index(drop=True),
        X_val=X_val.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_val=y_val.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
    )
