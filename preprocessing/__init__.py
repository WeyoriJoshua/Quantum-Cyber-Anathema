"""Leakage-safe preprocessing utilities for Quantum Cyber Anathema (QCA)."""

from .loader import (
    DATASET_SPECS,
    DatasetSpec,
    load_dataset,
    standardize_binary_label,
)
from .audit import LeakageAudit, audit_dataframe
from .clean import clean_features, clean_dataset
from .split import stratified_train_val_test_split
from .quantum_pipeline import QuantumTabularPreprocessor

__all__ = [
    "DATASET_SPECS",
    "DatasetSpec",
    "load_dataset",
    "standardize_binary_label",
    "LeakageAudit",
    "audit_dataframe",
    "clean_features",
    "clean_dataset",
    "stratified_train_val_test_split",
    "QuantumTabularPreprocessor",
]
