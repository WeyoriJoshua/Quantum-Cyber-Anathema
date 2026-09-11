import math

import numpy as np
import pandas as pd

from preprocessing.audit import audit_dataframe
from preprocessing.clean import clean_dataset, clean_features
from preprocessing.loader import load_dataset
from preprocessing.quantum_pipeline import QuantumTabularPreprocessor
from preprocessing.split import stratified_train_val_test_split


def make_synthetic(n=200, seed=7):
    rng = np.random.default_rng(seed)
    y = pd.Series(np.tile([0, 1], n // 2), name="label")
    X = pd.DataFrame(
        {
            "Flow ID": [f"flow-{i}" for i in range(n)],
            "Src IP": [f"10.0.0.{i % 255}" for i in range(n)],
            "Duration": rng.gamma(2.0, 3.0, size=n),
            "Bytes": rng.lognormal(5.0, 1.0, size=n),
            "Protocol": rng.choice(["TCP", "UDP", "ICMP"], size=n),
            "Service": rng.choice(["http", "dns", "ssh"], size=n),
            "constant": 1,
        }
    )
    X.loc[0, "Duration"] = np.inf
    X.loc[1, "Bytes"] = np.nan
    return X, y


def test_quantum_pipeline_is_leakage_safe_shape_and_range():
    X, y = make_synthetic()
    X, _ = clean_features(
        X,
        dataset_drop_candidates=("Flow ID", "Src IP"),
    )

    splits = stratified_train_val_test_split(
        X, y, test_size=0.2, validation_size=0.2, random_state=42
    )

    pre = QuantumTabularPreprocessor(n_qubits=4, random_state=42)
    train = pre.fit_transform(splits.X_train)
    val = pre.transform(splits.X_val)
    test = pre.transform(splits.X_test)

    assert train.shape[1] == 4
    assert val.shape[1] == 4
    assert test.shape[1] == 4
    assert np.isfinite(train).all()
    assert np.isfinite(val).all()
    assert np.isfinite(test).all()
    assert train.min() >= -1e-6
    assert train.max() <= math.pi + 1e-6


def test_audit_detects_identifier_and_constant():
    X, _ = make_synthetic()
    audit = audit_dataframe(X)
    assert "Flow ID" in audit.identifier_candidates
    assert "constant" in audit.constant_columns
    assert audit.infinite_cells >= 1


def test_clean_dataset_controls_duplicates_and_conflicting_labels():
    X = pd.DataFrame({
        "a": [1, 1, 2, 2, 3],
        "b": [5, 5, 6, 6, 7],
    })
    y = pd.Series([0, 0, 0, 1, 1])
    Xc, yc, report = clean_dataset(X, y)

    assert report["duplicate_feature_label_pairs_removed"] == 1
    assert report["conflicting_feature_rows_detected"] == 2
    assert len(Xc) == len(yc) == 2
    assert set(yc.tolist()) == {0, 1}


def test_loader_max_rows_is_stratified(tmp_path):
    n0, n1 = 90, 10
    df = pd.DataFrame({
        "Flow Duration": np.arange(n0+n1),
        "Label": ["BENIGN"] * n0 + ["DDoS"] * n1,
    })
    path = tmp_path / "cic.csv"
    df.to_csv(path, index=False)

    _, y, meta = load_dataset("CICIDS2017", path, max_rows=20, random_state=42)
    counts = y.value_counts().sort_index().to_dict()
    assert len(y) == 20
    assert counts == {0: 18, 1: 2}
    assert meta["stratified_subsample_applied"] is True
