import numpy as np

from evaluation.metrics import select_threshold
from experiments.common import create_matched_benchmark_splits


def test_select_threshold_returns_valid_value():
    y = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.4, 0.6, 0.9])
    threshold, best, table = select_threshold(y, score, metric="mcc")
    assert 0.05 <= threshold <= 0.95
    assert best <= 1.0
    assert len(table) == 181


def test_matched_benchmark_sizes_and_classes():
    rng = np.random.default_rng(42)
    def part(n):
        X = rng.normal(size=(n, 4))
        y = np.array([0] * int(n * 0.8) + [1] * (n - int(n * 0.8)))
        return X, y
    Xtr,ytr=part(1000); Xv,yv=part(400); Xt,yt=part(400)
    data={"X_train":Xtr,"y_train":ytr,"X_val":Xv,"y_val":yv,"X_test":Xt,"y_test":yt}
    b=create_matched_benchmark_splits(data,train_size=600,val_size=200,test_size=200,seed=42)
    assert b["X_train"].shape==(600,4)
    assert b["X_val"].shape==(200,4)
    assert b["X_test"].shape==(200,4)
    assert set(np.unique(b["y_train"]))=={0,1}
    assert set(np.unique(b["y_val"]))=={0,1}
    assert set(np.unique(b["y_test"]))=={0,1}
