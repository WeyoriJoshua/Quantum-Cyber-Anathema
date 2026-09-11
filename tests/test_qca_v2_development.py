from pathlib import Path

import numpy as np
import torch

from models.weighted_vqc import balanced_class_weights, weighted_binary_cross_entropy
from qca.v2_loss import boundary_hardness, probability_margin_penalty
from run_qca_v2_development import _fixed_inner_split


def test_balanced_class_weights_are_inverse_frequency():
    y = np.array([0] * 80 + [1] * 20)
    w0, w1 = balanced_class_weights(y)
    assert np.isclose(w0, 100 / 160)
    assert np.isclose(w1, 100 / 40)
    assert w1 > w0


def test_weighted_bce_is_finite_and_differentiable():
    p = torch.tensor([0.2, 0.8], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([0.0, 1.0], dtype=torch.float64)
    loss = weighted_binary_cross_entropy(
        p,
        y,
        weight_negative=0.7,
        weight_positive=2.0,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert p.grad is not None
    assert torch.all(torch.isfinite(p.grad))


def test_boundary_hardness_peaks_at_half():
    p = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)
    h = boundary_hardness(p)
    expected = torch.tensor([0.0, 0.5, 1.0, 0.5, 0.0], dtype=torch.float64)
    assert torch.allclose(h, expected)


def test_margin_penalty_targets_class_separation():
    p = torch.tensor([0.20, 0.45, 0.55, 0.80], dtype=torch.float64)
    y = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float64)
    penalty = probability_margin_penalty(p, y, margin=0.15)
    assert np.isclose(float(penalty[0]), 0.0)
    assert np.isclose(float(penalty[3]), 0.0)
    assert float(penalty[1]) > 0.0
    assert float(penalty[2]) > 0.0


def test_fixed_inner_split_is_deterministic_and_stratified():
    X = np.arange(400).reshape(100, 4)
    y = np.array([0] * 80 + [1] * 20)
    tr1, va1 = _fixed_inner_split(X, y, fraction=0.20, split_seed=314159)
    tr2, va2 = _fixed_inner_split(X, y, fraction=0.20, split_seed=314159)
    assert np.array_equal(tr1, tr2)
    assert np.array_equal(va1, va2)
    assert len(tr1) == 80
    assert len(va1) == 20
    assert int(np.sum(y[va1] == 1)) == 4


def test_development_runner_has_no_test_array_access():
    source = Path("run_qca_v2_development.py").read_text(encoding="utf-8")
    assert 'data["X_test"]' not in source
    assert 'data["y_test"]' not in source
