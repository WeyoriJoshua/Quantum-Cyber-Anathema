import numpy as np
import pytest

from evaluation.metrics import classification_metrics
from evaluation.evolution import (
    composite_defense_score,
    defensive_intelligence_gain,
    recursive_evolution_index,
    empirical_qecr_proxy,
)
from experiments.common import stratified_subsample_arrays
from qca.memory import AdversarialMemory


def test_metrics_and_evolution_proxies():
    y = np.array([0, 0, 1, 1, 1, 0])
    p0 = np.array([0.4, 0.3, 0.6, 0.55, 0.51, 0.45])
    p1 = np.array([0.1, 0.2, 0.9, 0.8, 0.85, 0.2])

    m0 = classification_metrics(y, (p0 >= 0.5).astype(int), p0)
    m1 = classification_metrics(y, (p1 >= 0.5).astype(int), p1)

    s0 = composite_defense_score(m0)
    s1 = composite_defense_score(m1)

    assert s1 >= s0
    assert defensive_intelligence_gain(s1, s0) >= 0
    assert recursive_evolution_index(s1, s0) >= 0
    assert empirical_qecr_proxy(p0, p1) > 0


def test_adversarial_memory_keeps_high_priority_unique_samples():
    mem = AdversarialMemory(capacity=3, seed=42)
    for i, priority in enumerate([0.1, 0.9, 0.4, 1.2, 0.2]):
        mem.add(
            np.array([i, i+1], dtype=float),
            i % 2,
            priority=priority,
            error=priority / 2,
            uncertainty=priority / 3,
            cycle=1,
        )

    top = mem.items()[0]
    mem.add(
        top.x,
        top.y,
        priority=top.priority - 0.1,
        error=0.1,
        uncertainty=0.1,
        cycle=2,
    )

    priorities = [item.priority for item in mem.items()]
    assert len(mem) == 3
    assert priorities == sorted(priorities, reverse=True)
    assert min(priorities) >= 0.4


def test_stratified_subsample_preserves_minority():
    X = np.arange(200).reshape(100, 2)
    y = np.array([0] * 90 + [1] * 10)
    Xs, ys = stratified_subsample_arrays(X, y, 20, seed=42)
    assert Xs.shape == (20, 2)
    assert np.sum(ys == 0) == 18
    assert np.sum(ys == 1) == 2


def test_qca_loss_if_torch_available():
    torch = pytest.importorskip("torch")
    from qca.loss import qca_reflexive_loss

    p = torch.tensor([0.1, 0.8, 0.55], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([0.0, 1.0, 1.0], dtype=torch.float64)

    loss, error, uncertainty, importance, weights = qca_reflexive_loss(p, y)
    assert torch.isfinite(loss)
    assert (weights >= 1.0).all()
    loss.backward()
    assert p.grad is not None


def test_qca_trainer_protocol_with_small_torch_model():
    torch = pytest.importorskip("torch")
    from qca.trainer import fit_qca

    class TinyProbModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 1, dtype=torch.float64)

        def forward(self, x):
            return torch.sigmoid(self.linear(x.to(torch.float64))).squeeze(-1).clamp(1e-7, 1-1e-7)

    rng = np.random.default_rng(0)
    X0 = rng.normal(-1, 0.3, size=(30, 2))
    X1 = rng.normal(1, 0.3, size=(30, 2))
    X = np.vstack([X0, X1])
    y = np.array([0]*30 + [1]*30)
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]

    model = TinyProbModel()
    history, memory = fit_qca(
        model,
        X[:40], y[:40],
        X[40:], y[40:],
        warmup_epochs=2,
        warmup_patience=2,
        cycles=1,
        epochs_per_cycle=1,
        batch_size=8,
        memory_capacity=10,
        replay_ratio=0.2,
        restore_best=True,
        seed=42,
        verbose=False,
    )

    assert history["Cycle"].tolist() == [0, 1]
    assert history["Selected for Test"].sum() == 1
    assert hasattr(model, "qca_best_cycle_")
    assert len(memory) <= 10
