from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal
import json
import math

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None

try:
    import pennylane as qml
except Exception:
    qml = None


def pennylane_available() -> bool:
    return qml is not None and torch is not None


if nn is not None:
    _BaseModule = nn.Module
else:
    class _BaseModule:  # pragma: no cover - dependency guard
        pass


class PennyLaneVQC(_BaseModule):
    """
    Standard variational quantum classifier baseline.

    Supported encodings:
      - angle
      - data_reuploading

    The QCA model subclasses/reuses this quantum core.
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int = 3,
        *,
        encoding: Literal["angle", "data_reuploading"] = "angle",
        device_name: str = "default.qubit",
        shots: int | None = None,
        seed: int = 42,
    ):
        if not pennylane_available():
            raise ImportError(
                "PennyLane + PyTorch are required. "
                "Install requirements-quantum.txt."
            )
        super().__init__()

        if n_qubits < 1:
            raise ValueError("n_qubits must be >=1")
        if n_layers < 1:
            raise ValueError("n_layers must be >=1")
        if encoding not in {"angle", "data_reuploading"}:
            raise ValueError("encoding must be 'angle' or 'data_reuploading'")

        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.encoding = encoding
        self.device_name = device_name
        self.shots = shots
        self.seed = seed

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.dev = qml.device(
            device_name,
            wires=self.n_qubits,
            shots=shots,
        )

        @qml.qnode(
            self.dev,
            interface="torch",
            diff_method="backprop" if shots is None and device_name == "default.qubit" else "parameter-shift",
        )
        def circuit(inputs, weights):
            if self.encoding == "angle":
                qml.AngleEmbedding(
                    inputs,
                    wires=range(self.n_qubits),
                    rotation="Y",
                )
                qml.StronglyEntanglingLayers(
                    weights,
                    wires=range(self.n_qubits),
                )
            else:
                for layer in range(self.n_layers):
                    qml.AngleEmbedding(
                        inputs,
                        wires=range(self.n_qubits),
                        rotation="Y",
                    )
                    qml.StronglyEntanglingLayers(
                        weights[layer:layer+1],
                        wires=range(self.n_qubits),
                    )

            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

        init = 0.01 * torch.randn(
            self.n_layers,
            self.n_qubits,
            3,
            dtype=torch.float64,
        )
        self.weights = nn.Parameter(init)

    def forward(self, x):
        if x.ndim == 1:
            x = x.unsqueeze(0)

        outputs = []
        x = x.to(dtype=torch.float64)
        for sample in x:
            outputs.append(self.circuit(sample, self.weights))

        z = torch.stack(outputs).reshape(-1)
        probability = (1.0 - z) / 2.0
        return torch.clamp(probability, 1e-7, 1 - 1e-7)


@dataclass
class TrainingHistory:
    epoch: list[int]
    train_loss: list[float]
    val_loss: list[float]

    def to_dict(self):
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
        }


def binary_cross_entropy(probabilities, targets, epsilon: float = 1e-7):
    p = torch.clamp(probabilities, epsilon, 1.0 - epsilon)
    y = targets.to(dtype=p.dtype)
    return -(y * torch.log(p) + (1-y) * torch.log(1-p)).mean()


def _loader(X, y, batch_size: int, shuffle: bool):
    x = torch.as_tensor(np.asarray(X), dtype=torch.float64)
    target = torch.as_tensor(np.asarray(y), dtype=torch.float64)
    ds = TensorDataset(x, target)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def evaluate_loss(model, X, y, batch_size=64):
    model.eval()
    losses = []
    with torch.no_grad():
        for xb, yb in _loader(X, y, batch_size, shuffle=False):
            p = model(xb)
            losses.append(float(binary_cross_entropy(p, yb).item()))
    return float(np.mean(losses)) if losses else float("nan")


def fit_vqc(
    model: PennyLaneVQC,
    X_train,
    y_train,
    X_val,
    y_val,
    *,
    epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 0.01,
    patience: int = 5,
    checkpoint_path: str | Path | None = None,
    verbose: bool = True,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = TrainingHistory([], [], [])
    best_val = float("inf")
    best_state = None
    stale = 0

    train_loader = _loader(X_train, y_train, batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []

        for xb, yb in train_loader:
            optimizer.zero_grad()
            p = model(xb)
            loss = binary_cross_entropy(p, yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        train_loss = float(np.mean(epoch_losses))
        val_loss = evaluate_loss(model, X_val, y_val, batch_size=batch_size)

        history.epoch.append(epoch)
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)

        if verbose:
            print(
                f"Epoch {epoch:03d} | "
                f"train={train_loss:.6f} | val={val_loss:.6f}"
            )

        if val_loss < best_val - 1e-8:
            best_val = val_loss
            stale = 0
            best_state = {
                k: v.detach().clone()
                for k, v in model.state_dict().items()
            }
            if checkpoint_path is not None:
                checkpoint_path = Path(checkpoint_path)
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, checkpoint_path)
        else:
            stale += 1
            if stale >= patience:
                if verbose:
                    print(f"Early stopping after epoch {epoch}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


def predict_vqc(model: PennyLaneVQC, X, *, batch_size: int = 64) -> np.ndarray:
    model.eval()
    preds = []
    dummy_y = np.zeros(len(X), dtype=np.float64)
    with torch.no_grad():
        for xb, _ in _loader(X, dummy_y, batch_size, shuffle=False):
            preds.append(model(xb).detach().cpu().numpy())
    return np.concatenate(preds).astype(float)
