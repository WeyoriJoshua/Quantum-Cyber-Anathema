from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import math

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler


@dataclass
class QuantumPreprocessingMetadata:
    n_qubits: int
    n_numeric_columns: int
    n_categorical_columns: int
    encoded_feature_count: int
    pca_components_used: int
    padded_components: int
    explained_variance_ratio_sum: float
    angle_range: tuple[float, float]
    numeric_columns: list[str]
    categorical_columns: list[str]

    def to_dict(self) -> dict:
        return {
            "n_qubits": self.n_qubits,
            "n_numeric_columns": self.n_numeric_columns,
            "n_categorical_columns": self.n_categorical_columns,
            "encoded_feature_count": self.encoded_feature_count,
            "pca_components_used": self.pca_components_used,
            "padded_components": self.padded_components,
            "explained_variance_ratio_sum": self.explained_variance_ratio_sum,
            "angle_range": list(self.angle_range),
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
        }


class QuantumTabularPreprocessor:
    """Leakage-safe tabular-to-q-qubit feature transformer."""

    def __init__(
        self,
        n_qubits: int = 6,
        *,
        angle_min: float = 0.0,
        angle_max: float = math.pi,
        min_category_frequency: int | float | None = None,
        random_state: int = 42,
    ):
        if n_qubits < 1:
            raise ValueError("n_qubits must be >= 1.")
        if not float(angle_max) > float(angle_min):
            raise ValueError("angle_max must be greater than angle_min.")
        self.n_qubits = int(n_qubits)
        self.angle_min = float(angle_min)
        self.angle_max = float(angle_max)
        self.min_category_frequency = min_category_frequency
        self.random_state = int(random_state)
        self.column_transformer_: ColumnTransformer | None = None
        self.pca_: PCA | None = None
        self.angle_scaler_: MinMaxScaler | None = None
        self.metadata_: QuantumPreprocessingMetadata | None = None
        self._fitted = False

    def _build_column_transformer(self, X: pd.DataFrame) -> ColumnTransformer:
        numeric_columns = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
        categorical_columns = [c for c in X.columns if c not in numeric_columns]

        numeric_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        onehot_kwargs: dict[str, Any] = {
            "handle_unknown": "ignore",
            "sparse_output": False,
        }
        if self.min_category_frequency is not None:
            onehot_kwargs["min_frequency"] = self.min_category_frequency

        categorical_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(**onehot_kwargs)),
            ]
        )

        transformers = []
        if numeric_columns:
            transformers.append(("numeric", numeric_pipe, numeric_columns))
        if categorical_columns:
            transformers.append(("categorical", categorical_pipe, categorical_columns))
        if not transformers:
            raise ValueError("No usable feature columns remain after cleaning.")

        return ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            sparse_threshold=0.0,
            verbose_feature_names_out=False,
        )

    @staticmethod
    def _as_dense_float64(X) -> np.ndarray:
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"Expected a 2D feature matrix, got shape={X.shape}.")
        return X

    def _pad(self, Z: np.ndarray) -> np.ndarray:
        if Z.shape[1] > self.n_qubits:
            raise RuntimeError("Internal error: PCA returned more components than qubits.")
        if Z.shape[1] == self.n_qubits:
            return Z
        pad_width = self.n_qubits - Z.shape[1]
        return np.pad(Z, ((0, 0), (0, pad_width)), mode="constant", constant_values=0.0)

    def fit(self, X_train: pd.DataFrame):
        X_train = X_train.copy()
        self.column_transformer_ = self._build_column_transformer(X_train)
        encoded = self._as_dense_float64(self.column_transformer_.fit_transform(X_train))

        max_components = min(
            self.n_qubits,
            encoded.shape[1],
            max(encoded.shape[0] - 1, 1),
        )
        if max_components < 1:
            raise ValueError("Insufficient rank/samples for PCA.")

        self.pca_ = PCA(n_components=max_components, random_state=self.random_state)
        Z = self.pca_.fit_transform(encoded)
        Zq = self._pad(Z)

        self.angle_scaler_ = MinMaxScaler(feature_range=(self.angle_min, self.angle_max))
        self.angle_scaler_.fit(Zq)

        numeric_columns = X_train.select_dtypes(include=[np.number, "bool"]).columns.astype(str).tolist()
        categorical_columns = [str(c) for c in X_train.columns if str(c) not in numeric_columns]

        self.metadata_ = QuantumPreprocessingMetadata(
            n_qubits=self.n_qubits,
            n_numeric_columns=len(numeric_columns),
            n_categorical_columns=len(categorical_columns),
            encoded_feature_count=int(encoded.shape[1]),
            pca_components_used=int(max_components),
            padded_components=int(self.n_qubits - max_components),
            explained_variance_ratio_sum=float(np.sum(self.pca_.explained_variance_ratio_)),
            angle_range=(self.angle_min, self.angle_max),
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
        )
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() on training data before transform().")
        encoded = self._as_dense_float64(self.column_transformer_.transform(X))
        Z = self.pca_.transform(encoded)
        Zq = self._pad(Z)
        Xq = self.angle_scaler_.transform(Zq)
        Xq = np.clip(Xq, self.angle_min, self.angle_max)
        return Xq.astype(np.float32)

    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        return self.fit(X_train).transform(X_train)

    def save(self, path: str | Path) -> Path:
        if not self._fitted:
            raise RuntimeError("Cannot save an unfitted preprocessor.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "QuantumTabularPreprocessor":
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Object in {path} is not a {cls.__name__}.")
        return obj

    def save_metadata(self, path: str | Path) -> Path:
        if self.metadata_ is None:
            raise RuntimeError("No metadata available before fit().")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.metadata_.to_dict(), indent=2), encoding="utf-8")
        return path
