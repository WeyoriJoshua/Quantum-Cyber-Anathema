from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random

import numpy as np


@dataclass
class MemoryItem:
    priority: float
    order: int
    x: np.ndarray
    y: int
    error: float
    uncertainty: float
    cycle: int


class AdversarialMemory:
    """Fixed-capacity memory of unique hard/uncertain QCA samples.

    A sample is keyed by its float32 feature vector and label. Re-observing the
    same sample does not consume another memory slot; the higher-priority record
    is retained. Priority replay samples stochastically without replacement with
    probability proportional to priority, avoiding deterministic replay of only
    the same top-k examples on every cycle.
    """

    def __init__(self, capacity: int = 1000, seed: int = 42):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = int(capacity)
        self.seed = int(seed)
        self._items: dict[str, MemoryItem] = {}
        self._counter = 0
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._items)

    @staticmethod
    def _key(x: np.ndarray, y: int) -> str:
        arr = np.ascontiguousarray(np.asarray(x, dtype=np.float32).reshape(-1))
        h = hashlib.blake2b(digest_size=16)
        h.update(arr.tobytes())
        h.update(int(y).to_bytes(2, "little", signed=True))
        return h.hexdigest()

    def add(
        self,
        x,
        y,
        *,
        priority: float,
        error: float,
        uncertainty: float,
        cycle: int,
    ):
        x_arr = np.asarray(x, dtype=np.float32).reshape(-1).copy()
        y_int = int(y)
        priority = float(priority)
        if not np.isfinite(priority):
            return

        item = MemoryItem(
            priority=priority,
            order=self._counter,
            x=x_arr,
            y=y_int,
            error=float(error),
            uncertainty=float(uncertainty),
            cycle=int(cycle),
        )
        self._counter += 1
        key = self._key(x_arr, y_int)

        existing = self._items.get(key)
        if existing is not None:
            if item.priority > existing.priority:
                self._items[key] = item
            return

        if len(self._items) < self.capacity:
            self._items[key] = item
            return

        min_key, min_item = min(
            self._items.items(),
            key=lambda kv: (kv[1].priority, kv[1].order),
        )
        if (item.priority, item.order) > (min_item.priority, min_item.order):
            del self._items[min_key]
            self._items[key] = item

    def items(self, *, descending: bool = True) -> list[MemoryItem]:
        return sorted(
            self._items.values(),
            key=lambda item: (item.priority, item.order),
            reverse=descending,
        )

    def sample(
        self,
        n: int,
        *,
        prioritized: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        items = self.items(descending=True)
        if not items or n <= 0:
            return (
                np.empty((0, 0), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
            )

        n = min(int(n), len(items))
        if prioritized:
            priorities = np.asarray([max(i.priority, 0.0) for i in items], dtype=float)
            if priorities.sum() <= 0:
                indices = self._np_rng.choice(len(items), size=n, replace=False)
            else:
                probs = priorities + 1e-12
                probs /= probs.sum()
                indices = self._np_rng.choice(
                    len(items), size=n, replace=False, p=probs
                )
            chosen = [items[int(i)] for i in indices]
        else:
            chosen = self._rng.sample(items, n)

        X = np.stack([item.x for item in chosen]).astype(np.float32)
        y = np.asarray([item.y for item in chosen], dtype=np.int64)
        return X, y

    def summary(self) -> dict:
        items = self.items()
        if not items:
            return {
                "size": 0,
                "capacity": self.capacity,
                "mean_priority": 0.0,
                "mean_error": 0.0,
                "mean_uncertainty": 0.0,
                "malicious_fraction": 0.0,
                "cycles_present": [],
            }
        return {
            "size": len(items),
            "capacity": self.capacity,
            "mean_priority": float(np.mean([i.priority for i in items])),
            "mean_error": float(np.mean([i.error for i in items])),
            "mean_uncertainty": float(np.mean([i.uncertainty for i in items])),
            "malicious_fraction": float(np.mean([i.y for i in items])),
            "cycles_present": sorted(set(i.cycle for i in items)),
        }
