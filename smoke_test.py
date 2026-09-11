import math
import numpy as np
import pandas as pd

from preprocessing.clean import clean_dataset
from preprocessing.split import stratified_train_val_test_split
from preprocessing.quantum_pipeline import QuantumTabularPreprocessor

rng = np.random.default_rng(42)
n = 200
X = pd.DataFrame({
    "Flow ID": [f"flow-{i}" for i in range(n)],
    "duration": rng.gamma(2, 2, n),
    "bytes": rng.lognormal(4, 1, n),
    "protocol": rng.choice(["TCP", "UDP", "ICMP"], n),
})
y = pd.Series(np.tile([0, 1], n // 2))

X, y, _ = clean_dataset(X, y, dataset_drop_candidates=("Flow ID",))
splits = stratified_train_val_test_split(X, y)
pre = QuantumTabularPreprocessor(n_qubits=4)
Xq = pre.fit_transform(splits.X_train)

assert Xq.shape == (120, 4)
assert np.isfinite(Xq).all()
assert Xq.min() >= -1e-7
assert Xq.max() <= math.pi + 1e-7

print("SMOKE TEST PASSED")
print("Quantum train shape:", Xq.shape)
print("Metadata:", pre.metadata_.to_dict())
