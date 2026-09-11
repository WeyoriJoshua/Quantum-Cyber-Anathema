from __future__ import annotations

import importlib
import platform
import sys

from project_config import CONFIG


PACKAGES = [
    ("numpy", "NumPy"),
    ("pandas", "Pandas"),
    ("scipy", "SciPy"),
    ("sklearn", "scikit-learn"),
    ("xgboost", "XGBoost"),
    ("torch", "PyTorch"),
    ("pennylane", "PennyLane"),
    ("qiskit", "Qiskit"),
    ("qiskit_machine_learning", "Qiskit Machine Learning"),
]


def main():
    print("Python executable:", sys.executable)
    print("Python version:", sys.version.split()[0])
    print("Platform:", platform.platform())
    print("Project root:", CONFIG.project_root)
    print("Raw dataset path:", CONFIG.raw_dataset_path)
    print("Raw dataset exists:", CONFIG.raw_dataset_path.exists())
    print("Quantum splits:", CONFIG.quantum_splits_path)
    print("Quantum splits exist:", CONFIG.quantum_splits_path.exists())
    print()

    failures = []
    for module_name, display_name in PACKAGES:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "unknown")
            print(f"[OK] {display_name}: {version}")
        except Exception as exc:
            failures.append((display_name, str(exc)))
            print(f"[FAIL] {display_name}: {exc}")

    try:
        import torch
        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("CUDA device:", torch.cuda.get_device_name(0))
    except Exception:
        pass

    if failures:
        raise SystemExit("Environment check failed; install requirements-quantum.txt.")

    print("\nENVIRONMENT CHECK PASSED")


if __name__ == "__main__":
    main()
