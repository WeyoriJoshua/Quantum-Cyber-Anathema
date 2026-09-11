from __future__ import annotations

from typing import Literal


def qiskit_ml_available() -> bool:
    try:
        import qiskit  # noqa: F401
        import qiskit_machine_learning  # noqa: F401
        return True
    except Exception:
        return False


def _feature_map(
    kind: Literal["z", "zz"],
    n_features: int,
    reps: int = 2,
):
    try:
        from qiskit.circuit.library import z_feature_map, zz_feature_map
        if kind == "z":
            return z_feature_map(n_features, reps=reps)
        if kind == "zz":
            return zz_feature_map(n_features, reps=reps, entanglement="linear")
    except Exception:
        from qiskit.circuit.library import ZFeatureMap, ZZFeatureMap
        if kind == "z":
            return ZFeatureMap(feature_dimension=n_features, reps=reps)
        if kind == "zz":
            return ZZFeatureMap(feature_dimension=n_features, reps=reps, entanglement="linear")
    raise ValueError("kind must be 'z' or 'zz'")


def _ansatz(n_qubits: int, reps: int = 2):
    try:
        from qiskit.circuit.library import real_amplitudes
        return real_amplitudes(n_qubits, reps=reps, entanglement="linear")
    except Exception:
        from qiskit.circuit.library import RealAmplitudes
        return RealAmplitudes(num_qubits=n_qubits, reps=reps, entanglement="linear")


def _optimizer(maxiter: int):
    try:
        from qiskit_machine_learning.optimizers import COBYLA
    except Exception:  # pragma: no cover
        from qiskit_algorithms.optimizers import COBYLA
    return COBYLA(maxiter=maxiter)


def _exact_sampler(seed: int = 42):
    try:
        from qiskit_machine_learning.primitives import QMLSampler
        return QMLSampler(shots=None, seed=seed)
    except Exception:
        try:
            from qiskit.primitives import StatevectorSampler
            return StatevectorSampler(seed=seed)
        except Exception:
            return None


def build_qiskit_vqc(
    n_features: int,
    *,
    feature_map: Literal["z", "zz"] = "zz",
    feature_reps: int = 2,
    ansatz_reps: int = 2,
    maxiter: int = 100,
    seed: int = 42,
):
    if not qiskit_ml_available():
        raise ImportError(
            "Qiskit Machine Learning is required. "
            "Install requirements-quantum.txt."
        )

    from qiskit_machine_learning.algorithms import VQC

    fmap = _feature_map(feature_map, n_features, reps=feature_reps)
    ansatz = _ansatz(n_features, reps=ansatz_reps)

    kwargs = dict(
        feature_map=fmap,
        ansatz=ansatz,
        optimizer=_optimizer(maxiter),
        loss="cross_entropy",
    )

    sampler = _exact_sampler(seed)
    if sampler is not None:
        kwargs["sampler"] = sampler

    return VQC(**kwargs)


def build_qsvc(
    n_features: int,
    *,
    feature_map: Literal["z", "zz"] = "zz",
    feature_reps: int = 2,
    seed: int = 42,
):
    """Build a Qiskit fidelity quantum-kernel SVM.

    Probability calibration is intentionally performed externally when needed.
    This avoids the deprecated scikit-learn SVC(probability=...) interface while
    preserving Qiskit's FidelityQuantumKernel for kernel evaluation.
    """
    if not qiskit_ml_available():
        raise ImportError(
            "Qiskit Machine Learning is required. "
            "Install requirements-quantum.txt."
        )

    from sklearn.svm import SVC
    from qiskit_machine_learning.kernels import FidelityQuantumKernel

    fmap = _feature_map(feature_map, n_features, reps=feature_reps)
    quantum_kernel = FidelityQuantumKernel(
        feature_map=fmap,
        evaluate_duplicates="none",
    )

    def quantum_kernel_callable(X, Y):
        return quantum_kernel.evaluate(x_vec=X, y_vec=Y)

    return SVC(
        kernel=quantum_kernel_callable,
        random_state=seed,
    )
