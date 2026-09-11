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
        # Qiskit 2.x functional circuit constructors.
        from qiskit.circuit.library import (
            z_feature_map,
            zz_feature_map,
        )

        if kind == "z":
            return z_feature_map(
                n_features,
                reps=reps,
            )

        if kind == "zz":
            return zz_feature_map(
                n_features,
                reps=reps,
                entanglement="linear",
            )

    except Exception:
        # Compatibility fallback for older supported interfaces.
        from qiskit.circuit.library import (
            ZFeatureMap,
            ZZFeatureMap,
        )

        if kind == "z":
            return ZFeatureMap(
                feature_dimension=n_features,
                reps=reps,
            )

        if kind == "zz":
            return ZZFeatureMap(
                feature_dimension=n_features,
                reps=reps,
                entanglement="linear",
            )

    raise ValueError(
        "kind must be 'z' or 'zz'"
    )


def _ansatz(
    n_qubits: int,
    reps: int = 2,
):
    try:
        from qiskit.circuit.library import (
            real_amplitudes,
        )

        return real_amplitudes(
            n_qubits,
            reps=reps,
            entanglement="linear",
        )

    except Exception:
        from qiskit.circuit.library import (
            RealAmplitudes,
        )

        return RealAmplitudes(
            num_qubits=n_qubits,
            reps=reps,
            entanglement="linear",
        )


def _optimizer(
    maxiter: int,
):
    """
    Create the COBYLA optimizer.

    In Qiskit Machine Learning 0.9+, optimizers may be
    provided directly by QML. A fallback is retained for
    compatible qiskit-algorithms installations.
    """

    try:
        from qiskit_machine_learning.optimizers import (
            COBYLA,
        )

    except Exception:  # pragma: no cover
        from qiskit_algorithms.optimizers import (
            COBYLA,
        )

    return COBYLA(
        maxiter=maxiter
    )


def _exact_sampler(
    seed: int = 42,
):
    """
    Return an exact local V2 sampler compatible with
    Qiskit Machine Learning 0.9.x.
    """

    try:
        from qiskit_machine_learning.primitives import (
            QMLSampler,
        )

        return QMLSampler(
            shots=None,
            seed=seed,
        )

    except Exception:
        try:
            from qiskit.primitives import (
                StatevectorSampler,
            )

            return StatevectorSampler(
                seed=seed
            )

        except Exception:
            return None


# ============================================================
# QISKIT VARIATIONAL QUANTUM CLASSIFIER
# ============================================================

def build_qiskit_vqc(
    n_features: int,
    *,
    feature_map: Literal["z", "zz"] = "zz",
    feature_reps: int = 2,
    ansatz_reps: int = 2,
    maxiter: int = 100,
    seed: int = 42,
):
    """
    Build the Qiskit Variational Quantum Classifier.

    Parameters
    ----------
    n_features:
        Number of quantum-compatible input dimensions.

    feature_map:
        Quantum feature map: 'z' or 'zz'.

    feature_reps:
        Number of feature-map repetitions.

    ansatz_reps:
        Number of variational ansatz repetitions.

    maxiter:
        Maximum COBYLA optimization iterations.

    seed:
        Reproducibility seed.
    """

    if not qiskit_ml_available():
        raise ImportError(
            "Qiskit Machine Learning is required. "
            "Install requirements-quantum.txt."
        )

    from qiskit_machine_learning.algorithms import (
        VQC,
    )

    fmap = _feature_map(
        feature_map,
        n_features,
        reps=feature_reps,
    )

    ansatz = _ansatz(
        n_features,
        reps=ansatz_reps,
    )

    kwargs = dict(
        feature_map=fmap,
        ansatz=ansatz,
        optimizer=_optimizer(maxiter),
        loss="cross_entropy",
    )

    sampler = _exact_sampler(
        seed
    )

    if sampler is not None:
        kwargs["sampler"] = sampler

    return VQC(
        **kwargs
    )


# ============================================================
# QUANTUM-KERNEL SVM
# ============================================================

def build_qsvc(
    n_features: int,
    *,
    feature_map: Literal["z", "zz"] = "zz",
    feature_reps: int = 2,
    seed: int = 42,
):
    """
    Build a quantum-kernel support vector classifier.

    Qiskit is responsible for construction and evaluation of
    the FidelityQuantumKernel.

    scikit-learn SVC performs the classical SVM optimization.

    Probability calibration is intentionally NOT performed here.
    When probability estimates are required, the returned model
    should be wrapped externally with CalibratedClassifierCV.

    This design avoids the deprecated SVC(probability=...)
    interface in scikit-learn >= 1.9 while preserving the
    Qiskit quantum kernel.
    """

    if not qiskit_ml_available():
        raise ImportError(
            "Qiskit Machine Learning is required. "
            "Install requirements-quantum.txt."
        )

    from sklearn.svm import SVC

    from qiskit_machine_learning.kernels import (
        FidelityQuantumKernel,
    )

    # --------------------------------------------------------
    # Construct Qiskit quantum feature map
    # --------------------------------------------------------

    fmap = _feature_map(
        feature_map,
        n_features,
        reps=feature_reps,
    )

    # --------------------------------------------------------
    # Construct Qiskit fidelity quantum kernel
    # --------------------------------------------------------

    quantum_kernel = FidelityQuantumKernel(
        feature_map=fmap,
        evaluate_duplicates="none",
    )

    # --------------------------------------------------------
    # Callable quantum kernel for sklearn SVC
    # --------------------------------------------------------

    def quantum_kernel_callable(
        X,
        Y,
    ):
        return quantum_kernel.evaluate(
            x_vec=X,
            y_vec=Y,
        )

    # --------------------------------------------------------
    # Classical SVM optimization using quantum kernel matrix
    #
    # IMPORTANT:
    # No probability=True
    # No probability=False
    # --------------------------------------------------------

    return SVC(
        kernel=quantum_kernel_callable,
        random_state=seed,
    )
