from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ProjectConfig:
    """Single source of truth for the CICIDS2017 final-scale seed-42 experiment."""

    # ============================================================
    # DATASET / EXPERIMENT
    # ============================================================

    dataset_name: str = "CICIDS2017"

    # Keep this run separate from the previous debug experiment.
    experiment_tag: str = "CICIDS2017_5K_seed42"

    seed: int = 42

    # Quantum dimensionality
    n_qubits: int = 4

    # Increased from 2,000 → 5,000
    max_rows: int | None = 5000

    # ============================================================
    # MATCHED BENCHMARK
    # Same partition used by ALL headline models
    # ============================================================

    benchmark_train: int = 1000
    benchmark_val: int = 300
    benchmark_test: int = 300

    # ============================================================
    # STANDARD PENNYLANE VQC
    # ============================================================

    vqc_epochs: int = 60
    vqc_patience: int = 10

    # ============================================================
    # QUANTUM CYBER ANATHEMA
    #
    # Maximum training budget:
    # 20 warm-up epochs + (4 cycles × 10 epochs) = 60
    # ============================================================

    qca_warmup_epochs: int = 20
    qca_warmup_patience: int = 8
    qca_cycles: int = 4
    qca_epochs_per_cycle: int = 10

    # ============================================================
    # COMMON OPTIMIZATION SETTINGS
    # ============================================================

    batch_size: int = 16
    learning_rate: float = 0.01

    # Qiskit VQC uses optimizer iterations rather than epochs
    qiskit_vqc_maxiter: int = 200


    # ============================================================
    # PUBLICATION ROBUSTNESS / MULTI-SEED PROTOCOL
    # ============================================================

    # Fixed model-initialization/training seeds evaluated on the SAME
    # 1000/300/300 benchmark partition. This isolates stochastic
    # optimization variability while keeping the data partition paired.
    final_seeds: tuple[int, ...] = (42, 123, 2026, 7, 99)

    # Dedicated output namespace. The completed single-seed result under
    # CICIDS2017_5K_seed42 is preserved and is never overwritten.
    multiseed_tag: str = "CICIDS2017_5K_multiseed_q4"


    # ============================================================
    # QCA-v2 VALIDATION-ONLY DEVELOPMENT
    # ============================================================

    # This stage never accesses the previously inspected benchmark test set.
    qca_v2_development_tag: str = "CICIDS2017_5K_QCAv2_dev_q4"
    qca_v2_split_seed: int = 314159
    qca_v2_inner_val_fraction: float = 0.20

    # Frozen first-pass mechanism settings. They are development parameters,
    # not post-test changes to the completed QCA experiment.
    qca_v2_margin_lambda: float = 0.50
    qca_v2_probability_margin: float = 0.15
    qca_v2_boundary_weight: float = 1.00
    qca_v2_boundary_threshold: float = 0.50



    # ============================================================
    # UNSW-NB15 EXTERNAL VALIDATION (FROZEN AFTER QCA-v2 DEVELOPMENT)
    # ============================================================

    unsw_external_tag: str = "UNSW_NB15_external_q4"
    unsw_benchmark_seed: int = 42
    unsw_train_size: int = 1000
    unsw_val_size: int = 300
    unsw_test_size: int = 300

    # Frozen QCA-v2 mechanism selected on CICIDS2017 development data.
    # These settings must not be tuned using the UNSW-NB15 test set.
    unsw_qca_v2_variant: str = "QCA_MarginHardReplay"
    unsw_qca_v2_margin_lambda: float = 0.50
    unsw_qca_v2_probability_margin: float = 0.15
    unsw_qca_v2_boundary_weight: float = 1.00
    unsw_qca_v2_boundary_threshold: float = 0.50

    # ============================================================
    # PATHS
    # ============================================================

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT


    @property
    def unsw_dataset_dir(self) -> Path:
        return self.project_root / "datasets" / "UNSW-NB15"

    @property
    def unsw_training_path(self) -> Path:
        return self.unsw_dataset_dir / "UNSW_NB15_training-set.csv"

    @property
    def unsw_testing_path(self) -> Path:
        return self.unsw_dataset_dir / "UNSW_NB15_testing-set.csv"

    @property
    def unsw_preprocessed_dir(self) -> Path:
        return self.project_root / "results" / "preprocessed" / self.unsw_external_tag

    @property
    def unsw_benchmark_splits_path(self) -> Path:
        return self.unsw_preprocessed_dir / "benchmark_splits.npz"

    @property
    def raw_dataset_path(self) -> Path:
        return (
            self.project_root
            / "datasets"
            / "CIC-IDS-2017"
            / "CSV"
            / "MachineLearningCSV"
            / "MachineLearningCVE"
        )

    @property
    def preprocessed_dir(self) -> Path:
        return (
            self.project_root
            / "results"
            / "preprocessed"
            / self.experiment_tag
        )

    @property
    def quantum_splits_path(self) -> Path:
        return (
            self.preprocessed_dir
            / "quantum_splits.npz"
        )

    @property
    def benchmark_splits_path(self) -> Path:
        return (
            self.preprocessed_dir
            / "benchmark_splits.npz"
        )


CONFIG = ProjectConfig()
