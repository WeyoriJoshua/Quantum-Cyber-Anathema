from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ProjectConfig:
    """Single source of truth for the CICIDS2017 final-scale seed-42 experiment."""

    dataset_name: str = "CICIDS2017"
    experiment_tag: str = "CICIDS2017_5K_seed42"
    seed: int = 42
    n_qubits: int = 4
    max_rows: int | None = 5000

    # Matched benchmark used by all headline models.
    benchmark_train: int = 1000
    benchmark_val: int = 300
    benchmark_test: int = 300

    # Standard PennyLane VQC.
    vqc_epochs: int = 60
    vqc_patience: int = 10

    # QCA maximum budget: 20 warm-up + (4 cycles x 10 epochs) = 60.
    qca_warmup_epochs: int = 20
    qca_warmup_patience: int = 8
    qca_cycles: int = 4
    qca_epochs_per_cycle: int = 10

    # Common optimization settings.
    batch_size: int = 16
    learning_rate: float = 0.01
    qiskit_vqc_maxiter: int = 200

    # Multi-seed publication protocol.
    final_seeds: tuple[int, ...] = (42, 123, 2026, 7, 99)
    multiseed_tag: str = "CICIDS2017_5K_multiseed_q4"

    # QCA-v2 validation-only development.
    qca_v2_development_tag: str = "CICIDS2017_5K_QCAv2_dev_q4"
    qca_v2_split_seed: int = 314159
    qca_v2_inner_val_fraction: float = 0.20
    qca_v2_margin_lambda: float = 0.50
    qca_v2_probability_margin: float = 0.15
    qca_v2_boundary_weight: float = 1.00
    qca_v2_boundary_threshold: float = 0.50

    # UNSW-NB15 external validation (frozen after QCA-v2 development).
    unsw_external_tag: str = "UNSW_NB15_external_q4"
    unsw_benchmark_seed: int = 42
    unsw_train_size: int = 1000
    unsw_val_size: int = 300
    unsw_test_size: int = 300
    unsw_qca_v2_variant: str = "QCA_MarginHardReplay"
    unsw_qca_v2_margin_lambda: float = 0.50
    unsw_qca_v2_probability_margin: float = 0.15
    unsw_qca_v2_boundary_weight: float = 1.00
    unsw_qca_v2_boundary_threshold: float = 0.50

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
        return self.project_root / "results" / "preprocessed" / self.experiment_tag

    @property
    def quantum_splits_path(self) -> Path:
        return self.preprocessed_dir / "quantum_splits.npz"

    @property
    def benchmark_splits_path(self) -> Path:
        return self.preprocessed_dir / "benchmark_splits.npz"


CONFIG = ProjectConfig()
