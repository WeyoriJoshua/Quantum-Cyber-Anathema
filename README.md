# Quantum Cyber Anathema (QCA)

Reproducible research implementation of **Quantum Cyber Anathema**, a quantum-classical empirical instantiation of the broader **Cyber Anathema Theory (CAT)** research programme.

The core empirical question is whether hostile observations can be converted into useful defensive training signals through reflexive weighting, adversarial memory, boundary-focused replay, and recursive model updating—without assuming that adaptation automatically implies generalization or quantum advantage.

## Scientific scope

This repository supports the empirical manuscript:

**Quantum Cyber Anathema: Reflexive Quantum–Classical Learning for Network Intrusion Detection with Multi-Seed Evaluation and Cross-Dataset External Validation**

The associated theory manuscript develops the broader Cyber Anathema Theory. The theory paper and this empirical implementation are related but distinct: CAT is substrate-neutral, whereas this repository evaluates a specific quantum-classical intrusion-detection implementation.

> **Important scientific boundary:** the quantum experiments in this repository use local statevector/simulator-based execution. The results are a quantum-machine-learning proof of concept and **must not be interpreted as evidence of quantum computational advantage**.

## Authors

- **Joshua Akowuje Weyori**
- **Dr Peter Nimbe**

Formal citation metadata are provided in [`CITATION.cff`](CITATION.cff).

## Experimental stages

The publication workflow is deliberately staged to protect the test sets.

1. **CICIDS2017 matched benchmark** — leakage-controlled preprocessing and architecture-matched classical/quantum controls.
2. **Five-seed evaluation** — fixed seeds `42, 123, 2026, 7, 99` on the same benchmark partition.
3. **QCA-v2 development** — validation-only mechanism development; the previously inspected CICIDS2017 test set is not accessed.
4. **Frozen external validation** — the selected QCA-v2 mechanism is frozen before evaluation on UNSW-NB15.

The frozen QCA-v2 configuration is:

```text
Variant: QCA_MarginHardReplay
Qubits: 4
Circuit layers: 2
Learning rate: 0.01
Batch size: 16
Uncertainty beta: 0.10
Margin lambda: 0.50
Probability margin: 0.15
Replay mode: boundary
Boundary weight: 1.00
Boundary threshold: 0.50
Memory capacity: 600
Replay ratio: 0.20
Warm-up epochs: 20
Cycles: 4
Epochs per cycle: 10
Maximum training budget: 60 epochs
Seeds: 42, 123, 2026, 7, 99
```

## Main findings represented by this codebase

The repository is designed to reproduce the study rather than to advertise only positive results.

- Classical tree ensembles were the strongest predictive baselines in the matched experiments.
- QCA-v2 showed a development-stage improvement over a matched Static VQC on CICIDS2017 validation data.
- That ranking advantage did **not** replicate on the independent UNSW-NB15 external benchmark.
- The study therefore does **not** claim quantum predictive superiority.

## Repository structure

```text
.
├── models/                         # Classical, PennyLane and Qiskit model definitions
├── qca/                            # QCA-v1 and QCA-v2 loss, memory and training logic
├── preprocessing/                  # Leakage audit, cleaning, splitting and quantum pipeline
├── evaluation/                     # Metrics, evolution summaries and plotting helpers
├── experiments/                    # Shared experiment utilities
├── tests/                          # Protocol and preprocessing tests
├── results/                        # Compact publication result tables only
├── project_config.py               # Publication configuration
├── run_preprocessing.py
├── run_classical_baselines.py
├── run_pennylane_vqc.py
├── run_qiskit_baselines.py
├── run_qca.py
├── run_qca_ablation.py
├── run_multiseed_benchmark.py
├── run_qca_v2_development.py
├── prepare_unsw_external_benchmark.py
└── run_unsw_external_validation.py
```

The script-based runners are the canonical publication workflow. Exploratory notebooks and raw/generated large artifacts are intentionally not part of the release archive so that reviewers have a single unambiguous implementation path.

## Environment

Recommended: **Python 3.11**.

```powershell
py -3.11 -m venv QUANTUM
.\QUANTUM\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements-verified.txt
```

The verified publication environment includes PennyLane, Qiskit, Qiskit Machine Learning, scikit-learn, XGBoost and PyTorch. See `requirements-verified.txt`.

## Datasets

Raw datasets are intentionally **not** stored in this repository.

### CICIDS2017

Place the official MachineLearningCVE CSV files under:

```text
datasets/
└── CIC-IDS-2017/
    └── CSV/
        └── MachineLearningCSV/
            └── MachineLearningCVE/
```

### UNSW-NB15

Place the official train/test CSV files under:

```text
datasets/
└── UNSW-NB15/
    ├── UNSW_NB15_training-set.csv
    └── UNSW_NB15_testing-set.csv
```

The UNSW external-validation pipeline excludes target-adjacent annotation fields such as `attack_cat` from the binary prediction features and removes exact feature overlap between the development pool and the external test pool before matched sampling.

See `DATASET_SETUP.md` for dataset placement and protocol notes.

## Reproduction order

For the final publication workflow:

```powershell
python run_preprocessing.py
python run_classical_baselines.py
python run_pennylane_vqc.py
python run_qiskit_baselines.py
python run_qca.py
python run_qca_ablation.py
python run_multiseed_benchmark.py --suite core
python run_qca_v2_development.py
python prepare_unsw_external_benchmark.py
python run_unsw_external_validation.py
```

See `EXECUTION_ORDER.md`, `MULTISEED_EXPERIMENT_GUIDE.md`, `QCA_V2_DEVELOPMENT_GUIDE.md`, and `UNSW_EXTERNAL_VALIDATION_GUIDE.md` for stage-specific details.

## Reproducibility safeguards

The code implements the following controls:

- canonical binary labels (`0 = benign`, `1 = attack`);
- train-only fitting of imputation, scaling, encoding, PCA, and angle transformation;
- duplicate and contradictory-label controls before splitting;
- matched observations across headline model comparisons;
- validation-only threshold/model selection;
- fixed multi-seed protocol;
- architecture-matched Static VQC control;
- explicit QCA component ablations;
- frozen QCA-v2 configuration before external validation;
- external test data excluded from tuning.

## Testing

```powershell
python smoke_test.py
pytest -q
```

## Results included in the repository

Only compact publication-facing result tables are retained in `results/`. Raw datasets, large checkpoints, intermediate arrays, caches and bulky generated artifacts are excluded.

The retained tables cover:

- CICIDS2017 five-seed summary metrics;
- paired QCA-vs-Static-VQC comparison;
- QCA-v2 development-stage variant ranking and paired comparisons;
- UNSW-NB15 external descriptive ranking and frozen QCA-v2 paired comparisons.

## Citation

If you use this software or reproduce the study, use the citation metadata in [`CITATION.cff`](CITATION.cff). A permanent archival DOI may be added after the first public release is deposited in an archival service such as Zenodo.

## License

This repository is released under the **MIT License**. See [`LICENSE`](LICENSE).

## Data availability

CICIDS2017 and UNSW-NB15 are third-party benchmark datasets and are not redistributed here. Users should obtain them from official or otherwise authorized distribution sources and place them in the folder layout described above.

## Status

This repository is maintained as the reproducibility archive for the QCA empirical study. The broader Cyber Anathema Theory is a separate conceptual contribution; this codebase should be interpreted as one empirical implementation, not as the entirety of the theory.
