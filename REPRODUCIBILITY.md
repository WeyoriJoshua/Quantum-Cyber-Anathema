# Reproducibility Protocol

This document records the experimental order and the safeguards used in the Quantum Cyber Anathema (QCA) empirical study.

## 1. Scientific boundary

QCA is evaluated here as a simulator-based quantum-machine-learning proof of concept. The repository does not claim quantum computational advantage, hardware speedup, or universally superior intrusion-detection performance.

## 2. Fixed seeds

Primary multi-seed experiments use:

```text
42, 123, 2026, 7, 99
```

These seeds quantify stochastic training/initialization variability on fixed benchmark partitions. They are not substitutes for uncertainty over new data resamples.

## 3. CICIDS2017 development sequence

The intended order is:

```powershell
python run_preprocessing.py
python run_classical_baselines.py
python run_pennylane_vqc.py
python run_qiskit_baselines.py
python run_feature_map_ablation.py
python run_qca.py
python run_qca_ablation.py
python run_multiseed_benchmark.py --suite core
python run_qca_v2_development.py
```

Important controls:

- cleaning and duplicate/conflict controls occur before splitting;
- fitted preprocessing is learned from training data only;
- the headline classical and quantum models use the same matched benchmark observations;
- validation data select thresholds and model/cycle settings;
- QCA-v2 development is validation-only and does not access the CICIDS2017 test set;
- the selected QCA-v2 mechanism is frozen before external validation.

## 4. Frozen QCA-v2 configuration

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
```

The configuration is represented in `project_config.py` and must not be retuned after inspection of the UNSW-NB15 external test results.

## 5. UNSW-NB15 external validation

Prepare the external benchmark:

```powershell
python prepare_unsw_external_benchmark.py --dry-run
python prepare_unsw_external_benchmark.py
```

Then validate the frozen protocol:

```powershell
python run_unsw_external_validation.py --dry-run
python run_unsw_external_validation.py
```

The external protocol follows these rules:

- official UNSW-NB15 training data supply training and validation observations;
- official UNSW-NB15 testing data supply the external test observations;
- `id` is excluded as a record identifier;
- `attack_cat` is excluded as target-adjacent attack-family information;
- duplicate/conflicting exact-feature observations are controlled before sampling;
- external-test observations overlapping training features are removed before matched sampling;
- preprocessing is fit on the benchmark training partition only;
- external-test data are not used for model fitting, early stopping, cycle selection, configuration selection, or threshold selection.

## 6. Expected matched benchmark sizes

### CICIDS2017

```text
Training:   1000
Validation:  300
Test:        300
Dimensions:    4
```

### UNSW-NB15

```text
Training:   1000
Validation:  300
External test: 300
Dimensions:    4
```

The exact class counts are printed by the runners and recorded in generated manifests.

## 7. Primary evaluation metrics

The study reports:

- F1 score;
- Matthews correlation coefficient (MCC);
- PR-AUC;
- ROC-AUC;
- precision;
- recall/sensitivity;
- specificity;
- balanced accuracy;
- Brier score;
- log loss;
- confusion-matrix counts;
- training and inference timing where appropriate.

For imbalanced intrusion detection, PR-AUC and MCC are treated as important complements to accuracy.

## 8. QCA operational proxies

The repository may report defensive-intelligence/evolution quantities such as validation defense-score gain, recursive evolution change, and predictive-entropy change. These are operational research proxies. They must not be described as thermodynamic entropy, physical quantum entropy conversion, or proof of autonomous defensive evolution.

## 9. Result interpretation

The final study found that QCA-v2 produced a development-stage improvement relative to the architecture-matched Static VQC on CICIDS2017 validation data, but the ranking advantage did not replicate on the independent UNSW-NB15 external benchmark. Classical tree ensembles remained the strongest predictive baselines.

This negative external result is part of the intended scientific record and must not be removed or retuned away.
