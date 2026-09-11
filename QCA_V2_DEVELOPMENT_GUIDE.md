# QCA-v2 Validation-Only Development Guide

## Why this stage exists

The completed five-seed CICIDS2017 experiment showed that the current QCA procedure changes probability scores and improves its internal validation Defense Score, but does not improve final test decisions over the architecture-matched static PennyLane VQC. QCA-v2 therefore focuses on mechanisms that can alter class separation and the decision boundary rather than merely shift confidence calibration.

The previously inspected 300-sample test partition is frozen and is **not accessed by the QCA-v2 development runner**.

## Development data protocol

The runner opens the frozen `benchmark_splits.npz` but accesses only:

- `X_train`, `y_train`: original 1000-sample matched training partition
- `X_val`, `y_val`: original 300-sample validation partition, now used as an outer development-validation set

It deliberately does not access `X_test` or `y_test`.

The original 1000-sample training partition is split once, with fixed split seed `314159`, into:

- 800 samples: development training
- 200 samples: inner validation

The same 800/200 split is reused for every model seed and every QCA-v2 variant. This keeps model-seed comparisons paired and prevents data-split variation from being confused with stochastic optimization variation.

## Variants

The default runner compares six architecture-matched variants:

1. `StaticVQC` — ordinary BCE PennyLane VQC.
2. `WeightedVQC` — inverse-frequency class-balanced BCE control.
3. `QCA_Current` — the existing QCA implementation with its frozen seed-42 validation-selected configuration.
4. `QCA_Margin` — current reflexive weighting plus an additive probability-margin penalty.
5. `QCA_HardReplay` — current reflexive loss with replay priority focused on misclassified and near-boundary samples.
6. `QCA_MarginHardReplay` — margin-aware reflexive loss plus hard-boundary replay.

All use the same two-layer angle-encoded four-qubit circuit and the same maximum optimization budget used in the existing experiment.

## QCA-v2 mechanisms

### Weighted VQC control

For class `c`, the training weight is

`w_c = N / (2 N_c)`.

Weights are fitted from the 800-sample development-training labels only. This control tests whether any apparent QCA benefit is simply ordinary minority-class reweighting.

### Margin-aware objective

QCA-v2 retains the original reflexively weighted BCE and adds an optional probability-space margin term:

`L_total = L_reflexive_BCE + lambda_margin * L_margin`.

With the default probability margin of `0.15`, malicious samples are encouraged toward `p >= 0.65` and benign samples toward `p <= 0.35`. The margin penalty becomes zero once that separation is achieved.

Defaults:

- `margin_lambda = 0.50`
- `margin = 0.15`

### Hard-boundary replay

Boundary hardness is

`h = 1 - 2 |p - 0.5|`.

It is maximal at the decision boundary and minimal near probabilities 0 and 1. Boundary replay stores misclassified samples and samples with high boundary hardness, then prioritizes them using error, boundary hardness and malicious-class priority.

Defaults:

- `boundary_weight = 1.0`
- `boundary_threshold = 0.50`
- existing memory capacity = 600
- existing replay ratio = 0.20

## Selection safeguards

- VQC early stopping uses the 200-sample inner validation partition.
- QCA cycle selection uses inner-validation Defense Score only.
- Classification thresholds are selected on inner validation only.
- The 300-sample outer validation partition is evaluated only after the fitted model/cycle and threshold are fixed.
- The frozen 300-sample test partition is not loaded.

The outer validation results may be used to choose a QCA-v2 formulation for later external confirmation. They must not be described as final independent test performance.

## Commands

First validate paths and the fixed split:

```powershell
python run_qca_v2_development.py --dry-run
```

Expected structure:

```text
Original training partition: (1000, 4)
Development train: (800, 4)
Inner validation: (200, 4)
Outer validation: (300, 4)
TEST PARTITION ACCESSED: False
Model seeds: [42, 123, 2026, 7, 99]
```

For a software-only first run using seed 42:

```powershell
python run_qca_v2_development.py --seeds 42
```

Do not change parameters based on the seed-42 outer-validation values alone. If the run completes without software errors, continue the frozen design across all five seeds:

```powershell
python run_qca_v2_development.py
```

The runner is resumable. Completed variant-seed pairs are skipped automatically.

## Main outputs

Results are written under:

`results/qca_v2_development/CICIDS2017_5K_QCAv2_dev_q4/`

Key outputs:

- `development_manifest.json`
- `fixed_development_split_indices.npz`
- `all_seed_outer_validation_metrics.csv`
- `summary_mean_sd_95CI.csv`
- `paired_comparisons_vs_StaticVQC.csv`
- `paired_comparisons_vs_WeightedVQC.csv`
- `prediction_behavior_vs_StaticVQC.csv`
- `outer_validation_variant_ranking.csv`
- `all_inner_validation_predictions.csv`
- `all_outer_validation_predictions.csv`

Per-variant folders also contain inner threshold searches, outer-validation confusion matrices, probability predictions, training histories, QCA evolution histories and memory summaries.

## Decision rule after this stage

A QCA-v2 variant is interesting only if it shows more than a threshold shift. Look for:

- consistently higher outer-validation PR-AUC and MCC than `StaticVQC`;
- improvement over `WeightedVQC`, demonstrating value beyond ordinary class weighting;
- larger malicious-versus-benign score separation;
- meaningful decision disagreement with StaticVQC where changes are more often error-to-correct than correct-to-error;
- consistency across the five model seeds.

If no QCA-v2 variant meets those criteria, retain the current QCA as a negative/diagnostic result rather than forcing a superiority claim.

If one variant is consistently stronger, freeze it before moving to external confirmation on UNSW-NB15.
