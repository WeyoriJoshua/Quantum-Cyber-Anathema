# UNSW-NB15 External Validation Guide

## Purpose

This stage externally evaluates the frozen QCA-v2 mechanism selected during CICIDS2017 validation-only development. No QCA-v2 hyperparameter is tuned using the UNSW-NB15 external test benchmark.

## Data supplied

The project includes the standard files:

- `datasets/UNSW-NB15/UNSW_NB15_training-set.csv`
- `datasets/UNSW-NB15/UNSW_NB15_testing-set.csv`

Observed raw sizes:

- Official training: 175,341 rows
- Official testing: 82,332 rows

The model target is the binary `label` column (`0 = normal`, `1 = attack`).

`id` is excluded as a record identifier. `attack_cat` is excluded because it is an attack-family annotation adjacent to the binary target and would introduce target leakage into binary attack detection.

## Leakage-control and duplicate-control protocol

The prepared external benchmark applies the following safeguards before sampling:

1. Contradictory exact feature rows are removed within each official split.
2. Duplicate feature+label rows are removed within each official split.
3. Official testing rows whose model features duplicate rows in the official training file are removed before the external test sample is drawn.
4. Training and validation come only from the official training file.
5. External test comes only from the official testing file.
6. Encoding, scaling, PCA and angle scaling are fitted only on the 1,000-row benchmark training partition.
7. Validation and test call transform only.

After leakage/duplicate control, the available pools were:

- Clean official training: 100,811 rows (`51,661` normal, `49,150` attack)
- Decontaminated official testing: 52,738 rows (`33,755` normal, `18,983` attack)

The fixed matched benchmark is:

- Train: 1,000 rows (`512` normal, `488` attack)
- Validation: 300 rows (`154` normal, `146` attack)
- External test: 300 rows (`192` normal, `108` attack)

The 42 non-target model features are processed into 65 encoded columns and reduced to four PCA dimensions. The fitted four components explain approximately `0.4782` of the variance in the encoded training representation. The four components are scaled to `[0, pi]` using training-fitted parameters.

## Frozen QCA-v2 configuration

The following mechanism was selected before UNSW-NB15 test evaluation, using CICIDS2017 five-seed validation-only development:

- Variant: `QCA_MarginHardReplay`
- Encoding: angle
- Qubits: 4
- Layers: 2
- Beta uncertainty: 0.10
- Margin lambda: 0.50
- Probability margin: 0.15
- Replay priority: boundary
- Boundary weight: 1.00
- Boundary threshold: 0.50
- Memory capacity: 600
- Replay ratio: 0.20
- Warm-up: 20 epochs
- Reflexive cycles: 4
- Epochs per cycle: 10
- Maximum optimization budget: 60 epochs

Do not change these QCA-v2 hyperparameters after viewing UNSW-NB15 external test performance.

## Models

The external-validation runner evaluates:

- Logistic Regression
- Random Forest
- XGBoost
- Static PennyLane VQC
- Class-weighted VQC
- Current/original QCA
- Frozen QCA-v2 (`QCA_MarginHardReplay`)

All headline models use the same 4-dimensional benchmark representation and the same fixed train/validation/test samples.

## Seeds

`42, 123, 2026, 7, 99`

The data benchmark is fixed across seeds. Seed variation therefore measures training/initialization variability rather than data-resampling variability.

## Recommended commands

From the project root, first verify the prepared benchmark:

```powershell
python run_unsw_external_validation.py --dry-run
```

Expected core output:

```text
Benchmark: (1000, 4) (300, 4) (300, 4)
Seeds: [42, 123, 2026, 7, 99]
QCA-v2 frozen variant: QCA_MarginHardReplay
External test used for tuning: False
```

Then run seed 42 as a software check:

```powershell
python run_unsw_external_validation.py --seeds 42
```

If it completes, run the remaining seeds. Because resume is enabled, seed 42 will be skipped automatically:

```powershell
python run_unsw_external_validation.py
```

## Output directory

`results/external_validation/UNSW_NB15_external_q4/`

Main outputs:

- `all_seed_external_test_metrics.csv`
- `summary_mean_sd_95CI.csv`
- `QCA_V2_external_paired_comparisons.csv`
- `all_external_test_predictions.csv`
- `all_validation_predictions.csv`
- `external_test_descriptive_ranking.csv`
- `external_validation_manifest.json`
- per-seed/per-model thresholds, predictions, confusion matrices and histories

## Interpretation

The external test set is confirmatory. It may be used to evaluate the frozen QCA-v2 mechanism, but it must not be used to tune margin parameters, replay settings, architecture, quantum dimensionality, cycle count or other QCA-v2 design choices.

Simulator-based predictive performance does not establish quantum computational advantage.
