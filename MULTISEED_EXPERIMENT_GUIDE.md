# Multi-Seed Publication Experiment Guide

## Purpose

This stage tests whether the QCA result is robust to stochastic model initialization/training rather than being a single-seed artifact.

The primary comparison is QCA versus the architecture-matched static PennyLane VQC. Classical models remain strong reference controls.

## Frozen protocol

- Dataset/preprocessing source: `CICIDS2017_5K_seed42`
- Fixed benchmark: 1000 train / 300 validation / 300 test
- Qubits/features: 4
- Seeds: 42, 123, 2026, 7, 99
- Static PennyLane VQC: maximum 60 epochs, patience 10
- QCA: 20 warm-up + 4 × 10 reflexive epochs = maximum budget 60
- Threshold: validation-only MCC by default
- Test: evaluated only after model and threshold are frozen

The runner reads the completed seed-42 `selected_qca_config.json` and freezes its validation-selected QCA settings, including `beta_uncertainty=0.10`, across the primary multi-seed analysis. This avoids changing the protocol after inspecting test performance.

## Step 1 — Dry run

From the project root:

```powershell
python run_multiseed_benchmark.py --dry-run
```

Confirm:

```text
Benchmark: (1000, 4) (300, 4) (300, 4)
Seeds: [42, 123, 2026, 7, 99]
Suite: core
```

## Step 2 — Run the primary core suite

```powershell
python run_multiseed_benchmark.py --suite core
```

This runs:

- Logistic Regression
- Random Forest
- XGBoost
- PennyLane VQC using the same quantum core/encoding as QCA
- QCA

The command is resumable. If execution is interrupted, run the same command again.

## Step 3 — Review generated statistics

Main files:

`results/multiseed/CICIDS2017_5K_multiseed_q4/all_seed_metrics.csv`

`results/multiseed/CICIDS2017_5K_multiseed_q4/summary_mean_sd_95CI.csv`

`results/multiseed/CICIDS2017_5K_multiseed_q4/QCA_vs_VQC_paired_comparison.csv`

`results/multiseed/CICIDS2017_5K_multiseed_q4/all_test_predictions.csv`

The paired comparison includes:

- mean QCA − VQC difference
- standard deviation of paired differences
- 95% t confidence interval of the seed-level difference
- QCA wins/ties/losses
- exact sign-flip p-value
- Wilcoxon p-value
- Holm-adjusted p-values

With only five seeds, inferential p-values have low power. Report effect sizes/differences and confidence intervals prominently rather than treating p-values as decisive.

## Step 4 — Optional full quantum-control suite

Only after the core result is complete:

```powershell
python run_multiseed_benchmark.py --suite full
```

Because resume is enabled by default, completed core models are skipped and only the additional Qiskit controls are run.

The fidelity quantum-kernel SVM is extremely expensive under local simulation. The completed seed-42 experiment required several hours for this model. Five-seed repetition may therefore require more than a day depending on hardware.

## Interpretation rule

The multi-seed analysis uses one fixed benchmark across seeds. Therefore:

- it quantifies stochastic training/initialization variability;
- it permits paired-by-seed QCA versus VQC comparisons;
- it does not quantify uncertainty from drawing a new train/test partition.

Do not call the result a quantum advantage unless there is evidence beyond simulator-based predictive performance and appropriate computational comparisons.
