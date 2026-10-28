# HQCA CIC-IDS-2017 Zero-Day Complementarity-Shift Validation

## Purpose

This notebook implements the pre-specified zero-day validation phase of the merged Cyber Anathema / HQCA-CQCS manuscript.

It tests whether the **Classical–Quantum Complementarity Shift** observed during UNSW-NB15 OOD analysis generalizes under a pre-specified CIC-IDS-2017 source/attack-family holdout validation protocol. CIC-IDS-2017 also appears in the broader project as a standard benchmark, so this phase is not described as an independent-dataset confirmation; independence here refers to the locked held-out source/attack-family conditions and target exclusion from development.

## Locked zero-day conditions

The experiment holds out the complete source CSV associated with:

1. PortScan
2. Infiltration
3. Web Attack

For each condition, the source CSV is absent from development and the target attack family is removed from every remaining development source.

The zero-day test contains only benign traffic plus the target unseen attack family from the held-out source.

## Seeds

Exactly 10: `2026–2035`. Do not delete unfavorable seeds.

## Models

- Classical CAT
- Quantum CAT
- 50:50 Hybrid
- Static Weighted Hybrid
- Generic RF Stacking
- Adaptive HQCA

## Primary contribution tested

The primary quantity is `P(Quantum correct | Classical wrong)`, measured in development OOF predictions and the zero-day held-out attack condition.

The notebook reports `zero-day complementarity / development complementarity` as **Classical–Quantum Complementarity Shift**.

It also reports raw quantum-only correction prevalence, classical-only correction prevalence, oracle Classical-or-Quantum accuracy/headroom, MCC, balanced accuracy, F1, ROC-AUC, PR-AUC, controlled-FPR recall, static fusion weights, and paired 10-seed tests.

## Environment

Native Windows, Python (QUANTUM_GPU), NVIDIA GeForce RTX 3060 Laptop GPU.

## Dataset

Expected local root:
`E:\QUANTUM_CYBER\Cyber Anathema\datasets\CIC-IDS-2017\CSV\MachineLearningCSV\MachineLearningCVE`

## Runtime

The full locked run contains 3 zero-day conditions × 10 seeds × branch OOF/final fits. A one-seed smoke test is permitted only for technical debugging; restore the locked configuration before manuscript results are generated.
