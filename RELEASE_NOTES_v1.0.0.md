# Quantum Cyber Anathema v1.0.0

First public research release of the **Quantum Cyber Anathema (QCA)** reproducibility archive.

## Scope

This release supports the empirical study:

**Quantum Cyber Anathema: Reflexive Quantum–Classical Learning for Network Intrusion Detection with Multi-Seed Evaluation and Cross-Dataset External Validation**

QCA is an empirical quantum-classical implementation of the broader Cyber Anathema Theory research programme.

## Included

- leakage-controlled CICIDS2017 preprocessing;
- classical baselines (Logistic Regression, Random Forest, XGBoost);
- PennyLane Static VQC;
- Qiskit VQC / fidelity quantum-kernel baseline support;
- QCA-v1 reflexive weighting and replay mechanisms;
- QCA-v2 margin and boundary-focused hard-replay mechanisms;
- five-seed evaluation using seeds 42, 123, 2026, 7 and 99;
- validation-only QCA-v2 development protocol;
- frozen UNSW-NB15 external-validation protocol;
- compact publication-facing result tables;
- methodological guard tests and reproducibility documentation;
- MIT License and `CITATION.cff` metadata.

## Frozen QCA-v2 configuration

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

## Scientific findings represented by this release

- Classical tree ensembles were the strongest predictive baselines in the matched experiments.
- QCA-v2 produced development-stage improvements over a matched Static VQC on CICIDS2017 validation data.
- Those ranking improvements did not replicate on the independent UNSW-NB15 external benchmark.
- This release therefore does **not** claim quantum computational or predictive advantage.

## Reproducibility audit

The release source snapshot passed the repository audit with:

```text
21 passed, 2 skipped
```

The two skipped tests are dataset-dependent UNSW checks that require locally prepared, non-distributed benchmark artifacts.

See `AUDIT_REPORT.md`, `REPRODUCIBILITY.md`, `EXECUTION_ORDER.md`, `QCA_V2_DEVELOPMENT_GUIDE.md`, and `UNSW_EXTERNAL_VALIDATION_GUIDE.md` for details.

## Data

Raw CICIDS2017 and UNSW-NB15 datasets are not redistributed. Obtain them from their official or authorized sources and follow `DATASET_SETUP.md`.

## Authors

- Joshua Akowuje Weyori
- Peter Nimbe

## License

MIT License.
