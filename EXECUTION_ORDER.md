# Quantum Cyber Anathema — Current Execution Order

Use the Python 3.11 `QUANTUM` environment selected in VS Code.

## Completed single-seed 5K experiment

The frozen preprocessing/output namespace is:

`CICIDS2017_5K_seed42`

The fixed matched benchmark is:

- Train: 1000 samples
- Validation: 300 samples
- Test: 300 samples
- Quantum dimensions/qubits: 4

The completed single-seed notebooks are:

1. `03_Classical_Baselines.ipynb`
2. `04_Standard_PennyLane_VQC.ipynb`
3. `05_Qiskit_VQC_QSVC_LEAN.ipynb`
4. `06_Quantum_Feature_Map_Ablation_LEAN.ipynb`
5. `07_Quantum_Cyber_Anathema.ipynb`
6. `08_QCA_Recursive_Evolution_and_Ablation.ipynb`

Do not rerun preprocessing unless intentionally regenerating the frozen benchmark.

## Publication robustness stage — run now

First validate the planned multi-seed experiment:

```powershell
python run_multiseed_benchmark.py --dry-run
```

Then run the core multi-seed suite:

```powershell
python run_multiseed_benchmark.py --suite core
```

Core models:
- Logistic Regression
- Random Forest
- XGBoost
- architecture-matched PennyLane VQC
- QCA

Seeds:
`42 123 2026 7 99`

All seeds use the same fixed 1000/300/300 benchmark. Only model/training randomness changes.

The runner is resumable. If interrupted, run the same command again; completed model/seed pairs are skipped.

## Optional full quantum-control suite

After the core suite is complete:

```powershell
python run_multiseed_benchmark.py --suite full
```

This resumes the completed core runs and adds:
- Qiskit VQC using the pilot-validation-selected feature map
- Qiskit FidelityQuantumKernel + calibrated sklearn SVC

The quantum-kernel SVM is computationally expensive on a simulator, so the full suite can take many additional hours.

## Statistical outputs

The multi-seed runner writes:

- `all_seed_metrics.csv`
- `summary_mean_sd_95CI.csv`
- `QCA_vs_VQC_paired_comparison.csv`
- `all_test_predictions.csv`
- `all_validation_predictions.csv`
- per-model confusion matrices
- per-seed thresholds
- per-seed QCA evolution histories
- `run_manifest.json`

Decision thresholds are selected on validation data only. Test data remain untouched until model fitting and threshold selection are frozen.

## QCA-v2 mechanism-development stage — current next step

The completed five-seed test results are frozen. Do not overwrite them.

QCA-v2 development deliberately excludes the frozen 300-sample test partition. Run:

```powershell
python run_qca_v2_development.py --dry-run
python run_qca_v2_development.py --seeds 42
```

The seed-42 command is a software validation run only. If it completes without errors, continue the same frozen design across all five seeds:

```powershell
python run_qca_v2_development.py
```

Read `QCA_V2_DEVELOPMENT_GUIDE.md` before changing any QCA-v2 hyperparameter.
