# QCA-v2 Development Build

This build extends the completed five-seed CICIDS2017 project without modifying or overwriting the frozen single-seed or multi-seed results.

## Added code

- `models/weighted_vqc.py`
- `qca/v2_loss.py`
- `qca/v2_trainer.py`
- `run_qca_v2_development.py`
- `tests/test_qca_v2_development.py`
- `QCA_V2_DEVELOPMENT_GUIDE.md`

## Development safeguards

- The runner accesses only `X_train`, `y_train`, `X_val`, and `y_val` from the frozen benchmark NPZ.
- It never accesses `X_test` or `y_test`.
- A single fixed 800/200 stratified inner split of the existing 1000-sample training partition is reused across all variants and model seeds.
- The existing 300-sample validation partition is an outer development-validation set, not a new independent test set.
- QCA-v2 cycle selection and classification-threshold selection use inner validation only.
- The completed QCA seed-42 hyperparameters remain the base configuration.

## Default variants

1. StaticVQC
2. WeightedVQC
3. QCA_Current
4. QCA_Margin
5. QCA_HardReplay
6. QCA_MarginHardReplay

## Validation performed before packaging

- Python syntax compilation passed for all new modules.
- `PYTHONPATH=. pytest -q tests/test_qca_v2_development.py tests/test_multiseed_statistics.py` passed: 9 tests.
- QCA-v2 dry-run correctly produced 800 development-training, 200 inner-validation, and 300 outer-validation samples.
- Static code audit confirmed no `data["X_test"]` or `data["y_test"]` access in `run_qca_v2_development.py`.

The user should rerun the dry-run locally because absolute paths in the user's Windows environment differ from the build environment.
