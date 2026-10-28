# Quantum Cyber Anathema / Adaptive HQCA–CQCS

Reproducible research implementation of the **Cyber Anathema** programme, with the current empirical manuscript focused on adaptive hybrid quantum–classical intrusion detection, zero-day distribution shift, and nonstationary branch complementarity.

## Current manuscript

**Cyber Anathema: Nonstationary Classical–Quantum Complementarity in Adaptive Intrusion Detection under Zero-Day Distribution Shift**

The manuscript combines:

- leakage-safe Classical CAT, Quantum CAT, Static Hybrid, Generic Stacking, and Adaptive HQCA;
- standard evaluation on UNSW-NB15 and CIC-IDS-2017;
- prospective validation on ToN-IoT;
- attack-family/source-held-out zero-day evaluation;
- Classical–Quantum Complementarity Shift (CQCS);
- calibration and threshold-transfer analysis;
- controlled-FPR evaluation;
- paired bootstrap, exact McNemar, Friedman, Wilcoxon/Holm, and effect-size analyses.

> **Scientific boundary:** the quantum experiments use simulator/state-vector execution. The project does **not** claim hardware or computational quantum advantage, universal Quantum > Classical performance, or universal Hybrid > Classical superiority.

## Authors

- **Joshua Akowuje Weyori** — first and corresponding author
- Jonathan Nkrumah
- Richmond Ansah-Sarpong
- Ayerko Emmanuel Herson
- Isaac Baiden
- Bashiru Ibrahim Tare
- Yuonakuu Solomon Naabaseru
- Saeed Sulley Junior
- Courage Kwesi Ayidzoe
- George Owusu Aduenin
- William Joe Baasie

See `publication/merged_hqca_cqcs/AUTHORS_AND_CONTRIBUTIONS.md` for affiliations, contact emails, and the draft CRediT statement.

## Main scientific finding

The strongest result is **not** that a hybrid or quantum classifier universally wins.

Across ordinary reference/prospective tests, strong Classical and generic fusion baselines are difficult to improve. Under source/attack-family distribution shift, however, the relationship between the Classical and Quantum branches can change sharply. The manuscript formalizes this with:

```text
Gamma_Q(D) = P(Quantum correct | Classical wrong, D)

CQCS_Q(D_reference -> D_target)
  = Gamma_Q(D_target) / Gamma_Q(D_reference)
```

Held-out Web Attack shows a large expansion of conditional quantum rescue, while PortScan shows the opposite contraction. The key implication is that a fusion policy learned on development traffic can become stale after distribution shift.

## Publication package

The reviewer-facing reproducibility material is under:

`publication/merged_hqca_cqcs/`

It includes:

- the zero-day GPU notebook and execution notes;
- paired standard/prospective statistical comparisons;
- 10-seed zero-day model summaries;
- CQCS summaries;
- controlled-FPR results;
- paired statistical tests;
- split/source audits;
- runtime robust summaries;
- authorship/contribution metadata; and
- a pre-submission reviewer audit.

## Repository structure

```text
.
├── models/
├── qca/
├── preprocessing/
├── evaluation/
├── experiments/
├── tests/
├── results/
├── publication/
│   └── merged_hqca_cqcs/
├── project_config.py
└── ...
```

## Environment

Recommended: **Python 3.11**.

```powershell
py -3.11 -m venv QUANTUM
.\QUANTUM\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements-verified.txt
```

The later GPU validation experiments were executed on native Windows with PyTorch/CUDA and an NVIDIA RTX 3060 Laptop GPU. The custom four-qubit CUDA state-vector implementation was checked against a PennyLane CPU reference before use.

## Datasets

Raw benchmark datasets are intentionally **not** redistributed.

The project uses:

- UNSW-NB15
- CIC-IDS-2017
- ToN-IoT

Obtain them from their official academic providers and place them according to `DATASET_SETUP.md`.

## Reproducibility safeguards

The current merged-paper workflow includes:

- target/proxy leakage controls;
- preprocessing fitted only on training folds;
- out-of-fold branch predictions for fusion;
- development-only threshold selection;
- strong generic fusion controls;
- frozen prospective evaluation;
- pre-specified 10-seed zero-day conditions;
- prediction-level paired bootstrap and McNemar tests;
- repeated-seed Friedman/Wilcoxon/Holm analyses;
- explicit negative-result reporting;
- simulator/backend equivalence checking; and
- no post-hoc promotion of test-set ablations to the primary model.

## Citation

Use the metadata in `CITATION.cff`. A permanent archival DOI can be added after a tagged release is deposited to an archival service.

## License

MIT License.
