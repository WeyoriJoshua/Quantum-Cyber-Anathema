# Quantum Cyber Anathema Repository Audit

Audit date: 2026-09-11

## Scope

This audit evaluates the repository as a reproducibility archive for the empirical Quantum Cyber Anathema (QCA) study. It does not constitute a fresh rerun of the full CICIDS2017 and UNSW-NB15 experiments, which require third-party raw datasets and substantial compute time.

## Release decision

**PASS — repository structure and publication package are suitable for public release, subject to the authors choosing when to change repository visibility and create a tagged release.**

## Verified publication components

- `README.md` describes the scientific scope, authors, experimental stages, frozen QCA-v2 settings, dataset layout, execution order, safeguards, testing, results, citation and license.
- `LICENSE` contains the MIT License approved by the authors.
- `CITATION.cff` lists Joshua Akowuje Weyori and Peter Nimbe as authors.
- `requirements.txt`, `requirements-quantum.txt` and `requirements-verified.txt` document the software environment.
- `project_config.py` records the publication configuration, fixed seeds and frozen external-validation settings.
- Classical, PennyLane, Qiskit, weighted-VQC, QCA-v1 and QCA-v2 implementation modules are present.
- CICIDS2017 preprocessing, matched benchmarking, multi-seed evaluation and QCA-v2 development runners are present.
- UNSW-NB15 external benchmark preparation and external-validation runners are present.
- Protocol tests cover preprocessing, matched splits, multi-seed statistics, QCA-v2 development guards and UNSW external-validation guards.
- Compact publication-facing CSV results are retained under `results/`.
- Raw CICIDS2017 and UNSW-NB15 datasets are not redistributed.
- Large model checkpoints, virtual environments, caches and bulky generated artifacts are excluded from the publication archive.

## Technical audit results

The publication source snapshot used for release preparation was checked with:

```text
python -m compileall -q .
pytest -q
```

Result:

```text
21 passed, 2 skipped
```

The two skipped tests are expected dataset-dependent UNSW checks. They are designed to run after the user locally prepares the non-distributed UNSW benchmark artifacts.

A source scan found no embedded GitHub tokens, obvious API-key/password assignments, or absolute Windows drive paths in the publication source.

## Methodological safeguards represented in the repository

The repository documents and implements the following controls:

- canonical binary labels (`0 = benign`, `1 = attack`);
- train-only fitting of preprocessing transformations;
- duplicate and contradictory-label controls;
- matched observations for headline comparisons;
- validation-only threshold and model selection;
- fixed five-seed protocol (`42, 123, 2026, 7, 99`);
- architecture-matched Static VQC control;
- explicit QCA ablations;
- validation-only QCA-v2 development;
- frozen QCA-v2 configuration before UNSW-NB15 evaluation;
- exclusion of the external test split from model fitting, early stopping and threshold selection;
- explicit statement that the study does not establish quantum computational or predictive advantage.

## Results retained for auditability

The repository includes compact result tables for:

- CICIDS2017 five-seed model metrics and mean/SD/95% CI summaries;
- paired QCA-versus-Static-VQC comparisons;
- QCA-v2 development-stage variant ranking and paired comparisons;
- UNSW-NB15 external descriptive ranking;
- frozen QCA-v2 external paired comparisons.

These tables preserve both positive development-stage findings and the negative/conditional external-validation result.

## Non-blocking limitations of this repository audit

1. The full experiments were not retrained during this repository audit because the raw benchmark datasets are intentionally not distributed in the repository and quantum-model training is computationally expensive.
2. The two dataset-dependent UNSW tests require locally generated benchmark artifacts and are therefore skipped in a clean source-only environment.
3. A permanent DOI is not yet assigned. A tagged public release can later be archived through a service such as Zenodo.

## Recommended release sequence

1. Keep the repository private until the corresponding manuscript metadata are final.
2. Change repository visibility to public when ready for submission/publication.
3. Create a tagged release such as `v1.0.0`.
4. Archive that release and obtain a DOI if desired.
5. Replace any provisional software-availability wording in the manuscript with the public repository URL and, when available, the DOI.

## Audit conclusion

The repository contains the required source code, protocol documentation, citation metadata, license, tests and compact result summaries for a reviewer-facing QCA reproducibility package. No blocking repository-structure issue was identified in this audit.
