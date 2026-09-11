# Dataset Setup

Raw benchmark datasets are not redistributed in this repository.

## CICIDS2017

Place the MachineLearningCVE CSV files under:

```text
datasets/
└── CIC-IDS-2017/
    └── CSV/
        └── MachineLearningCSV/
            └── MachineLearningCVE/
```

The CICIDS2017 preprocessing pipeline converts the multiclass attack labels to the binary target used in the empirical study:

```text
0 = benign
1 = attack
```

The pipeline controls duplicate feature-label pairs and contradictory exact-feature groups before splitting, then fits imputation/encoding/scaling/PCA only on the training partition.

## UNSW-NB15

Place:

```text
datasets/
└── UNSW-NB15/
    ├── UNSW_NB15_training-set.csv
    └── UNSW_NB15_testing-set.csv
```

The external-validation preparation script intentionally excludes:

```text
id
attack_cat
```

`id` is a record identifier. `attack_cat` is an attack-family annotation adjacent to the binary prediction target and is therefore excluded to avoid target leakage.

The script also removes exact feature overlap from the official test pool when that observation already occurs in the cleaned official training pool before selecting the matched external benchmark.

## Data licensing

Users are responsible for obtaining the datasets from their official or otherwise authorized distribution sources and complying with the applicable dataset terms. This repository contains preprocessing and evaluation code only.
