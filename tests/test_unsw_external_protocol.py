import json

import numpy as np

from project_config import CONFIG


def test_unsw_benchmark_shapes_and_classes():
    path = CONFIG.unsw_benchmark_splits_path
    assert path.exists()
    with np.load(path, allow_pickle=False) as d:
        assert d["X_train"].shape == (1000, 4)
        assert d["X_val"].shape == (300, 4)
        assert d["X_test"].shape == (300, 4)
        for key in ("y_train", "y_val", "y_test"):
            assert set(np.unique(d[key])) == {0, 1}


def test_unsw_preprocessing_guards():
    path = CONFIG.unsw_preprocessed_dir / "preprocessing_report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    guards = report["methodological_guards"]
    assert guards["official_test_never_used_for_training_or_validation"] is True
    assert guards["attack_cat_excluded_from_features"] is True
    assert guards["id_excluded_from_features"] is True
    assert guards["official_test_rows_overlapping_training_features_removed_before_sampling"] is True
    assert guards["preprocessing_fit_on_benchmark_train_only"] is True
    assert report["benchmark"]["feature_overlap_between_benchmark_partitions"] == 0


def test_frozen_qca_v2_external_settings():
    assert CONFIG.unsw_qca_v2_variant == "QCA_MarginHardReplay"
    assert CONFIG.unsw_qca_v2_margin_lambda == 0.50
    assert CONFIG.unsw_qca_v2_probability_margin == 0.15
    assert CONFIG.unsw_qca_v2_boundary_weight == 1.00
    assert CONFIG.unsw_qca_v2_boundary_threshold == 0.50
