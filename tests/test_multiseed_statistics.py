import numpy as np
import pandas as pd

from run_multiseed_benchmark import (
    exact_sign_flip_pvalue,
    summarize_metrics,
    paired_qca_vqc_comparison,
)


def test_exact_sign_flip_all_zero():
    assert exact_sign_flip_pvalue([0, 0, 0, 0, 0]) == 1.0


def test_summary_has_mean_sd_ci():
    df = pd.DataFrame(
        {
            "Seed": [1, 2, 3, 4, 5],
            "Model": ["QCA"] * 5,
            "F1": [0.4, 0.5, 0.6, 0.5, 0.5],
        }
    )
    out = summarize_metrics(df)
    row = out[(out["Model"] == "QCA") & (out["Metric"] == "F1")].iloc[0]
    assert row["N Seeds"] == 5
    assert np.isclose(row["Mean"], 0.5)
    assert np.isfinite(row["SD"])
    assert np.isfinite(row["95% CI Lower"])
    assert np.isfinite(row["95% CI Upper"])


def test_paired_qca_vqc_comparison():
    rows = []
    for seed, vqc, qca in zip(
        [42, 123, 2026, 7, 99],
        [0.40, 0.42, 0.41, 0.43, 0.39],
        [0.45, 0.46, 0.44, 0.47, 0.43],
    ):
        rows.append({"Seed": seed, "Model": "PennyLaneVQC_Angle", "F1": vqc})
        rows.append({"Seed": seed, "Model": "QCA", "F1": qca})
    out = paired_qca_vqc_comparison(pd.DataFrame(rows))
    row = out[out["Metric"] == "F1"].iloc[0]
    assert row["N Paired Seeds"] == 5
    assert row["QCA Wins"] == 5
    assert row["Mean Difference QCA-VQC"] > 0
