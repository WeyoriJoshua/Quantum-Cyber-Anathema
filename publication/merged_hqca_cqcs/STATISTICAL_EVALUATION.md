# Comparative statistical evaluation

The merged manuscript includes two complementary levels of statistical comparison.

## Standard/prospective prediction-level comparisons

Adaptive HQCA is compared with branch and fusion baselines using paired prediction-level analysis:

- bootstrap 95% confidence intervals for Delta MCC;
- exact McNemar tests on discordant classifications;
- explicit counts of HQCA-correct/comparator-wrong and HQCA-wrong/comparator-correct cases.

Key results:

- **UNSW-NB15:** HQCA vs Classical CAT Delta MCC = -0.02390, 95% CI [-0.03725, -0.01100], McNemar p = 3.75e-05. HQCA does not beat the classical branch.
- **UNSW-NB15:** HQCA vs Quantum CAT Delta MCC = +0.09989, 95% CI [0.08326, 0.11808], p = 7.55e-36.
- **Standard CIC-IDS-2017:** HQCA and Classical CAT have identical hard classifications; HQCA vs Quantum CAT Delta MCC = +0.16385, 95% CI [0.13139, 0.19661], p = 9.76e-25.
- **Prospective ToN-IoT:** HQCA vs Classical CAT Delta MCC = -0.00099, CI reaches 0, p = 1.0; HQCA vs RF stacking Delta MCC = -0.00194, 95% CI [-0.00706, 0.00311], p = 0.6875. HQCA remains statistically indistinguishable from the strongest classical/generic fusion controls while outperforming Quantum CAT.

These results establish the reference-regime baseline: **there is no universal hybrid gain**.

## Zero-day repeated-seed comparisons

For each held-out family, the paper reports:

- Friedman omnibus tests across six models;
- Kendall's W effect size;
- Holm adjustment across primary metrics;
- exact paired Wilcoxon tests;
- win/tie/loss counts;
- matched-pairs rank-biserial effect sizes;
- bootstrap confidence intervals;
- seed-wise CQCS direction stability.

For held-out **Web Attack**, the omnibus effect is large (MCC Kendall W about 0.756; balanced accuracy/F1 about 0.800; recall about 0.815), and Quantum CAT beats every comparator on MCC and recall in all 10 seeds. The manuscript still avoids a universal quantum-superiority claim because Classical CAT retains stronger ROC-AUC and higher post-hoc target-oracle MCC.

For **PortScan**, omnibus differences do not survive Holm correction; for **Infiltration**, significant differences are reported but interpretation is bounded by only 36 positive target examples.

## Interpretation

The statistical evidence supports a narrower claim than model superiority: **the ordering and complementary utility of the Classical, Quantum, and hybrid branches changes across regimes**. This is the empirical basis for CQCS and stationary-fusion failure.
