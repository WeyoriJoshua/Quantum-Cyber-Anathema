# Reviewer audit and revision record

## Overall assessment

The merged manuscript has a strong publishable core, but the pre-revision version was vulnerable to rejection for four main reasons: (1) literature coverage was too compressed for the breadth of the novelty claim; (2) the paper could be read as another hybrid-QML accuracy comparison; (3) inferential statistics were concentrated in the zero-day section; and (4) several deployment and validity limitations required more explicit treatment.

## Major concerns addressed

1. **Novelty positioning.** Reframed the contribution from universal hybrid/quantum superiority to nonstationary heterogeneous branch utility under distribution shift. CQCS is presented as a conditional branch-utility framework, not as metric proliferation.
2. **Literature review.** Added recent 2024–2026 systematic reviews, QML-IDS, low-latency hybrid QML, zero-day/OOD target-FPR evaluation, temporal-shift audits, calibration, and concept-drift literature.
3. **Comparative statistical evaluation.** Added paired bootstrap confidence intervals and exact McNemar tests for UNSW-NB15, standard CIC-IDS-2017, and prospective ToN-IoT, alongside the existing zero-day Friedman/Wilcoxon/Holm/effect-size analyses.
4. **Comparator scope.** Added an explicit limitation that the included models test branch interaction and fusion stationarity rather than constitute an exhaustive SOTA IDS leaderboard.
5. **Validation terminology.** Replaced overstrong independent-confirmation language for CIC-IDS-2017 with “pre-specified zero-day validation,” since CIC-IDS-2017 was already used in the standard benchmark phase.
6. **Runtime reporting.** Replaced mean±SD by median [IQR] because one Infiltration wall-clock run is an outlier; the raw run is retained in the archive.
7. **Authorship.** Joshua Akowuje Weyori remains first/corresponding author. All supplied co-authors and affiliations are included. Draft CRediT roles must be confirmed by all authors before submission.

## Claims intentionally not made

- No hardware or computational quantum advantage.
- No universal Quantum > Classical or Hybrid > Classical ordering.
- No claim that CQCS is architecture-independent.
- No claim that CIC-IDS-2017 represents current production traffic.
- No claim that Adaptive HQCA is the best available intrusion detector.
