# Literature positioning for the merged HQCA–CQCS manuscript

The manuscript is positioned against recent work in three adjacent areas:

1. **Hybrid/QML intrusion detection** — studies such as QML-IDS and QATNet show that quantum or hybrid representations can be competitive, but they primarily evaluate model-level predictive performance.
2. **QML evidence maturity and reproducibility** — recent systematic reviews report heavy simulator dependence, heterogeneous evaluation protocols, dataset-dependent gains, and incomplete reporting of false positives, statistical significance, cost, and hardware feasibility.
3. **Distribution shift and operational zero-day evaluation** — recent IDS studies show that fixed operating points, calibration, false-alarm budgets, and temporal/attack-family shift can materially change deployment behavior.

## Selected recent comparison set

| Study | Main focus | Shift/zero-day | Statistical/operational focus | Gap relative to this work |
|---|---|---|---|---|
| Abreu et al. (2024), QML-IDS, DOI 10.1109/ISCC61673.2024.10733655 | Hybrid/QML IDS across public datasets | Limited | Model-level accuracy/classification | Does not formalize development-to-target branch complementarity |
| Kadi et al. (2025), DOI 10.1109/OJCOMS.2025.3537957 | Quantum/classical encoding comparison | Limited | Encoding choice and predictive performance | No CQCS or fusion-stationarity test |
| Kaissar et al. (2026), Future Internet 18(5):234 | Systematic survey of QML IDS | Review | Highlights simulator/legacy-dataset/evaluation inconsistency | Motivates standardized, shift-aware validation |
| Bernard et al. (2026), DOI 10.1186/s43067-026-00390-6 | Systematic review of hybrid quantum–classical security | Review | Evidence maturity, cost, statistical-reporting gaps | Supports bounded rather than universal quantum claims |
| Louise & Kavitha (2026), DOI 10.1109/AIMLA67915.2026.11522502 | Systematic review of hybrid QML cybersecurity | Review | Notes simulator dependence, latency, scaling/training barriers | Motivates explicit operational/statistical safeguards |
| Danasabe et al. (2026), QATNet, DOI 10.1109/TQE.2026.3674396 | Low-latency hybrid QML IDS | No dedicated branch-shift test | Modern datasets, noise simulation, latency | Does not test nonstationary branch utility |
| Al Malwi et al. (2026), DOI 10.32604/cmc.2026.075326 | Quantum-enhanced zero-day IDS | Yes | Held-out CICIDS attack families | Quantum signal usefulness, but not CQCS or stationary-fusion failure |
| Ha & Kim (2026), DOI 10.3390/app16052284 | Operational zero-day IDS under target-FPR calibration | Yes | Fixed operating-point transfer and false-alarm budgets | Motivates our controlled-FPR and threshold-transfer analysis |
| Ahmed et al. (2026), DOI 10.3389/frai.2026.1928858 | Temporal distribution shift in IDS | Yes | Monitoring/adaptation failure under year-to-year shift | Establishes the importance of deployment-time nonstationarity |
| Alalawi (2026), DOI 10.14569/IJACSA.2026.0170812 | IDS calibration under distribution shift | Yes | Calibration, cost, leave-one-attack-out | Supports separating ranking, calibration, and operating-point behavior |

## Novelty boundary

The paper does **not** claim that hybrid or quantum IDS models are generally superior. Its distinctive contribution is to make the *relationship between heterogeneous branches* the object of analysis:

`Gamma_Q(D) = P(Q correct | C wrong, D)`

and

`CQCS_Q(D_r -> D_t) = Gamma_Q(D_t) / Gamma_Q(D_r)`.

The study then connects this branch-utility shift to a concrete system-level failure mode: a fusion policy optimized on development traffic can become stale when the deployment distribution changes.
