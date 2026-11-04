# Evidence summary — as of 2026-11-03

## Competition-facing facts

- Candidate: **R2B** — A3 YOLO26n-seg detector plus residual source-conditioned native-ROI U-Net (R2A).
- Kaggle public-score display: **0.29** for both the earlier `submission_e10_a11r2c.csv` and the exact re-upload `SolarSpine_R2B_ENTREGA_KAGGLE.csv`, as shown by the participant on 2026-11-03. Original A6 detector displays **0.24**.
- No additional improvement came from re-uploading the exact same CSV. Kaggle scores displayed to two decimals are not accurate to more digits.
- Development Fold4 strict micro-PQ: A6 **0.2935407215569436**; R2B **0.3481529764938783**. Fold4 was observed repeatedly and is **not blind**.
- Matched Fold4 one-to-one TP/FP/FN: A6 791/997/847; R2B 897/891/741. Both had 1,788 proposals on 231 annotation records.
- R23 topology study: `R23_NO_GO__KEEP_CONTROLS`; not part of the accepted solution.

## Reproduction

Independent historical Colab T4 replay on 2026-11-02: 180 images, 1,422 predicted instances in each stage, six zero-instance images, zero overlaps. Both regenerated CSVs were byte-identical to their accepted references.

A **separate jury-package** Colab GPU run on 2026-11-03 reported `NEW_JURY_RUN_EXACT_MATCH__REPORT_AND_RELEASE_PENDING` in the user's `jury_validation.json`:

| Output | Reproduced SHA256 |
|---|---|
| Detector CSV | `347307dc95727dbd9ea97d9d474c2125eb5bb14515e79c8bce563e51ead61a39` |
| Final R2B CSV | `c241bda04c1fefdab5fe1b6b6d90f4aeac23a02a2003f830c0b9b1b289d3294f` |

The refreshed **public-repository notebook** in this package changes only the setup/bootstrap path (clone the release repo instead of consuming a source ZIP). It has not yet received a new independent T4 run. It invokes the same byte-preserved inference runner, so a fresh external-clone smoke/full execution remains the final distribution check.

Checkpoints required (SHA256):
- A3 detector: `bbb3c55175f68f80c213ac9fd6d1a1ab5bd63993696b609348d36f0472515875`.
- R2A refiner: `e4b9c3bacce816cd1d4dfb863622b93774743a6a401ae844088e33540882f397`.

No Kaggle private test labels were inspected. The replay does not provide new PQ or final leaderboard position. Public access of the code and weights and the official four-page Overleaf report remain outstanding.
