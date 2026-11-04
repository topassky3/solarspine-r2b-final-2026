# R2B jury package validation — 2026-11-03

| Gate | Result | Scope |
|---|---|---|
| Seven original inference modules and source hash locks | PASS | Bytes match immutable lock |
| New independent jury package on Colab Tesla T4 | **PASS** | `NEW_JURY_RUN_EXACT_MATCH__REPORT_AND_RELEASE_PENDING` |
| Detector CSV SHA | **PASS** | `347307dc95727dbd9ea97d9d474c2125eb5bb14515e79c8bce563e51ead61a39` |
| Refined CSV SHA | **PASS** | `c241bda04c1fefdab5fe1b6b6d90f4aeac23a02a2003f830c0b9b1b289d3294f` |
| Original historical Colab replay | PASS | 180 images, 1422 masks, 6 empty, 0 overlap |
| CPU check of clean public repo layout | To be run in this distribution | Does not substitute GPU inference |
| Public Git clone and two weights | **PENDING** | User authorization/licensing/access needed |
| Fresh clone-based notebook (rather than source-ZIP notebook) GPU run | **PENDING** | Bootstrap changed; prediction code unchanged |
| Four-page PDF in organizer's Overleaf template | **PENDING** | Earlier draft was not the official template |
| Final Google Form | **NOT SUBMITTED** | Kaggle R2B CSV already uploaded separately |

The two original code/model branches are unchanged. This is a distribution and documentation cleanup, **not** a new trained model or improved leaderboard score.
