# SolarSpine R2B — Solar Filament Instance Segmentation

**Solar Filament Segmentation Challenge 2026 · Reproducible competition submission**

[![Python](https://img.shields.io/badge/Python-3.13-informational)](https://www.python.org/)
[![Inference](https://img.shields.io/badge/Inference-Tesla%20T4-blue)](#environment)
[![Public%20PQ](https://img.shields.io/badge/Kaggle%20public%20score-0.29-blue)](#results)

SolarSpine segments individual solar filaments from **2048 × 2048 GONG H-alpha images**. It combines a trained YOLO26n-seg instance detector with a native-resolution residual U-Net that refines the geometry of each proposed filament without introducing ground-truth information at inference time.

> **Release status (4 Nov 2026):** the frozen inference code and both SHA256-verified trained checkpoints are present in this private release repository (weights via Git LFS). An authenticated Colab QA run reproduced the accepted submission CSV byte-for-byte. Before public submission, complete third-party licensing review and verify anonymous Git clone, Git LFS download and a fresh jury execution without tokens.

## Method

```text
Official H-alpha JPEG (2048 × 2048)
       │
       ▼
YOLO26n-seg · 1024 inference · confidence 0.25
       │
       ▼
A6: exclusive source masks (no overlapping instances)
       │
       ▼
Three source-only 256 × 256 ROI tiles per candidate
       │
       ▼
Residual U-Net · H-alpha + source mask + neighbor mask
       │
       ▼
Source-preserving merge + deterministic first-owner rule
       │
       ▼
Native COCO RLE → submission_r2b.csv
```

The actual frozen implementation is `delivery/r2b/run_inference.py`, `delivery/r2b/source/scripts/r2b_replay_backend.py` and **seven SHA256-locked original source modules**. All inference and mask operators are preserved from the accepted experiment; the source lock is verified before execution. The detector and U-Net run in separate fresh Python processes. TEST labels and previously accepted CSVs are **not** input to inference.

## Results

| Measure | Detector A6 | Detector + U-Net R2B |
|---|---:|---:|
| Strict micro-PQ, exposed Fold4 development | 0.29354 | **0.34815** |
| True positives | 791 | 897 |
| False positives | 997 | 891 |
| False negatives | 847 | 741 |
| Kaggle public score (reported 3 Nov 2026) | 0.24 | **0.29** |

**Interpretation:** Fold4 contains 231 repeatedly observed annotation records, so its +0.05461 PQ change is a *development* comparison, not an independent estimate of leaderboard gain. The Kaggle public score is displayed rounded to two decimals and is **not** an exact private or final score. The newer R23 topology experiment failed its preregistered promotion gates and was **not** used in this submission.

A new jury-facing full inference run on Tesla T4 regenerated both stage outputs with exact reference hashes: detector `347307dc95727dbd9ea97d9d474c2125eb5bb14515e79c8bce563e51ead61a39`, refined output `c241bda04c1fefdab5fe1b6b6d90f4aeac23a02a2003f830c0b9b1b289d3294f`. This is a reproducibility check; it is **not** new training or a new public PQ measurement. See [VALIDATION](delivery/r2b/VALIDATION.md) and [Evidence](docs/EVIDENCE.md).

## Get started

1. Obtain the official competition **TEST JPEGs** from the organizers, not from this repository.
2. Clone this repository and download its two SHA256-locked trained checkpoints using `git lfs pull`; no separate model download or training is needed.
3. Use the recorded Tesla T4 / CUDA 13.0 runtime and match dependency versions.
4. Open [SolarSpine_R2B_Jurado_T4.ipynb](notebooks/SolarSpine_R2B_Jurado_T4.ipynb) on Colab with a matching Tesla T4 runtime, supply the official TEST JPEGs at the documented `IMAGES` location, and run all cells. Once the repository is public, the notebook clones it anonymously, downloads weights with Git LFS and verifies the produced CSV. **No personal Drive, private checkout or separate code ZIP is required**, but the competition TEST images must be supplied by the evaluator.

Command-line equivalent from the repository root:

```bash
python scripts/preflight_release.py --strict
python delivery/r2b/run_inference.py \
  --images /path/to/official/test_images \
  --detector-weights models/detector/best.pt \
  --refiner-dir models/refiner \
  --output-dir /tmp/solarspine_r2b_new_run
```

The resulting `submission_r2b.csv` contains the exact columns `filament_id,segmentation_rle` and original image-ID-based instance IDs. It has one row per predicted filament and no rows for images with zero detected filaments. The inferred masks are 2048 × 2048 compressed COCO RLE. On the accepted 180-image TEST roster, 1,422 instance rows and six empty images were obtained; these are descriptive facts about the frozen pipeline, not a hardcoded answer.

## Environment

The **observed and tested** configuration is Python 3.13.16, NVIDIA Tesla T4, CUDA 13.0, PyTorch 2.11.0+cu130, torchvision 0.26.0+cu130, Ultralytics 8.4.177, NumPy 2.1.3, Pillow 11.3.0, pycocotools 2.0.11, SciPy 1.16.3, and OpenCV-Python 4.12.0.88. See [`requirements.txt`](requirements.txt) for exact versions. PyTorch wheels are platform/CUDA-specific; the notebook checks the existing runtime and refuses a different stack rather than silently altering the scientific pipeline.

Observed internal T4 replay prediction time: 371.916 seconds summed across stages, excluding installation, setup and audit. It is not total notebook wall time.

## Repository map

| Path | Purpose |
|---|---|
| [`notebooks/SolarSpine_R2B_Jurado_T4.ipynb`](notebooks/SolarSpine_R2B_Jurado_T4.ipynb) | Jury notebook, clones this public repo directly |
| [`delivery/r2b/run_inference.py`](delivery/r2b/run_inference.py) | Standalone two-stage runner |
| [`delivery/r2b/source/`](delivery/r2b/source) | Frozen model/mask source and SHA lock |
| [`models/`](models/) | Authorized, hash-verified checkpoints (**included through Git LFS; SHA256 checked**) |
| [`scripts/preflight_release.py`](scripts/preflight_release.py) | Source/weight integrity & release gate |
| [`docs/EVIDENCE.md`](docs/EVIDENCE.md) | Reproducibility claims and limits |
| [`docs/RELEASE_CHECKLIST.md`](docs/RELEASE_CHECKLIST.md) | Remaining conditions before release |
| [`tests/`](tests/) | CPU release-structure regression tests |

## Availability, provenance and licensing

Both accepted model checkpoints are included via Git LFS, with hashes verified by `python scripts/preflight_release.py --strict`. The official competition TEST JPEGs are **not** redistributed; evaluators must obtain them from the competition source. While this repository is private, the owner has tested authenticated cloning and exact-CSV reproduction; **anonymous GitHub and LFS access remain unverified until public release**. Never include private Drive IDs, TEST labels, secret tokens, or experiment backups.

Copyright 2026 Juan Felipe Orozco Cortés. The original project contains an MIT license for original code, but **third-party weights, Ultralytics and dataset licenses may impose additional conditions**. Read [`THIRD_PARTY_NOTICE.md`](THIRD_PARTY_NOTICE.md) before publishing. The public repository should be approved by its owner, not created by exposing the full private research repository.
