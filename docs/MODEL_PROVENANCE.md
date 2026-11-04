# SolarSpine R2B — model and training provenance

This document distinguishes **checkpoint-recorded settings** from **independently demonstrated properties** and missing training materials. No weights or inference code were modified to prepare these notes.

## Detector (A3 / YOLO26n-seg)

| Property | Recorded or verified value | Evidence level |
|---|---|---|
| Final weights | `models/detector/best.pt` | Git LFS + SHA256 verification |
| SHA256 | `bbb3c55175f68f80c213ac9fd6d1a1ab5bd63993696b609348d36f0472515875` | Verified |
| Architecture | YOLO26n-seg | Checkpoint metadata |
| Upstream initialization | `yolo26n-seg.pt`, Ultralytics/YOLO26 Hugging Face snapshot cache | Checkpoint path metadata; snapshot retrieval not independently reconstructed |
| Ultralytics version | `8.4.177` | Checkpoint metadata / frozen runtime |
| Recorded license | `AGPL-3.0` | Checkpoint metadata; consult upstream license terms |
| Training run date | `2026-10-19T15:56:05.437073+00:00` | Checkpoint timestamp |
| Training configuration | 44 epochs configured; 1024 image size; AdamW; seed 2026; pretrained=True | Checkpoint train_args; **not** evidence every configured epoch completed |
| Data configuration | `solarspine_e10_a3_local/data.yaml` | Filename recorded in checkpoint; contents not shipped |

## Native ROI refiner (R2A residual U-Net)

| Property | Value | Evidence level |
|---|---|---|
| Final weights | `models/refiner/best.pt` | Git LFS + SHA256 verification |
| SHA256 | `e4b9c3bacce816cd1d4dfb863622b93774743a6a401ae844088e33540882f397` | Verified |
| Architecture | `NativeROIRefiner` | `models/refiner/training_contract.json` |
| Training tiles | 2,955 | Contract-recorded |
| Validation tiles | 870 | Contract-recorded |
| Epoch budget | 8 | Contract-recorded |
| Training loss | `R1Y_masked_loss_0.5_BCE_0.5_Dice` | Contract-recorded |
| Learning rate | 0.0003 | Contract-recorded |
| Seed | 20261021 | Contract-recorded |
| TEST GT use | Contract declares `Kaggle_TEST_unused=true` | Recorded assertion; not independently audited here |

The training contract names private experiment `E10-01-A11R2A-FULL-NATIVE-OOF-REFINER-T4-TRAIN`, source Git commit `e9b0528d9e6893cd1332bbe8bbaaf0366f4894c7`, and provenance hashes for the patch index and protocol. Exact input training archives and private paths are intentionally not distributed.

## Reproduction boundary

A verified Colab Tesla T4 **inference** run regenerated the previously accepted CSV with SHA256:

`c241bda04c1fefdab5fe1b6b6d90f4aeac23a02a2003f830c0b9b1b289d3294f`

This proves byte-identical **inference**, not independent recreation of both models from scratch. The repo currently includes the frozen inference source and the two final trained checkpoints, **not a complete independent training recreation package**. Under the competition rules, winning entries must furnish training code and documentation; recover and independently validate the original trainer, its config and data-source instructions before claiming full training reproducibility. Do not invent a trainer from these metadata.

Official rules: https://www.kaggle.com/competitions/filament-segmentation-2026/rules
