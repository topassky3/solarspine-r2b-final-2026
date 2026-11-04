# SolarSpine R2B — standalone jury inference (pre-release)

This is a **packaging candidate**, not a certified public submission. It runs the accepted A3 YOLO26n-seg detector followed by the accepted R2A residual U-Net refinement using exactly seven source files and the original `r2b_replay_backend.py` copied byte-for-byte from the accepted repository. It does not read ground-truth masks, original CSVs, personal Google Drive, or private audit reports.

## Inputs

1. Official competition test JPEG directory, **180 images**; original filenames are retained as instance-ID stems.
2. A3 `best.pt`, SHA256 `bbb3c55175f68f80c213ac9fd6d1a1ab5bd63993696b609348d36f0472515875`.
3. R2A refiner directory containing `best.pt`, SHA256 `e4b9c3bacce816cd1d4dfb863622b93774743a6a401ae844088e33540882f397`.
4. A new writable output directory separate from the image and package trees.

Checkpoints and official images are **not included** in the source ZIP. Their authorized distribution/access must be arranged before final release. A `source/` directory containing the seven original production Python modules, the accepted source-lock JSON, and the original backend is mandatory; the ZIP builder copies and verifies them from the private source checkout, never fetches Drive or GT.

## Run

The recorded and internally replayed environment was Python 3.13.16, Tesla T4, CUDA 13.0, PyTorch 2.11.0+cu130, torchvision 0.26.0+cu130, Ultralytics 8.4.177 and the remaining versions in `requirements.txt`. An installation on a different CUDA/Python platform is **not** certified. Use an environment that already contains compatible CUDA PyTorch; never install an incompatible wheel merely to satisfy a label.

```bash
python delivery/r2b/run_inference.py \
  --images /input/test \
  --detector-weights /input/a3/best.pt \
  --refiner-dir /input/r2a \
  --output-dir /output/new_r2b_run
```

The parent process launches **two fresh interpreters**, one for each stage. The refiner consumes only new detector RLE records from the same run; accepted CSVs are never read. After each complete stage the program writes a CSV and locked manifest. The final CSV is `submission_r2b.csv` with `filament_id,segmentation_rle` and compressed COCO RLE. All masks must be 2048×2048, nonempty, round-trip losslessly and be disjoint within each image. No row is invented for an image without detections; its ID is in the manifest.

Outputs include `contract.json`, per-image image SHA and RLE records, stage `manifest.json`, `detector.csv`, `submission_r2b.csv`, and `summary.json`. Identical output files can be resumed; different outputs are protected and halt the run. A `--smoke --expected-images N` mode is diagnostic only and cannot be submitted.

The wrapper does not upload to Kaggle or make models public. It is a new entry point and must be validated against the **existing** accepted A6 and R2C CSVs in a controlled T4 run before the package can be called certified.

## Scope of evidence

The 2026-11-02 internal Colab replay did reproduce both accepted CSVs byte-for-byte (180 images, 1,422 instances, six empty images, zero overlap). It took 371.916 seconds of summed prediction time, excluding setup and CPU checks. This is evidence for the **old replay runner**, not yet for this jury runner. Consult `VALIDATION.md` for the new runner's status. Its recorded Torch GPU allocated-memory peaks (342.543 MiB / 117.761 MiB) exclude other CUDA driver memory.

## Licensing and visibility

The source repository is private. Review Ultralytics pretrained weights/software license, original dataset access/usage terms and all third-party notices before release. The organizers require a publicly accessible code repository and a runnable notebook, but neither the source nor checkpoints may be published without the author's explicit approval.
