# SolarSpine R2B — third-party license and dataset notices

**Scope:** the separate jury/inference release `topassky3/solarspine-r2b-final-2026`.
The private research repository is **not** included in this release.

## 1. Ultralytics YOLO26 (detector and runtime)

- Runtime: `ultralytics==8.4.177` (installed as a dependency; not vendored).
- Fine-tuned detector: `models/detector/best.pt`, delivered by Git LFS.
- Its serialized checkpoint records: `license = AGPL-3.0 (https://ultralytics.com/license)`, Ultralytics version `8.4.177`, model architecture `yolo26n-seg`, and an upstream `yolo26n-seg.pt` file in an `Ultralytics/YOLO26` Hugging Face cache path.
- The checkpoint's training configuration records `epochs=44`, `imgsz=1024`, `pretrained=True`, `seed=2026`, and an AdamW optimizer; these are **recorded configuration values**, not independent proof that every epoch completed.

**Licensing:** Ultralytics describes its source code and models as licensed by default under AGPL-3.0, with a separate Enterprise option. We intend to publish the integrated open-source release under AGPL-3.0. Ultralytics retains ownership of its original components; the repository's original MIT notice does not override Ultralytics terms.

- Official Ultralytics guidance: https://docs.ultralytics.com/help/contributing/
- Ultralytics licensing: https://www.ultralytics.com/license
- AGPL-3.0 text: https://www.gnu.org/licenses/agpl-3.0.txt

## 2. SolarSpine original source

Original SolarSpine source code was previously distributed under MIT by its author. `LICENSE-MIT-ORIGINAL` preserves that grant and attribution. The **integrated release** containing the Ultralytics model is distributed under the AGPL-3.0 terms described in `LICENSE`, without removing third-party notices. This does not purport to relicense any third-party material independently.

## 3. Competition data — NOT REDISTRIBUTED

The **Solar Filament Segmentation Challenge 2026** rules allow competition-data use for competition participation and non-commercial academic research/education. That is not a blanket license to republish images, masks, TEST labels, or archives or to authorize commercial exploitation of those data.

- No official H-alpha image, TEST image, GT mask, dataset archive, or competition annotation is included in this repository.
- Evaluators obtain the 180 official TEST JPEGs from Kaggle and supply them separately to the notebook.
- The trained **refiner** was developed from the competition training material; whether its weights can be openly redistributed under the data terms is **not established by the checkpoint hash** and should be confirmed with the competition organizer if uncertain.

Official competition rules: https://www.kaggle.com/competitions/filament-segmentation-2026/rules

## 4. Other libraries

PyTorch, torchvision, NumPy, SciPy, OpenCV, Pillow, and pycocotools are installed separately using their pinned versions. Their respective upstream licenses and copyright notices remain in force; this repository does not claim authorship over them.

## Publication check

Before changing repository visibility, confirm: (1) you accept distributing the integrated release under AGPL-3.0, (2) trained checkpoint redistribution is permitted under applicable dataset and upstream terms, and (3) the repository contains no sensitive files or secrets. Public access and public Git LFS must be checked anonymously after publishing. This document is a provenance summary, not legal advice.
