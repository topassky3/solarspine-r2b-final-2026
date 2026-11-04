# Third-party notices — mandatory review before public release

This file does **not** grant redistribution rights for data, pretrained weights, or third-party software. The original research repo carries an MIT license for its own code, but the terms of included/dependent materials must be checked separately:

- **Ultralytics YOLO26n-seg pretrained components and runtime:** identify the exact weight source, its licensing conditions and any corresponding notices/requirements; do not assume that the original project's MIT license overrides Ultralytics licensing.
- **GONG / MAGFiLO competition images and annotations:** do not redistribute official TEST images, private labels, or training archives. The official jury should use the competition-provided dataset.
- **PyTorch, torchvision, SciPy, Pillow, pycocotools, NumPy, OpenCV:** use each project's corresponding software licenses and versioned package distribution.
- **Refiner checkpoint:** owner-generated weights; confirm that training data and upstream components permit release.

The organizer requires an openly accessible Git repository, with weights stored and loaded from that repository. Obtain explicit owner approval and complete this license review before making it public.
