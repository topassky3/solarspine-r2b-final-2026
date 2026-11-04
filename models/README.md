# Locked SolarSpine R2B checkpoints (Git LFS)

The accepted, trained model weights are now included in this repository under Git LFS.

| Repository path | Required SHA256 |
|---|---|
| `models/detector/best.pt` | `bbb3c55175f68f80c213ac9fd6d1a1ab5bd63993696b609348d36f0472515875` |
| `models/refiner/best.pt` | `e4b9c3bacce816cd1d4dfb863622b93774743a6a401ae844088e33540882f397` |

After cloning, run `git lfs pull`, followed by `python scripts/preflight_release.py --strict`. Both weights are loaded for inference without retraining. Do not replace them with other checkpoints.

**Release note:** having weights in Git LFS does not by itself grant permission to redistribute third-party components. Confirm model/data licensing before making the repository public. The anonymous checkpoint download must be validated after publishing.
