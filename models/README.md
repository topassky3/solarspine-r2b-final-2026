# Locked checkpoints — NOT INCLUDED IN PRE-RELEASE ARCHIVE

The final public jury repository **must** contain:

| Path in public Git repo (using Git LFS) | Expected SHA256 |
|---|---|
| `models/detector/best.pt` | `bbb3c55175f68f80c213ac9fd6d1a1ab5bd63993696b609348d36f0472515875` |
| `models/refiner/best.pt` | `e4b9c3bacce816cd1d4dfb863622b93774743a6a401ae844088e33540882f397` |

Both checkpoints were created/verified in the private research Drive. They are omitted here to avoid unapproved redistribution. **Do not substitute other `.pt` files.** Check that Git LFS bandwidth/storage allows the jury to download weights unauthenticated.

Model files are consumed directly, without retraining. `scripts/preflight_release.py --strict` refuses to pass until both are present and their hashes are exact.
