# Accepted source identity

`source/accepted_sources.lock.json` is the original `experiments/R2B-inference-replay/accepted_sources.lock.json` file from PR #131. The complete seven files under `source/src/...` are byte-for-byte copies from the same repository. `source/scripts/r2b_replay_backend.py` is likewise copied unchanged. The wrapper checks the seven files using their exact SHA256 before importing the backend, and checks both checkpoints by SHA256. All paths are verified relative to the distributed source tree. No source changes are authorized as part of packaging.

Build a distribution from the authenticated repository checkout using `scripts/build_r2b_jury_package.py`. No jury-side private clone is needed **once the resulting ZIP is distributed**. Do not include checkpoint bytes, Kaggle images, annotations or personal Google Drive tokens in the ZIP.
