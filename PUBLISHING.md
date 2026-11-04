# SolarSpine R2B — public release gate (owner controlled)

**Only** `topassky3/solarspine-r2b-final-2026` is the release target. Keep `topassky3/solarspine-pq` private. Do not publish TEST data, GT masks, backup ZIPs, tokens, or unreviewed research experiments.

## Verified before publication (4 Nov 2026)

- Release HEAD before this licensing patch: `e49db08`.
- Source and both trained checkpoints passed the strict SHA256 preflight; six tests passed on Windows.
- Authenticated Colab T4 inference regenerated the accepted CSV byte-for-byte (`c241bda04c1fefdab5fe1b6b6d90f4aeac23a02a2003f830c0b9b1b289d3294f`).
- `notebooks/SolarSpine_R2B_Jurado_T4.ipynb` is the public, token-free notebook, already in GitHub.

## Gate A — complete legal/provenance review BEFORE PUBLIC

1. Read `THIRD_PARTY_NOTICE.md` and `docs/MODEL_PROVENANCE.md`.
2. Affirmatively choose an AGPL-3.0-integrated release and install the authentic full `LICENSE` text using `scripts/Install-AGPLLicense.ps1`. The script saves the historical original MIT grant in `LICENSE-MIT-ORIGINAL`.
3. Confirm with the competition organizer if necessary that redistribution of the trained weights derived from competition data is permitted. No dataset images are distributed.
4. Check that the exact source material and weight bytes, rather than only their hashes, are present in Git LFS; review *Git history* for secret leaks before publishing (visibility changes expose public Git history).
5. Address training-code obligations: the verified public notebook fully reproduces **inference**, not the training from scratch; competition winners must provide training code. See `docs/MODEL_PROVENANCE.md`.
6. Run `python scripts/preflight_release.py --strict`, six regression tests, and `python scripts/check_publication_ready.py`.
7. Review `git diff` and `git diff --cached` before committing and pushing.

## Gate B — public access (ONLY AFTER gate A)

1. Owner changes visibility of **release repo only** on GitHub (Settings → General → Danger Zone → Change repository visibility → Public). Do not change the research repo.
2. In **a new clean environment with no GitHub token**, run `git clone https://github.com/topassky3/solarspine-r2b-final-2026.git`, then `git lfs pull` and the strict preflight.
3. Execute public jury notebook with the official 180 TEST JPEGs, and verify `SOLARSPINE_R2B_JURY_CSV_EXACT_MATCH` with the hash above.
4. Audit visible pages: GitHub repo, LFS download as anonymous user, README, licensing, source and download link. Keep the evidence logs.

## Gate C — final competition submission

1. Validate four-page technical report (Overleaf template) and make sure it correctly describes the frozen accepted method.
2. Complete the organizers' Google Form with report PDF and genuinely public GitHub URL by the applicable deadline. The notebook never submits the form or Kaggle submission.
3. If applicable, share public competition code on its Kaggle forum/notebook as provided by the competition's public-code-sharing rule, without uploading data or credentials.

Official competition rules: https://www.kaggle.com/competitions/filament-segmentation-2026/rules
Official overview: https://www.kaggle.com/competitions/filament-segmentation-2026/overview
