# SolarSpine R2B — publication checklist (owner controlled)

The private release repository is `topassky3/solarspine-r2b-final-2026`, **not** the original research repository `topassky3/solarspine-pq`. Do not publish the research repository or training data.

## Completed

- Seven original source files and two exact checkpoints are SHA256 verified.
- Both trained checkpoints are stored under Git LFS in the private release repository.
- A clean Windows clone passed strict preflight and six structure tests.
- The authenticated Colab reproduction notebook regenerated the accepted Kaggle CSV with SHA256 `c241bda04c1fefdab5fe1b6b6d90f4aeac23a02a2003f830c0b9b1b289d3294f`.
- The public-facing notebook is `notebooks/SolarSpine_R2B_Jurado_T4.ipynb` and never reads a personal GitHub token.

## Remaining before final submission

1. Review `THIRD_PARTY_NOTICE.md` and the provenance/licensing for the Ultralytics detector, weights, trained refiner and competition data. Do not treat the repository's MIT license as permission for third-party parts.
2. Verify the public-facing notebook and README in a private clone (`python scripts/preflight_release.py --strict`, plus `python -m unittest discover -s tests -p "test_release_layout.py" -v`).
3. Once the owner approves public release, use GitHub settings to make **only** `topassky3/solarspine-r2b-final-2026` public.
4. In a fresh anonymous Colab/runtime, run the jury notebook. It must `git clone` and `git lfs pull` without GitHub credentials. Supply the official 180 TEST JPEGs separately (dataset is not included).
5. Verify `SOLARSPINE_R2B_JURY_CSV_EXACT_MATCH` again. Preserve an unmodified reproducibility log and SHA256 evidence.
6. Review the report PDF and complete the organizer submission form manually, including the public Git URL and exact accepted CSV. The notebook does not submit automatically.

This checklist distinguishes **authenticated scientific reproducibility (passed)** from **anonymous public access and jury release (pending)**.
