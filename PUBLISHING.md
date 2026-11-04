# Publication procedure (owner-controlled, not executed)

**Never publish the private research repo `topassky3/solarspine-pq` directly.** Create a new, minimal repository from this directory once models/licenses are approved.

## Before public visibility

1. Review `THIRD_PARTY_NOTICE.md` and approved terms for pretrained weights and official competition images.
2. Copy exactly the two SHA256-locked checkpoint files into `models/detector/best.pt` and `models/refiner/best.pt`. Do not use alternate epochs.
3. Install Git LFS and run `git lfs install`, then confirm `git check-attr filter -- models/detector/best.pt` returns `lfs` (similarly for refiner).
4. Run `python scripts/preflight_release.py --strict`; require `status=PASS`.
5. Replace the blank `REPO_URL` in `notebooks/SolarSpine_R2B_Jurado_T4.ipynb` with the actual future public URL. The notebook can then clone anonymously.
6. Create a **new private** GitHub repository, commit only this directory, and verify its file tree (exclude CSV, backups, tokens and TEST images).
7. Only with explicit approval, change repo visibility to public; test `git clone`, LFS pull and Colab execution as an anonymous outsider.
8. Submit **the public Git URL**, final template-verified report PDF and exact Kaggle CSV through the final form.

## Optional Windows PowerShell commands (after the preceding checks)

```powershell
# Inside the extracted folder, AFTER the two checkpoint files are added and approved.
python scripts/preflight_release.py --strict
git init -b main
git lfs install
git add .
git status --short
git commit -m "Reproducible SolarSpine R2B jury package"
# Requires GitHub CLI authentication. Creates PRIVATE repo, not publicly accessible yet.
gh repo create topassky3/solarspine-r2b-final-2026 --private --source . --remote origin --push
```

Do not make it public or submit the Google form until all release gates are green. A final public URL is not yet known to exist.
