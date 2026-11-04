# Final release gate

- [x] Frozen source inventory and seven SHA256 locks included in package
- [x] Frozen backend and inference runner byte-preserved
- [x] Separate previous jury-package T4 replay passed exact CSV identity
- [x] Clean standalone repository structure with README and requirements
- [x] Separate setup notebook no longer requires a private repository or source ZIP
- [ ] License/attribution review for Ultralytics pretrained weights, runtime and dataset
- [ ] Upload exact detector checkpoint via Git LFS at `models/detector/best.pt`
- [ ] Upload exact refiner checkpoint via Git LFS at `models/refiner/best.pt`
- [ ] `python scripts/preflight_release.py --strict` reports PASS
- [ ] Create/review separate public GitHub repository (not entire private research history)
- [ ] Verify clone/pull and both weights from anonymous access on Tesla T4
- [ ] Rerun public repository notebook end to end; compare both CSV SHA256
- [ ] Adapt four-page report to the **official Overleaf template** and verify rendering
- [ ] Supply public Git URL, exact Kaggle CSV and official PDF in Google final form
- [ ] Receive submission confirmation by 15 Nov 2026 23:59 CDT

No branch merge, publication, model upload, or Google Form submission is implied by this checklist.
