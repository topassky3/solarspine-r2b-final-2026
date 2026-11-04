# Release checklist

This pre-release package contains frozen sources and a tested inference runner. The earlier R2B jury-package execution passed exact CSV identity on Colab T4; however **public Git distribution and its notebook bootstrap require further verification**. See [`docs/RELEASE_CHECKLIST.md`](../../docs/RELEASE_CHECKLIST.md) and run `python scripts/preflight_release.py --strict`.

Do not publish dataset images/labels, private Drive snapshots, secrets or the full research repository. Do not claim a new model or improved public score from the CSV re-upload. Official report template, weights, external access and Google Form remain pending.
