# Research Project — Reproducing Segment Any Anomaly (SAA+)

Goal: run [SAA+](https://github.com/caoyunkang/Segment-Any-Anomaly) (training-free
anomaly segmentation via Grounding DINO + SAM) on the `bottle` category of
[MVTec-AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad), as a
demo to show a professor. Development happens here with Claude; execution
happens on Kaggle (free GPU), since local hardware isn't sufficient.

## Workflow

Claude can't currently reach Kaggle's API directly from this environment (network
policy blocks it — see chat history), so we're using a manual bridge:

1. Claude writes/edits the notebook code in `kaggle/`.
2. You push it to Kaggle (CLI or website — see `kaggle/README.md`) and run it.
3. If it errors, you paste the error back here.
4. Claude fixes the code, you re-push. Repeat until it works.

## Status

- [x] SAA+ repo structure inspected (cloned upstream to check real dependencies/paths, not vendored into this repo)
- [x] Notebook scaffolded: clone → install deps → download weights → link bottle-only data → sanity check → full eval → results
- [x] Public MVTec-AD Kaggle mirror identified: `ipythonx/mvtec-ad`
- [ ] First successful run on Kaggle (needs you to push and report back any errors)
- [ ] Reported metrics (image/pixel AUROC) captured for bottle
- [ ] (Optional) extend to more MVTec-AD categories

## Kaggle account setup (one-time)

1. kaggle.com → profile picture → **Settings**
2. **Phone Verification** section → verify your phone (required to enable GPU on kernels)
3. **API** section → **Create New Token** → gives you the credentials for pushing kernels (see `kaggle/README.md` for how to install/use them)
4. Nothing else needed — the dataset (`ipythonx/mvtec-ad`) gets attached per-notebook, not per-account

## Next steps

See `kaggle/README.md` for exact push instructions.
