# RELEASE.md — maintainer notes: publishing the weight files

The git repository deliberately contains **no file above 12.8 MB** (GitHub's hard
per-file push limit is 100 MB). The four model binaries are published as
**GitHub Release assets** (limit: 2 GB per asset — all four fit comfortably).

## The four assets (v1.0 — `Aquarius Terimage · step 260,000`)

| # | Asset file | Exact size (bytes) | MB (decimal) | User drops it into |
|---|---|---|---|---|
| 1 | `aquarius_ternary_step260000_base3.safetensors` | 152,990,556 | 152.99 | `portable/` |
| 2 | `diffusion_model_step260000.safetensors` | 323,476,536 | 323.48 | `diffusion_model/` |
| 3 | `model_int8.safetensors` | 755,557,000 | 755.56 | `text_encoder/` |
| 4 | `VAE.safetensors` | 334,643,276 | 334.64 | `VAE/` |

Source copies (do not delete from the archive until the release is verified):

```
D:/AI_Library_D/Aquarius20261004/04_models/AquariusTerimageDemo/portable/aquarius_ternary_step260000_base3.safetensors
D:/AI_Library_D/Aquarius20261004/04_models/AquariusTerimageDemo/diffusion_model/diffusion_model_step260000.safetensors
D:/AI_Library_D/Aquarius20261004/04_models/AquariusTerimageDemo/text_encoder/model_int8.safetensors
D:/AI_Library_D/Aquarius20261004/04_models/AquariusTerimageDemo/VAE/VAE.safetensors
```

## Local staging — READY TO UPLOAD (prepared 2026-10-04)

`D:/AI_Library_D/Aquarius20261004/AquariusImage-release-assets/` contains everything
the release needs, already sha256-verified against the archive originals:

- the four weight files (exact byte sizes as in the table above)
- `sha256sums.txt` — publish as a **5th asset** so users can verify downloads
- `release_notes.md` — use as the `--notes-file` for the release
- `upload_release.sh` — one-shot: creates tag v1.0 + uploads all 5 assets
  (retry-safe, `--clobber` on re-run; needs `gh` authenticated)

After the repo is pushed, run:

```
bash /d/AI_Library_D/Aquarius20261004/AquariusImage-release-assets/upload_release.sh
```

## Suggested release commands

```bash
git init -b main
git add .
git commit -m "Aquarius: natively ternary text-to-image diffusion model (step 260,000)"
git remote add origin https://github.com/JunyiDu2009/AquariusImage.git
git push -u origin main

gh release create v1.0 --title "Aquarius Terimage - step 260,000 weights" --notes "
Four weight files for the 558M natively-ternary text-to-image model.
Place each file in the folder named in the 'Put it in' column of README.md.
Expected-output notice: images are blurry scene compositions without recognizable
objects - this is the documented state of this checkpoint (report section 7.1).
" 
gh release upload v1.0 \
  aquarius_ternary_step260000_base3.safetensors \
  diffusion_model_step260000.safetensors \
  model_int8.safetensors \
  VAE.safetensors
```

(Upload the four files from the archive paths above; they are NOT inside the repo
working tree, by design.)

## Alternative: Hugging Face

If preferred, upload the same four files to a Hugging Face model repo instead of
(or in addition to) Release assets, then replace the "Weights" table links in
`README.md` / `README_zh.md` with the HF URLs. Everything else in the repo stays
unchanged — the loaders only care about the local file paths.

## Pre-push sanity checks

```bash
# nothing over 100 MB tracked in git (expect the largest to be tokenizer.json, 12.8 MB)
git ls-files -z | xargs -0 du -b 2>/dev/null | sort -rn | head -5

# tracked repo size (expect roughly 60-65 MB total)
git count-objects -vH
```
