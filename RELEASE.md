# RELEASE.md — maintainer notes: publishing the release assets

Repository: **`AquariusTerimage`** — https://github.com/JunyiDu2009/AquariusTerimage
(AquariusDiffusion series; model name **Aquarius Terimage**).
Path in the series: `AquariusDiffusion/AquariusImage/AquariusTerimage`.
Licence: code, docs **and the weights** are Apache License 2.0 (see `LICENSE` / `NOTICE`).

The repo deliberately tracks **no file above 12.8 MB** (GitHub's hard per-file push limit
is 100 MB). The weights are published as **individual Release assets** (2 GB per asset) —
**no archive**: the repository already carries the Gradio UI, the text-encoder sidecars
and the docs, so a user only drops the four weight files into the same-named folders.

## The release assets (v1.0 — `Aquarius Terimage · step 260,000`)

| # | Asset file | Exact size (bytes) | MB (decimal) | User drops it into |
|---|---|---|---|---|
| 1 | `aquarius_ternary_step260000_base3.safetensors` | 152,990,556 | 152.99 | `portable/` |
| 2 | `diffusion_model_step260000.safetensors` | 323,476,536 | 323.48 | `diffusion_model/` |
| 3 | `model_int8.safetensors` | 755,557,000 | 755.56 | `text_encoder/` |
| 4 | `VAE.safetensors` | 334,643,276 | 334.64 | `VAE/` |
| 5 | `sha256sums.txt` | — | < 1 KB | (verification only) |

Source folder (do not delete until the release is verified):

```
D:/AI_Library_D/Aquarius20261004/04_models/AquariusTerimage/
```

## Local staging — READY TO UPLOAD (prepared 2026-10-04)

`D:/AI_Library_D/发布/GitHub/Release/AquariusDiffusion/AquariusImage/AquariusTerimage/`
holds everything the release needs, already sha256-verified against the archive originals:

- the four weight files (exact byte sizes as in the table above)
- `sha256sums.txt` — publish as the 5th asset so users can verify the downloads
- `release_notes.md` — use as the `--notes-file` for the release
- `upload_release.sh` — one-shot: creates tag `v1.0` + uploads all five assets
  (retry-safe, `--clobber` on re-run; needs `gh` authenticated)

The git working tree for the push is
`D:/AI_Library_D/发布/GitHub/Repo/AquariusDiffusion/AquariusImage/AquariusTerimage/` (its `origin`
is already set to `AquariusTerimage.git`). After the repo is pushed:

```
bash "/d/AI_Library_D/发布/GitHub/Release/AquariusDiffusion/AquariusImage/AquariusTerimage/upload_release.sh"
```

> Superseded routes (kept for the record — do **not** upload either):
> ① `Aquarius20261004/AquariusImage-release-assets/` staged the same four files; the
> weight files were moved out of it into the staging path above on 2026-10-04;
> ② `AquariusTerimage-v1.0-demo.zip` (a 1.59 GB self-contained archive that also carried
> the UI) was dropped on 2026-10-04 in favour of the plain files.

## Hugging Face mirror (DECIDED 2026-10-04)

The same four weight files are also being published on a **Hugging Face model repo** (the
technical report §9 points at it; the canonical download links live in `README.md` /
`README_zh.md`). Once the HF upload is live, add the model-card URL to the "Mirror" note
under the Weights section in both READMEs.

## Pre-push sanity checks

```bash
cd /d/AI_Library_D/发布/GitHub/Repo/AquariusDiffusion/AquariusImage/AquariusTerimage

# nothing over 100 MB tracked in git (expect the largest to be tokenizer.json, 12.8 MB)
git ls-files -z | xargs -0 du -b 2>/dev/null | sort -rn | head -5

# remote points at the right repo
git remote -v

# staged asset integrity
cd "/d/AI_Library_D/发布/GitHub/Release/AquariusDiffusion/AquariusImage/AquariusTerimage" && sha256sum -c sha256sums.txt
```

Note: the staging path contains the Chinese folder name `发布`; `upload_release.sh`
resolves its own directory first, so it works regardless of the caller's locale.
