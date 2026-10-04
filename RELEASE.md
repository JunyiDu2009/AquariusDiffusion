# RELEASE.md — maintainer notes: publishing the release package

Repository: **`AquariusTerimage`** — https://github.com/JunyiDu2009/AquariusTerimage
(AquariusDiffusion series; model name **Aquarius Terimage**).
Path in the series: `AquariusDiffusion/AquariusImage/AquariusTerimage`.
Licence: code, docs **and the weights** are Apache License 2.0 (see `LICENSE` / `NOTICE`).

The repo deliberately tracks **no file above 12.8 MB** (GitHub's hard per-file push
limit is 100 MB). The weights ship as **Release assets** (2 GB per asset), packaged as
one self-contained zip so the download also carries the runnable Gradio UI.

## The release assets (v1.0 — `Aquarius Terimage · step 260,000`)

| # | Asset | What it is | Size |
|---|---|---|---|
| 1 | `AquariusTerimage-v1.0-demo.zip` | the 4 weight files + bilingual Gradio UI + `run_ui.bat` (unpacks to an `AquariusTerimage/` folder) | ~1.5 GB |
| 2 | `sha256sums.txt` | sha256 of the zip **and** of the four files inside it | < 1 KB |

Inside the zip (unpacks to `AquariusTerimage/`):

| File | Exact size (bytes) | MB (decimal) | Folder |
|---|---|---|---|
| `aquarius_ternary_step260000_base3.safetensors` | 152,990,556 | 152.99 | `portable/` |
| `diffusion_model_step260000.safetensors` | 323,476,536 | 323.48 | `diffusion_model/` |
| `model_int8.safetensors` | 755,557,000 | 755.56 | `text_encoder/` |
| `VAE.safetensors` | 334,643,276 | 334.64 | `VAE/` |

Source folder that gets zipped (do not delete until the release is verified):

```
D:/AI_Library_D/Aquarius20261004/04_models/AquariusTerimage/
```

## Local staging — READY TO UPLOAD (prepared 2026-10-04)

`D:/AI_Library_D/发布/GitHub/Release/AquariusDiffusion/AquariusImage/AquariusTerimage/`
holds everything the release needs:

- `AquariusTerimage-v1.0-demo.zip` — the release asset
- `sha256sums.txt` — publish as the 2nd asset (covers the zip and the four files inside)
- `release_notes.md` — use as the `--notes-file` for the release
- `upload_release.sh` — one-shot: creates tag `v1.0` + uploads both assets
  (retry-safe, `--clobber` on re-run; needs `gh` authenticated)

The git working tree for the push is
`D:/AI_Library_D/发布/GitHub/Repo/AquariusDiffusion/AquariusImage/AquariusTerimage/` (its `origin`
is already set to `AquariusTerimage.git`). After the repo is pushed:

```
bash "/d/AI_Library_D/发布/GitHub/Release/AquariusDiffusion/AquariusImage/AquariusTerimage/upload_release.sh"
```

> Superseded: `D:/AI_Library_D/Aquarius20261004/AquariusImage-release-assets/` staged the
> four files as four *separate* assets (the earlier plan). The self-contained zip is now
> the chosen single-asset route; the old staging folder can be reclaimed once this release
> is verified.

## Hugging Face mirror (DECIDED 2026-10-04)

The same package is also being published on a **Hugging Face model repo** (the technical
report §9 points at it; the canonical download links live in `README.md` / `README_zh.md`).
Once the HF upload is live, add the model-card URL to the "Mirror" note under the Weights
section in both READMEs.

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
