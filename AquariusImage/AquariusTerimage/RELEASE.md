# RELEASE.md — maintainer notes: publishing the release assets

Repository: **`AquariusDiffusion`** — https://github.com/JunyiDu2009/AquariusDiffusion
(the series). This model lives in its `AquariusImage/AquariusTerimage/` folder;
model name **Aquarius Terimage**.
Licence: code, docs **and the weights** are Apache License 2.0 (see `LICENSE` / `NOTICE`).

The repo deliberately tracks **no file above 12.8 MB** (GitHub's hard per-file push limit
is 100 MB). The release ships **the runnable package as loose flat assets — no archive**:
the three weight files plus the UI/launcher and the text-encoder sidecars. A user
downloads them, drops each into the folder named in `release_notes.md`, and double-clicks
`run_ui.bat`.

## Weights actually shipped: three files

| # | Asset file | Exact size (bytes) | MB (decimal) | Goes into |
|---|---|---|---|---|
| 1 | `aquarius_ternary_step260000_base3.safetensors` | 152,990,556 | 152.99 | `portable/` |
| 2 | `model_int8.safetensors` | 755,557,000 | 755.56 | `text_encoder/` |
| 3 | `VAE.safetensors` | 334,643,276 | 334.64 | `VAE/` |

**Deliberately NOT shipped: `diffusion_model_step260000.safetensors` (323.48 MB, prebuilt
int4 runtime).** It is only a convenience snapshot of what the loader rebuilds anyway —
from the base-3 master it builds the int4 fused kernel at load time on CUDA, and the
unpack-to-float path on any other backend. `aq_play.default_ckpt()` looks in
`diffusion_model/` first and **falls back to `portable/`** when that folder has no export
(the fallback was added 2026-10-04; see the comment inside the function). Verified
end-to-end by importing `aq_play` from the staging folder itself: with no
`diffusion_model/`, it resolves `portable/…_base3.safetensors` (step 260000).

## Non-weight assets (copies of the repository)

`run_ui.bat`, `app.py`, `aq_play.py`, `aq_unet.py`, `aq_kernel.py`, `aq_lowbit.py`,
`load_te.py`, `requirements.txt`, `README.txt`, the seven `text_encoder/` sidecars
(`tokenizer.json`, `vocab.json`, `merges.txt`, `config.json`, `quant.json`,
`tokenizer_config.json`, `chat_template.jinja`) and `sha256sums.txt`.

They are **unversioned copies** of the repository's `code/` + `text_encoder/`: if a code
fix lands in the repo, re-copy it here and re-run the upload. `sha256sums.txt` covers the
**three weight files only** (bare file names — that is what a user has after downloading
from the release page); `upload_release.sh` resolves each name to its folder before
verifying.

## STATUS — code is LIVE (2026-10-04)

Pushed and verified:

- **https://github.com/JunyiDu2009/AquariusDiffusion** (public, branch `main`).
  The model page is `AquariusImage/AquariusTerimage/README.md`; its `docs/` carries both
  report PDFs.
- Pushing from this machine needs **two workarounds**: Windows schannel blocks on the CRL
  check, and the interactive **Git Credential Manager must be bypassed** (it pops an
  account-selection dialog and hangs the push). Hand the token over directly instead:

```bash
cd /d/AI_Library_D/发布/GitHub/Repo/AquariusDiffusion      # git root = series root
TOK=$(printf 'protocol=https\nhost=github.com\n\n' | git credential fill | sed -n 's/^password=//p')
git -c credential.helper= -c http.schannelCheckRevoke=false \
    -c http.extraHeader="Authorization: Basic $(printf 'x-access-token:%s' "$TOK" | base64 -w0)" \
    push origin main
```

- **Release `v1.0` is LIVE** (2026-10-05): 20 assets — the three weight files plus the
  loose UI/launcher and the text-encoder sidecars. Uploaded with `gh` at ~1 MB/s; the
  earlier "~70 KB/s, several hours" reading was an artefact of the credential manager
  stalling the transfer.
- Auth on this machine: **`gh auth login --web` (device code)** — the Git Credential
  Manager pops an account-selection dialog and hangs git, so never rely on it. Take the
  token with `gh auth token` (for API calls or the `http.extraHeader` push);
  **do not call `git credential fill`**.

## Local staging — READY TO UPLOAD (prepared 2026-10-04)

`D:/AI_Library_D/发布/GitHub/Release/AquariusDiffusion/AquariusImage/AquariusTerimage/`
**is** the runnable layout — the same shape a user ends up with:

```
portable/…_base3.safetensors · text_encoder/{model_int8.safetensors + 7 sidecars} · VAE/VAE.safetensors
run_ui.bat · app.py · aq_play.py · aq_unet.py · aq_kernel.py · aq_lowbit.py · load_te.py
requirements.txt · README.txt · sha256sums.txt · release_notes.md · upload_release.sh
```

- `upload_release.sh` — one-shot: verifies the three weights against `sha256sums.txt`,
  creates tag `v1.0`, uploads every asset flat with `--clobber` (retry-safe; needs `gh`
  authenticated)
- `release_notes.md` — use as the `--notes-file`; it carries the placement table

The git working tree is `D:/AI_Library_D/发布/GitHub/Repo/AquariusDiffusion/` (the series
root; `origin` = `AquariusDiffusion.git`). After the repo is pushed:

```
bash "/d/AI_Library_D/发布/GitHub/Release/AquariusDiffusion/AquariusImage/AquariusTerimage/upload_release.sh"
```

> Superseded on 2026-10-04 — **do not upload**, kept in `发布/GitHub/_superseded/`:
> ① `AquariusTerimage-v1.0-demo.zip` (1.59 GB archive that duplicated the UI), and
> ② `diffusion_model_step260000.safetensors` (the prebuilt int4 runtime).

## Hugging Face mirror (DECIDED 2026-10-04)

The same three weight files are also being published on a **Hugging Face model repo** (the
technical report §9 points at it; the canonical download links live in `README.md` /
`README_zh.md`). Once the HF upload is live, add the model-card URL to the "Mirror" note
under the Weights section in both READMEs.

## GitHub repo settings (description / topics — not tracked in git)

The About **Description**, the **Topics** and the optional **Website** are repository
settings, not files; the current values are already applied to
`JunyiDu2009/AquariusDiffusion`. Ready-to-paste copy (English / short / Chinese
descriptions, topics, the one-shot `gh repo edit` command) lives outside the repo:

```
D:/AI_Library_D/发布/GitHub/GITHUB_METADATA.md
```

## Pre-push sanity checks

```bash
cd /d/AI_Library_D/发布/GitHub/Repo/AquariusDiffusion

# nothing over 100 MB tracked in git (expect the largest to be tokenizer.json, 12.8 MB)
git ls-files -z | xargs -0 du -b 2>/dev/null | sort -rn | head -5

# remote points at the right repo
git remote -v
```

Note: the staging path contains the Chinese folder name `发布`; `upload_release.sh`
resolves its own directory first, so it works regardless of the caller's locale.
