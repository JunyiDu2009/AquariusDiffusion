# Aquarius Terimage — A From-Scratch Natively Ternary Text-to-Image Diffusion Model

**Native Low-bit Training (NLT)** · 558,347,012 params · **152.99 MB** single-file model (2.192 bpw) · full technical report included

English | [中文说明](README_zh.md)

**Model: Aquarius Terimage** — part of the **AquariusDiffusion** series (native low-bit
text-to-image diffusion), at `AquariusDiffusion/AquariusImage/AquariusTerimage`. This
repository is the **ternary** member; its planned **binary** counterpart
(`AquariusBinimage`) is **not trained** — the compute budget was exhausted (report §7.1).

---

## What is this?

Nearly all low-bit diffusion work first pre-trains in fp16 and then compresses (PTQ/QAT).
**Aquarius takes the opposite route: the weights live in the quantized space from step 0**
(Native Low-bit Training, NLT). A 558M-parameter text-to-image UNet is trained from random
init with **540,147,712 of its weights constrained to ternary values `{−s, 0, +s}`**
(one fp16 scale per 128 weights, g128 mean-scale + STE), on a single 32 GB consumer GPU,
for 260,000 steps (≈ 28.70 epochs of COCO train2017) — no collapse, no divergence, no OOM.

Two conclusions, both honest:

1. **Feasibility: YES.** NLT training is stable end-to-end, and the ternary weights can be
   placed **bit-exactly** into torch's built-in `_weight_int4pack_mm` int4 kernel
   (`q₄ = q·7 + 8 ∈ {1, 8, 15}`), so inference runs with the weights resident at
   **287.3 MB** and unpacking fused inside the GEMM.
2. **Model quality: NO.** This is the deliberate negative result. All of clip_ratio's rise
   happens between steps 9k–110k; the **net change is ≈ 0 over the following 130k steps**,
   and the all-time peak (0.3782) occurs at step 170k. Training was stopped at step 260,000
   on this saturation evidence, compounded by an exhausted compute budget.

> ### ⚠️ Expected output — read this first
> Generated images are **blurry scene compositions with no recognizable objects**.
> Sky/ground layering, colour tone and rough spatial layout do follow the prompt
> ("sunset over the ocean" ≫ "a cat"). **That is the expected state of this
> under-trained checkpoint — not a broken environment and not a bug in these scripts.**
> Same seed + same prompt = same image (deterministic DDIM, cosine schedule, η = 0).

## Key numbers

| Item | Value |
|---|---|
| Parameters | 558,347,012 (540,147,712 ternary-quantized; rest kept fp16) |
| Quantizer | ternary `{−s, 0, +s}`, one fp16 scale per 128 weights (g128 mean-scale) + STE |
| Training | 260,000 steps = 28.70 epochs, COCO train2017 (118k captions, 22 aspect-ratio buckets), single 32 GB GPU |
| Deliverable | **152.99 MB** single file, **2.192 bpw** — base-3 packing at the `log₂3 ≈ 1.585` bit floor |
| vs fp16 | **7.3×** smaller (fp16 weights = 1116.73 MB) |
| Runtime weights | int4 fused kernel, **bit-exact**, resident **287.3 MB** |
| Speed (RTX 5090, 512×512, 20 steps) | 1.34–1.44 s/image (all models resident, peak **1.92 GB**) · 2.27–2.51 s/image with `--te-mode cache` (peak 1.34 GB) |
| Hardware fit | comfortable on an 8 GB card |
| clip_ratio | rises 9k → 110k, peak **0.3782** @ step 170k, flat afterwards |

<p>
  <img src="figures/en/fig1_pipeline.png" width="49%" alt="pipeline" />
  <img src="figures/en/fig3_clip_ratio.png" width="49%" alt="clip ratio" />
</p>

<p>
  <img src="figures/en/fig4_samples.png" width="49%" alt="samples" />
  <img src="figures/ui_screenshot.png" width="49%" alt="demo UI" />
</p>

Evolution strip (18 sampling points, fixed 10-caption protocol, earlier protocol — only comparable within it):

<p><img src="figures/en/fig5_evolution.png" width="100%" alt="evolution strip" /></p>

100 random captions at 640×640: [figures/samples_100captions_web.png](figures/samples_100captions_web.png)

## Repository layout

```
AquariusTerimage/
├── code/                  all source (inference + training + packing + eval)
├── text_encoder/          tokenizer & config sidecars of the int8 text encoder
│                          (model_int8.safetensors itself is in the release zip)
├── diffusion_model/       (empty — drop the int4 runtime model here)
├── portable/              (empty — drop the base-3 master model here)
├── VAE/                   (empty — drop the SD1.5 VAE here)
├── docs/                  technical reports (CN/EN, PDF + Markdown source) + stages.json
├── figures/               report figures (zh/en), UI screenshot, 100-caption grid
├── requirements.txt
├── LICENSE · NOTICE        Apache-2.0 + third-party attribution notices
└── README.md
```

The `text_encoder/` sidecars + the three empty weight folders are where the three Release
weight files go: drop each file into the same-named folder and everything runs out of the
box (the loader rebuilds the int4 runtime kernel from `portable/` at load time).

## Quick start

```bash
# 1) install the CUDA build of torch FIRST (do not let pip pick on its own,
#    otherwise you get the CPU-only wheel and the fused kernel is unavailable)
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128

# 2) everything else
pip install -r requirements.txt
```

Download the three weight files from **Releases** (see the table below) into their folders, then:

| Windows | macOS / Linux |
|---|---|
| double-click `code/run_ui.bat` | `cd code && python app.py` |

```bash
python app.py --port 8888          # custom port
python app.py --lowvram            # peak < 3 GB, reloads models each time (slow)
python app.py --self-test          # no UI: verify the whole path on CPU
python aq_play.py "a cat on a chair" --res 512 --steps 20   # CLI, no UI
```

The bundled runtime model is the **int4 fused-kernel build → NVIDIA CUDA only**
(the ternary→int4 mapping is implemented on torch's `_weight_int4pack_mm`).
On a non-CUDA machine point the loader at the base-3 master instead
(`python app.py --ckpt portable/aquarius_ternary_step260000_base3.safetensors`);
it dequantises to fp32, so it is slower but platform-independent.

The text encoder keeps its weights in int8 and reconstructs them in the quantized
space — `transformers.from_pretrained` cannot read that file, which is why
`code/load_te.py` exists (already wired into `app.py` / `aq_play.py`).

### Weights — three files, straight from Releases (no archive)

| Release asset | Size | Put it in |
|---|---|---|
| `aquarius_ternary_step260000_base3.safetensors` (base-3 master) | 152.99 MB | `portable/` |
| `model_int8.safetensors` (text encoder, Qwen3.5-0.8B language tower, int8) | 755.56 MB | `text_encoder/` |
| `VAE.safetensors` (SD1.5 VAE, fp16) | 334.64 MB | `VAE/` |
| `sha256sums.txt` | < 1 KB | — verify with `sha256sum -c` |

The **fourth** model file you may have seen elsewhere — `diffusion_model_step260000.safetensors`
(prebuilt int4 runtime, 323 MB) — is **deliberately not shipped**: it is only a convenience
snapshot of what the loader rebuilds from the base-3 master **at load time, for whatever
device you have** (int4 fused kernel on an NVIDIA GPU, unpack-to-float elsewhere). Put the
three files above in place and run — no flags needed.

The release page also carries the Gradio UI, the Windows launcher and the text-encoder
sidecars as loose files, so the download runs without cloning this repository.

> **Mirror:** the same four files are also being published on **Hugging Face**; the HF
> link will be added here once live. This README stays the canonical index of download links.

> Do **not** drop the base-3 master into `diffusion_model/` — that folder is globbed
> and the "highest step" tie would make model selection ambiguous. Use `--ckpt` instead.

## Code map

| file | role |
|---|---|
| `code/aq_unet.py` | the 558M UNet + the NLT quantizer (g128 mean-scale, STE) |
| `code/aq_kernel.py` | low-bit fused GEMM paths (ternary → int4 `_weight_int4pack_mm`) |
| `code/aq_lowbit.py` | base-3 bit-exact pack/unpack |
| `code/load_te.py` | self-contained loader for the int8 text encoder |
| `code/aq_play.py` | inference engine + CLI (all tuning switches live here) |
| `code/app.py` | Gradio web UI (bilingual) |
| `code/run_ui.bat` | Windows launcher (ASCII-only + CRLF on purpose — see the comment inside) |
| `code/aq_train.py` | NLT training loop (weights quantized from step 0) |
| `code/aq_pack.py` / `aq_pack_int4.py` | checkpoint → base-3 deliverable / → int4 runtime file |
| `code/aq_clip_score.py` | CLIP image↔text alignment score, normalized against real-COCO upper bound |
| `code/aq_metrics.py` | statistical health metrics (vs noise floor / self-check baselines) |
| `code/aq_te_slim.py` / `aq_slim_ckpt.py` / `_te_slim_loader.py` / `aq_export_latest.py` | TE int8 slimming, checkpoint slimming, export utilities |
| `code/aq_paths.py` / `aq_units.py` | cross-machine path resolver (no hard-coded user dirs) / decimal MB-GB helpers |
| `code/aq_turbo.py` | **roadmap scaffold**: DMD2-style one-step distillation (G = ternary weights, STE kept ON) — not yet trained |
| `code/aq_sample.py` | stage sampler used for the evolution strip |

Training-side scripts (`aq_train.py`, `aq_sample.py`, `aq_turbo.py`, `aq_pack.py`)
assume the original training workspace layout (resolved via `aq_paths.py` and the
constants at the top of `aq_train.py`); they were exercised on a single 32 GB GPU.
Inference/demo files are self-contained and portable.

## Documentation

- [docs/Aquarius_Technical_Report_EN.pdf](docs/Aquarius_Technical_Report_EN.pdf) · [.md](docs/Aquarius_Technical_Report_EN.md)
- [docs/Aquarius_Technical_Report_CN.pdf](docs/Aquarius_Technical_Report_CN.pdf) · [.md](docs/Aquarius_Technical_Report_CN.md)
  - If you read only one section: **§5.6 "six transferable measured lessons"**.
- [docs/stages.json](docs/stages.json) — stage registry behind the evolution strip
- [RELEASE.md](RELEASE.md) — maintainer notes: how the release package is published

## License

- **Code, docs and the model weights: Apache License 2.0** (see [LICENSE](LICENSE) and
  [NOTICE](NOTICE)).
- Bundled third-party components keep their own upstream terms: the text encoder is
  derived from **Qwen3.5-0.8B**, the VAE and the UNet structure are isomorphic to
  **SD1.5**, and the training data is **COCO train2017** — check them before
  redistributing.

## Citation

```bibtex
@techreport{du2026aquarius,
  title       = {Aquarius: A From-Scratch, Natively Ternary, Text-to-Image Diffusion Model},
  author      = {Du, Junyi},
  institution = {Independent Research},
  year        = {2026},
  month       = {10},
  note        = {Technical report, code and weights}
}
```

## Acknowledgments

- **AI assistance**: implementation, data analysis and report drafting were carried out in collaboration with **GLM-5.3-Flash** (Zhipu Z.ai) and **DeepSeek V4.1 Flash** (DeepSeek AI) — see the report's Acknowledgements section.
- **TerDiT** (ICLR 2025) — first from-scratch ternary diffusion training; the route is validated at DiT scale.
- **Bonsai Image** (PrismML) — reference for low-bit packaging & release conventions.
- **SD1.5** (Stability AI) — VAE weights & UNet structural reference · **Qwen3.5** (Alibaba) — text encoder base · **COCO** — training data.
