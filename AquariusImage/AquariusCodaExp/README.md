# AquariusCodaExp · 0.08B One-Step Denoiser

> Pixel-space single-step denoising: feed in a "noisy but roughly clean" image, get the clean image back in **one forward pass**. No VAE, no diffusion loop, no sampling chain — just one regression step.

**48.1M params** (~0.08B) UNet · RGB 3-channel · trainable on RTX 5060 8GB · full pipeline incl. NVFP4 quantization and a Gradio demo.

> 🇨🇳 完整中文文档见 [README_zh.md](README_zh.md)。

## What it is

| | |
|---|---|
| Task | one-step denoising: `x_p = √(1−p)·x₀ + √p·ε` → recover `x₀` in a single forward |
| Noise range | p = noise variance ratio ∈ **1% – 50%** (main regime 1–30%, 30–50% extreme tier) |
| Architecture | UNet, GroupNorm + SiLU, mid-block multi-head attention, gradient checkpointing |
| Loss | MSE + LPIPS(VGG) |
| Data | COCO 2017 (115K train / 5K val) |

No sampling chain — inference is literally `model(x_p, p)`, milliseconds.

## Highlights

- **Multi-resolution × aspect-ratio buckets**: 11 buckets (256² – 640×384, incl. 16:9/9:16), per-step bucket sampling with pixel-budget-scaled batch — inference domain covers any /16-divisible input.
- **4-stage noise curriculum**: 1–10% → +20% → +30% → 1–50%.
- **Plateau early-stop**: when the 10/30/50% LPIPS gate stalls for 10K steps (≤0.002), auto-save and exit. Triggered at **76,000 steps** as predicted.
- **NVFP4 quantization**: E2M1 + per-16 E4M3 scale + FP32 per-tensor scale, hand-written codec (`code/nvfp4.py`), exported every 1000 steps during training. Note: this experiment is **FP16-trained with post-training NVFP4 export** (not native low-bit).

## Results (val2017 5K, NVFP4 dequant, step 76000)

| Noise | PSNR (dB) | LPIPS |
|---|---|---|
| p=1%  | 33.50 | 0.091 |
| p=10% | 28.54 | 0.191 |
| p=20% | 26.86 | 0.236 |
| p=30% | 25.78 | 0.270 |
| p=40% | 24.90 | 0.297 |
| p=50% | 24.07 | 0.326 |

Quantization cost: 96 MB (FP16) → **27.2 MB (NVFP4)**, 3.5× smaller. Full training curves in [`results/`](results/).

## Quick start

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install lpips

cd code
python train.py --data D:/COCO2017/train2017 --eval-data ../data/val2017 \
  --out runs/m2_formal --ckpt-dir ckpts/m2_formal \
  --steps 250000 --batch 6 --res 256 --save-every 1000 --eval-every 500 \
  --workers 8 --early-stop --multi-res --mem-frac 0.65

python demo.py          # Gradio UI at http://127.0.0.1:8792
```

## Structure

```
AquariusCodaExp/
├── README.md / README_zh.md   this page
├── code/                      train / model / nvfp4 / demo / dashboard / test_pipeline
├── docs/                      training plan v1.1 / v1.2 (v1.2 = authoritative)
└── results/                   training curves (eval records + gzipped full log)
```

## Roadmap

- [ ] recalibrate LPIPS acceptance lines
- [ ] M3: FP8 QAT (torchao)
- [ ] weight release
