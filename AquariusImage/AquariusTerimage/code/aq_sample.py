"""Aquarius sampling: DDIM 20 steps from a trained checkpoint -> PNGs via VAE fp16.

Loads ck["ema"] (COCO Lite convention; falls back to raw "model"), samples 6 fixed
prompts (precomputed embeddings by image_id), decodes with the SD1.5 VAE (fp16),
saves to deliverables/<mode>/samples/ with prompt slug + seed in the filename.

Usage:
  python aq_sample.py --mode ternary
  python aq_sample.py --mode fp --ckpt checkpoints/fp/final.pt
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from aq_train import (CHECKPOINTS_DIR, EMBEDDINGS_DIR, WORKSPACE,  # noqa: E402
                      AquariusData, build_model)
from aq_unet import set_quant_mode  # noqa: E402

SEED = 3407
DDIM_STEPS = 20
LATENT_SIZE = 64          # 512x512 image
VAE_DIR = WORKSPACE / "Aquarius_cloud" / "VAE" / "vae"
SCALE = 0.18215
PROMPT_IDS = [193622, 119964, 218026, 382406, 267802, 107960]
PROMPT_SLUGS = [
    "man_with_decorated_cow", "person_skateboard_field",
    "horse_eating_grass_city", "woman_walking_dog_city",
    "people_mopeds_busy_street", "motorcycle_empty_street",
]


def ddim_sample(model, ctx, abar, steps=DDIM_STEPS, size=LATENT_SIZE, seed=SEED, device="cuda"):
    """Deterministic DDIM (eta=0) with an x0-prediction network, cosine schedule."""
    b = ctx.shape[0]
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(b, 4, size, size, device=device, generator=g)
    ts = torch.linspace(0, 999, steps).round().long().flip(0)  # 999 -> 0
    abar = abar.to(device)
    # MUST run under no_grad: without it the 20-step loop chains every intermediate
    # activation into one retained autograd graph (inference behaving like training),
    # which overflows VRAM and spills into shared system memory via WDDM.
    with torch.no_grad():
        for i, t in enumerate(ts):
            tb = torch.full((b,), int(t), device=device, dtype=torch.long)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                x0_hat = model(x, tb, ctx)
            x0_hat = x0_hat.float()
            a_t = abar[int(t)]
            a_prev = abar[int(ts[i + 1])] if i + 1 < len(ts) else torch.tensor(1.0, device=device)
            eps = (x - a_t.sqrt() * x0_hat) / (1 - a_t).sqrt().clamp_min(1e-8)
            x = a_prev.sqrt() * x0_hat + (1 - a_prev).sqrt() * eps
    # ⚠️ 2026-10-03 修：原来是 clamp(-1,1)。真实潜向量整体 std ≈ 0.835、
    # |x| > 1 占 22%，夹 ±1 等于夹 ±1.2σ ⇒ 削掉 26% 的数值、画面发灰。
    # 详见 play/aq_play.py 文件头 LAT_CLAMP 处的实测数据。
    _lim = float(os.environ.get("AQ_LAT_CLAMP", "6.0"))
    return (x0_hat if len(ts) == 1 else x).clamp(-_lim, _lim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ternary", "binary", "fp"], required=True)
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--lang", choices=["en", "zh"], default="en")
    args = ap.parse_args()

    dev = "cuda"
    ckpt = Path(args.ckpt) if args.ckpt else CHECKPOINTS_DIR / args.mode / "final.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    out_dir = Path(args.out) if args.out else WORKSPACE / "deliverables" / args.mode / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    set_quant_mode(args.mode)
    model = build_model(256, None).to(dev)
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    weights = blob.get("ema") or blob.get("model")
    tag = "ema" if blob.get("ema") else "raw"
    model.load_state_dict(weights)
    model.eval()
    print(f"[sample] ckpt={ckpt} (weights='{tag}', step={blob.get('step')}) mode={args.mode}")

    data = AquariusData(WORKSPACE / "Aquarius_cloud" / "latents",
                        WORKSPACE / "Aquarius_cloud" / "embeddings", args.lang)
    ctx = torch.stack([data.get_embedding(i) for i in PROMPT_IDS]).to(dev)

    _, _, abar = __import__("aq_train").cosine_schedule()
    x0 = ddim_sample(model, ctx, abar)
    latents = x0 / SCALE

    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(str(VAE_DIR), torch_dtype=torch.float16).to(dev).eval()
    with torch.no_grad():
        imgs = vae.decode(latents.half()).sample          # [-1,1]
    imgs = (imgs.float() / 2 + 0.5).clamp(0, 1)

    from PIL import Image
    import numpy as np
    meta = []
    for i in range(imgs.shape[0]):
        arr = (imgs[i].permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
        slug = PROMPT_SLUGS[i]
        fn = f"{i:02d}_id{PROMPT_IDS[i]}_{slug}_seed{SEED}.png"
        Image.fromarray(arr).save(out_dir / fn)
        arr_min, arr_max, arr_mean = arr.min(), arr.max(), arr.mean().round(1)
        meta.append({"file": fn, "image_id": PROMPT_IDS[i], "prompt_slug": slug,
                     "min": int(arr_min), "max": int(arr_max), "mean": float(arr_mean)})
        print(f"[sample] {fn} min={arr_min} max={arr_max} mean={arr_mean}")
    (out_dir / "prompts.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"[sample] done -> {out_dir}")


if __name__ == "__main__":
    main()
