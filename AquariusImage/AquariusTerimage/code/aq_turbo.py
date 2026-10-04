"""Aquarius Terimage Turbo — DMD2-style one-step distillation (simplified).

Setup (per handoff 动作⑧ + user 2026-10-01 22:46 directive):
  teacher : fp-forward of the ternary ckpt_final raw fp32 master (frozen).
            (Reversible decision: skips a dedicated 4.4h fp run; swap in a
            dedicated fp teacher later if distillation quality is poor.)
  G       : the same ternary master, STE quantization kept ON, trainable.
  F       : fake-score network, same arch, fp mode, init from teacher, trainable.
  D       : small latent-space conv discriminator (real x0 vs G one-step fakes).

Losses per step (t ~ U[0,T), x_fake = G(x_T~N(0,I), t=T-1, ctx) one-step):
  G : ||F(x_fake,t) - Teacher(x_fake,t)||^2          (distribution matching)
      + lambda_gan * BCE(D(x_fake), real)            (adversarial)
  F : ||F(x_fake.detach(),t) - Teacher(x_fake.detach(),t).detach()||^2
  D : BCE(D(x_real),1) + BCE(D(x_fake.detach()),0)
Fallback (every --fallback-every steps, DMD2 drift guard): replace the step with
  standard real-data denoising for G and F (anchors both to the true score).

Usage:
  python aq_turbo.py --mode ternary --steps 5000
  python aq_turbo.py --base 64 --steps 3        # fast smoke
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from aq_train import (CHECKPOINTS_DIR, WORKSPACE, AquariusData,  # noqa: E402
                      build_model, cosine_schedule)
from aq_unet import set_quant_mode  # noqa: E402

NUM_TIMESTEPS = 1000
SCALE = 0.18215
# ⚠️ 别用 ±1 夹潜向量（2026-10-03 修）：真实潜向量整体 std ≈ 0.835、|x| > 1 占 22%，
# 夹 ±1 = 夹在 ±1.2σ ⇒ 削掉 26% 的数值、对比度丢失，且与训练目标不一致。
# 详见 play/aq_play.py 文件头 LAT_CLAMP 的实测数据。
LAT_CLAMP = float(os.environ.get("AQ_LAT_CLAMP", "6.0"))
PROMPT_IDS = [193622, 119964, 218026, 382406, 267802, 107960]
PROMPT_SLUGS = ["man_cow", "skateboard_field", "horse_city", "walk_dog",
                "mopeds_street", "motorcycle_street"]


class Discriminator(nn.Module):
    """Small conv discriminator on latents (4, h, w)."""

    def __init__(self, in_ch=4, base=64):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(in_ch, base, 4, stride=2, padding=1)), nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(base, base * 2, 4, stride=2, padding=1)), nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1)), nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(base * 4, 1, 1),
        )

    def forward(self, x):
        return self.net(x).flatten(1)


def load_master(model, blob, use_ema=False):
    weights = blob.get("ema") if use_ema else None
    if weights is None:
        weights = blob["model"]
    model.load_state_dict(weights)
    return model


@torch.no_grad()
def ddim20(model, ctx, abar, seed, size=64):
    """Deterministic 20-step DDIM with an x0-prediction model (eval)."""
    b = ctx.shape[0]
    g = torch.Generator(device=ctx.device).manual_seed(seed)
    x = torch.randn(b, 4, size, size, device=ctx.device, generator=g)
    ts = torch.linspace(0, 999, 20).round().long().flip(0)
    for i, t in enumerate(ts):
        tb = torch.full((b,), int(t), device=ctx.device, dtype=torch.long)
        x0_hat = model(x, tb, ctx).float()
        a_t, a_prev = abar[int(t)], (abar[int(ts[i + 1])] if i + 1 < len(ts) else torch.tensor(1.0, device=ctx.device))
        eps = (x - a_t.sqrt() * x0_hat) / (1 - a_t).sqrt().clamp_min(1e-8)
        x = a_prev.sqrt() * x0_hat + (1 - a_prev).sqrt() * eps
    # ⚠️ 2026-10-03 修：原来是 clamp(-1,1)。见 play/aq_play.py 的 LAT_CLAMP 说明 ——
    # 真实潜向量 std ≈ 0.835、|x| > 1 占 22%，夹 ±1 会削掉 26% 的数值。
    return x.clamp(-LAT_CLAMP, LAT_CLAMP)


def save_pngs(latents, vae, out_dir, prefix):
    from PIL import Image
    import numpy as np
    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        imgs = vae.decode((latents / SCALE).half()).sample
    imgs = (imgs.float() / 2 + 0.5).clamp(0, 1)
    files = []
    for i in range(imgs.shape[0]):
        arr = (imgs[i].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        fn = f"{prefix}_{PROMPT_SLUGS[i]}.png"
        Image.fromarray(arr).save(out_dir / fn)
        files.append(fn)
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ternary", "binary"], default="ternary")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr-g", type=float, default=5e-5)
    ap.add_argument("--lr-f", type=float, default=5e-4)   # TTUR: F = G x 10
    ap.add_argument("--lr-d", type=float, default=2e-4)
    ap.add_argument("--gan-weight", type=float, default=0.1)
    ap.add_argument("--fallback-every", type=int, default=200)
    ap.add_argument("--base", type=int, default=256)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--ckpt", type=str, default=None, help="override student checkpoint path")
    ap.add_argument("--teacher-ckpt", type=str, default=None,
                    help="teacher checkpoint (default: checkpoints\fp\final.pt; its EMA weights are used)")
    ap.add_argument("--device", type=str, default=None, help="force device (e.g. cpu for smoke)")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--size", type=int, default=64, help="latent size (64 = 512px)")
    args = ap.parse_args()

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rng = random.Random(args.seed + 7)

    _, _, abar = cosine_schedule()
    abar = abar.to(dev)

    # --- models -----------------------------------------------------------
    g_ckpt_path = Path(args.ckpt) if args.ckpt else CHECKPOINTS_DIR / args.mode / "final.pt"
    if not g_ckpt_path.exists():
        raise FileNotFoundError(f"student checkpoint not found: {g_ckpt_path}")
    g_blob = torch.load(g_ckpt_path, map_location="cpu", weights_only=False)

    # teacher: dedicated fp model (default checkpoints\fp\final.pt), EMA weights preferred
    t_ckpt_path = Path(args.teacher_ckpt) if args.teacher_ckpt else CHECKPOINTS_DIR / "fp" / "final.pt"
    if not t_ckpt_path.exists():
        raise FileNotFoundError(f"teacher checkpoint not found: {t_ckpt_path} (train fp first)")
    t_blob = torch.load(t_ckpt_path, map_location="cpu", weights_only=False)

    set_quant_mode("fp")  # teacher & F always fp
    teacher = build_model(args.base, None).to(dev)
    load_master(teacher, t_blob, use_ema=True)  # fp teacher, EMA preferred
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    f_net = build_model(args.base, None).to(dev)   # fake score, init from teacher
    load_master(f_net, t_blob, use_ema=True)

    set_quant_mode(args.mode)  # G keeps its native low-bit forward (STE on)
    G = build_model(args.base, None).to(dev)
    load_master(G, g_blob, use_ema=False)
    G.train()

    D = Discriminator(in_ch=4).to(dev)

    opt_g = torch.optim.AdamW(G.parameters(), lr=args.lr_g, weight_decay=0.0)
    opt_f = torch.optim.AdamW(f_net.parameters(), lr=args.lr_f, weight_decay=0.0)
    opt_d = torch.optim.AdamW(D.parameters(), lr=args.lr_d, weight_decay=0.0)
    bce = nn.BCEWithLogitsLoss()

    out_root = WORKSPACE / "deliverables" / ("turbo_" + args.mode)
    vae = None

    data = AquariusData(WORKSPACE / "Aquarius_cloud" / "latents",
                        WORKSPACE / "Aquarius_cloud" / "embeddings", "en")
    sample_ctx = torch.stack([data.get_embedding(i) for i in PROMPT_IDS]).to(dev).float()

    start, opt_states = 0, None
    last = CHECKPOINTS_DIR / f"turbo_{args.mode}" / "latest.pt"
    if args.resume and last.exists():
        blob = torch.load(last, map_location="cpu", weights_only=False)
        G.load_state_dict(blob["G"]); f_net.load_state_dict(blob["F"]); D.load_state_dict(blob["D"])
        opt_g.load_state_dict(blob["opt_g"]); opt_f.load_state_dict(blob["opt_f"]); opt_d.load_state_dict(blob["opt_d"])
        start = blob["step"]
        print(f"[turbo] resume from step {start}")

    teacher.eval()
    G.train(); f_net.train(); D.train()

    def maybe_vae():
        nonlocal vae
        if vae is None:
            from diffusers import AutoencoderKL
            vae = AutoencoderKL.from_pretrained(
                str(WORKSPACE / "Aquarius_cloud" / "VAE" / "vae"), torch_dtype=torch.float16).to(dev).eval()
        return vae

    def dm_loss(x_fake, t, ctx):
        """Distribution matching: F's x0-pred vs Teacher's x0-pred at the fake point.
        G-branch: grads flow through x_fake into G (F/teacher params excluded there)."""
        f_out = f_net(x_fake, t, ctx)
        t_out = teacher(x_fake, t, ctx)
        return F.mse_loss(f_out, t_out)

    t0, running = time.time(), []
    step = start
    while step < args.steps:
        x_real, ctx, _ids, _b = data.sample_batch(args.bs, rng)
        x_real = x_real.to(dev).float()
        ctx = ctx.to(dev).float()  # embeddings are bf16; fp32 keeps CPU/no-autocast paths happy
        bsz = x_real.shape[0]
        t = torch.randint(0, NUM_TIMESTEPS, (bsz,), device=dev)

        fallback = (args.fallback_every > 0 and (step + 1) % args.fallback_every == 0)

        # ---- G one-step fake sample + DM/GAN losses -----------------------
        set_quant_mode(args.mode)
        x_t_noise = torch.randn(bsz, 4, args.size, args.size, device=dev, generator=None)
        t_max = torch.full((bsz,), NUM_TIMESTEPS - 1, device=dev, dtype=torch.long)
        x_fake = G(x_t_noise, t_max, ctx)  # one-step: predict x0 straight from noise

        if fallback:
            # DMD2 drift guard: standard denoising on real data for G and F
            noise = torch.randn_like(x_real)
            a = abar[t].view(-1, 1, 1, 1)
            x_t = a.sqrt() * x_real + (1 - a).sqrt() * noise
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                pred_g = G(x_t, t, ctx)
            loss_g = F.mse_loss(pred_g.float(), x_real)
            opt_g.zero_grad(set_to_none=True); loss_g.backward(); opt_g.step()

            set_quant_mode("fp")
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                pred_f = f_net(x_t, t, ctx)
            loss_f = F.mse_loss(pred_f.float(), x_real)
            opt_f.zero_grad(set_to_none=True); loss_f.backward(); opt_f.step()
            set_quant_mode(args.mode)

            running.append(loss_g.item())
            step += 1
            if step % args.log_every == 0:
                print(f"turbo {step}/{args.steps} [FALLBACK] denoise loss {loss_g.item():.5f} "
                      f"{(time.time()-t0)/args.log_every:.2f}s/it", flush=True)
                running, t0 = [], time.time()
            continue

        # ---- D step (all detached; D never sees live G graph) ---------------
        set_quant_mode("fp")
        d_real = D(x_real.detach())
        d_fake_d = D(x_fake.detach())
        loss_d = bce(d_real, torch.ones_like(d_real)) + bce(d_fake_d, torch.zeros_like(d_fake_d))
        opt_d.zero_grad(set_to_none=True)
        loss_d.backward()
        opt_d.step()

        # ---- G step: DM + GAN in ONE backward (single pass through graph A) -
        set_quant_mode(args.mode)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            f_out = f_net(x_fake, t, ctx)      # F params get grads here -> zeroed below
            t_out = teacher(x_fake, t, ctx)    # params frozen; input-grad path only
            d_fake_g = D(x_fake)               # D params get grads here -> cleared next iter
        loss_dm = F.mse_loss(f_out.float(), t_out.float())
        loss_gan = bce(d_fake_g, torch.ones_like(d_fake_g))
        loss_g = loss_dm + args.gan_weight * loss_gan
        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()                      # frees graph A/B/C
        opt_g.step()

        # ---- F step: match teacher's x0-pred at the fake point (detached) ---
        opt_f.zero_grad(set_to_none=True)      # clear pollution from loss_g.backward
        with torch.no_grad():
            t_tgt = t_out.detach()
        set_quant_mode("fp")
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            f_out2 = f_net(x_fake.detach(), t, ctx)
        loss_f = F.mse_loss(f_out2.float(), t_tgt.float())
        loss_f.backward()
        opt_f.step()
        set_quant_mode(args.mode)

        running.append(loss_dm.item())
        step += 1

        if step % args.log_every == 0:
            dt = time.time() - t0
            print(f"turbo {step}/{args.steps} loss_dm {sum(running)/len(running):.5f} "
                  f"gan {loss_gan.item():.4f} {dt/args.log_every:.2f}s/it", flush=True)
            running, t0 = [], time.time()

        if step % args.ckpt_every == 0 or step == args.steps:
            cdir = CHECKPOINTS_DIR / f"turbo_{args.mode}"
            cdir.mkdir(parents=True, exist_ok=True)
            torch.save({"G": G.state_dict(), "F": f_net.state_dict(), "D": D.state_dict(),
                        "opt_g": opt_g.state_dict(), "opt_f": opt_f.state_dict(),
                        "opt_d": opt_d.state_dict(), "step": step, "args": vars(args)},
                       cdir / "latest.pt")
            torch.save({"G": G.state_dict(), "F": f_net.state_dict(), "D": D.state_dict(),
                        "opt_g": opt_g.state_dict(), "opt_f": opt_f.state_dict(),
                        "opt_d": opt_d.state_dict(), "step": step, "args": vars(args),
                        "mode": args.mode, "kind": "turbo"},
                       cdir / f"ckpt_{step:06d}.pt")
            print(f"[turbo ckpt] step {step} -> {cdir}", flush=True)

        # periodic progress samples: G one-step vs teacher 20-step, same noise
        if step % 1000 == 0 or step == args.steps:
            G.eval()
            vae = maybe_vae()
            torch.manual_seed(args.seed + step)
            g_one = G(torch.randn(len(PROMPT_IDS), 4, args.size, args.size, device=dev,
                                  generator=torch.Generator(device=dev).manual_seed(args.seed)),
                      torch.full((len(PROMPT_IDS),), NUM_TIMESTEPS - 1, device=dev, dtype=torch.long),
                      sample_ctx)
            f1 = save_pngs(g_one.clamp(-LAT_CLAMP, LAT_CLAMP), vae,
                           out_root / "samples" / f"step{step:05d}",
                           f"step{step:05d}_1step")
            G.train()
            teacher.eval()
            t20 = ddim20(teacher, sample_ctx, abar, seed=args.seed)
            f2 = save_pngs(t20, vae, out_root / "teacher_ref" / f"step{step:05d}", f"step{step:05d}_ddim20")
            teacher.eval() if False else None
            print(f"[turbo samples] step {step}: {len(f1)} one-step + {len(f2)} teacher-20step -> {out_root}",
                  flush=True)

    # final deliverable samples
    G.eval()
    vae = maybe_vae()
    g_one = G(torch.randn(len(PROMPT_IDS), 4, args.size, args.size, device=dev,
                          generator=torch.Generator(device=dev).manual_seed(args.seed)),
              torch.full((len(PROMPT_IDS),), NUM_TIMESTEPS - 1, device=dev, dtype=torch.long),
              sample_ctx)
    files = save_pngs(g_one.clamp(-LAT_CLAMP, LAT_CLAMP), vae, out_root / "samples", "final_1step")
    print(f"[turbo DONE] {args.mode} final one-step samples: {[str(out_root / 'samples' / f) for f in files]}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
