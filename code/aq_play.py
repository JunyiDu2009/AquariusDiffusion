# -*- coding: utf-8 -*-
"""Aquarius ternary text-to-image -- local inference / play script.

Design notes (all of them exist to fit an 8 GB card):
  1. **Time-multiplexed VRAM**: TE (0.8B), UNet (558M) and VAE are never
     resident at the same time. Encode text -> unload TE -> load UNet and
     sample -> unload -> load VAE and decode.
     Peak VRAM ~ max(TE ~1.6 GB, UNet+VAE ~2.6 GB) + activations, which
     leaves plenty of headroom on an 8 GB card.
  2. **Text handling matches training exactly**: raw text + max_length=32 +
     padding + truncation, taking last_hidden_state. This was **reverse
     engineered from measurements** (cos = 0.99909 against the precomputed
     embeddings), not guessed -- if text handling diverges from training the
     output breaks completely and reports no error at all.
  3. Sampling uses deterministic DDIM (eta=0) with a cosine noise schedule,
     the same as during training.

Usage:
  python aq_play.py "a cat sitting on a chair"
  python aq_play.py "sunset over the ocean" --seed 42 --steps 30
  python aq_play.py "mountains in fog" --res 512x640     # width 512 x height 640 (portrait)
  python aq_play.py --prompts-file my_prompts.txt --out ./my_pics

Size: every pixel dimension must be a **multiple of 64** (the model needs a
latent divisible by 8). Legal sides are 256/320/384/448/512/576/640; the 22
buckets actually used in training are listed in TRAINED_BUCKETS. Sizes outside
those 22 still run, but may degrade noticeably.

NOTE: the bundled checkpoint is **step 260,000 = 28.70 epochs** -- deliberately stopped
there (clip_ratio saturated + rental budget exhausted; see the technical report section
7.1).  Output is still **blurry scene composition** with no recognizable objects.
be **blurry scene composition** (you can see sky / ground layering) with **no
recognizable objects**. That is expected behaviour, not a broken script.
"""
import argparse
import gc
import contextlib
import os
import random
import re
import sys
import time
from pathlib import Path

import torch

# NOTE: the Windows console is GBK (cp936): printing a character that GBK
#       cannot encode raises UnicodeEncodeError and **kills the process
#       outright**. This file and app.py both run in a user console, so
#       downgrade encoding errors to '?' here; nothing can crash it. This only
#       affects logging, never any computation.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except Exception:                                         # noqa: BLE001
        pass

HERE = Path(__file__).resolve().parent
WS = HERE.parent if (HERE.parent / "Aquarius_cloud").is_dir() else HERE


def _cuda_ok():
    """Is CUDA **really** usable on this machine?

    NOTE: `torch.cuda.is_available()` alone is not enough -- with
    `CUDA_VISIBLE_DEVICES=""` set it **still returns True**, but
    `device_count() == 0`, so `get_device_properties(0)` raises
    `AssertionError: Invalid device id` (actually hit on 2026-10-04, wasted a
    whole debugging round).
    """
    try:
        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    except Exception:                                         # noqa: BLE001
        return False


def _pick(*cands):
    """Return the first existing path -- works for both the repo layout and the
    standalone delivery folder."""
    for c in cands:
        if c.is_dir():
            return c
    return cands[0]


# Standalone folder: TE/, VAE/, checkpoints/ sit right next to the script.
# Repo: TE/VAE live under Aquarius_cloud/, checkpoints at the root.
# The delivery package ("AquariusTerimage") uses **ASCII category folders**:
# text_encoder/ , diffusion_model/ , VAE/ , and VAE/ holds **exactly one
# .safetensors** (per the "model file only" requirement).
# Both layouts are listed below so the repo and the package share one code path.
TE_DIR = _pick(HERE / "text_encoder",
               HERE / "TE" / "Qwen3.5-0.8B",
               WS / "TE" / "Qwen3.5-0.8B",
               WS / "Aquarius_cloud" / "TE" / "Qwen3.5-0.8B")
VAE_DIR = _pick(HERE / "VAE",
                HERE / "VAE" / "vae",
                WS / "VAE" / "vae",
                WS / "Aquarius_cloud" / "VAE" / "vae")
CKPT_DIR = _pick(HERE / "diffusion_model",
                 HERE / "checkpoints" / "ternary",
                 WS / "checkpoints" / "ternary")

# SD1.5 VAE structure config (copied verbatim from
# Aquarius_cloud/VAE/vae/config.json) -- used to build the model when the
# delivery package ships no config.json. See load_vae().
_VAE_CFG = {
    "_class_name": "AutoencoderKL", "act_fn": "silu",
    "block_out_channels": [128, 256, 512, 512],
    "down_block_types": ["DownEncoderBlock2D"] * 4,
    "up_block_types": ["UpDecoderBlock2D"] * 4,
    "in_channels": 3, "out_channels": 3, "latent_channels": 4,
    "layers_per_block": 2, "norm_num_groups": 32, "sample_size": 512,
}

MAX_TOKENS = 32
SCALE = 0.18215
DDIM_STEPS = 20

# NOTE: numerical guard at the end of sampling. **Do not set this back to 1.0**
#       (corrected 2026-10-03).
#
# Measured (512x640 bucket, 512 real latents drawn at random across shards):
#     per-channel std [0.875, 0.977, 0.707, 0.737], overall std 0.835,
#     **22.14% of values have |x| > 1**
# The model's own x0 prediction: overall std 0.872, **25.92% with |x| > 1**.
#
# => `clamp(-1, 1)` truncates at about **+-1.2 sigma**, **systematically
#    discarding ~26% of the values** and pressing the output variance down to
#    **0.61x** -> grey, washed-out images with lost contrast.
#    Worse, it is **inconsistent with the training objective** -- x0 was never
#    clamped during training.
#
# +-6 sigma is effectively "no clamp" and exists only to catch NaN / blow-ups.
# To reproduce the old behaviour:   set AQ_LAT_CLAMP=1
LAT_CLAMP = float(os.environ.get("AQ_LAT_CLAMP", "6.0"))

# NOTE: cudnn.benchmark **must stay False** by default. Do not "optimize" it
#       back to True.
#
# It makes cuDNN try several convolution algorithms per tensor shape and pick
# the fastest -- that costs about 45 seconds, and **the result is stored in
# thread-local storage (TLS), i.e. it is private to each thread**.
# Gradio dispatches a new thread per request => **every single generation pays
# that 45 seconds again**.
#
# Measured on the target machine (RTX 5060 8GB / Windows), 640^2 / 20 steps:
#     main thread            4.1s -> 4.2s   (with benchmark on)
#     worker thread          49.4s          <- what the user actually feels
#     a fresh thread         47.6s
#     same thread, 2nd call  49.1s -> 4.0s  (only then is the cache warm)
#     benchmark off          4.1s -> 4.5s   (identical on any thread)
# -> **on this model benchmark is pure cost with zero benefit**.
# "Warm up at startup" does not help either: the warm-up runs on the main
# thread and TLS is not shared across threads.
#
# Escape hatch: only set AQ_CUDNN_BENCH=1 when you have confirmed a
# "fixed shape + single long-running thread" workload.
CUDNN_BENCH = os.environ.get("AQ_CUDNN_BENCH", "0") == "1"


def apply_cudnn_policy():
    torch.backends.cudnn.benchmark = CUDNN_BENCH
    if CUDNN_BENCH:
        print("[env] cudnn.benchmark = True (AQ_CUDNN_BENCH=1; only worth it for"
              " single-threaded fixed-shape runs)",
              flush=True)
RES = 512                 # default output side length (square)
OUT_DIR = HERE / "gradio_out"      # output dir for the Gradio UI (CLI uses --out)

# Legal output-size grid: every pixel dimension must be a multiple of 64
# (see the note in ddim)
RES_MIN, RES_MAX, RES_STEP = 256, 640, 64
RES_GRID = list(range(RES_MIN, RES_MAX + 1, RES_STEP))    # 256,320,384,448,512,576,640

# The 22 buckets training actually used, written as **(H, W)** -- matching the
# Aquarius_cloud/latents/latents_{H}x{W}_*.safetensors filenames.
# **Careful: this is HxW, the reverse of the "width x height" order used in the
# UI.** Do not mix the two up.
TRAINED_BUCKETS = [
    (256, 512), (256, 640), (320, 512), (320, 640), (384, 512), (384, 640),
    (448, 512), (448, 640), (512, 256), (512, 320), (512, 384), (512, 448),
    (512, 512), (512, 640), (576, 640), (640, 256), (640, 320), (640, 384),
    (640, 448), (640, 512), (640, 576), (640, 640),
]


def parse_res(s):
    """Parse '512' / '512x640' / '512x640' into (h, w). **'width x height'.**

    Example: '512x640' -> (640, 512), i.e. width 512, height 640 (portrait).
    """
    t = str(s).strip().lower().replace("*", "x").replace("\u00d7", "x").replace(" ", "")
    if "x" in t:
        a, b = t.split("x", 1)
        w, h = int(a), int(b)
    else:
        w = h = int(t)
    return h, w


def check_res(h, w):
    """Size legality plus a hint about whether it is inside the training
    distribution. Returns (ok, message)."""
    for name, v in (("height", h), ("width", w)):
        if v % 64:
            return False, f"{name} {v} is not a multiple of 64 (the latent must divide by 8)"
        if v < 64:
            return False, f"{name} {v} is too small"
    note = ""
    if (h, w) not in TRAINED_BUCKETS:
        near = sorted(TRAINED_BUCKETS, key=lambda b: abs(b[0] - h) + abs(b[1] - w))[:2]
        note = (f"  [WARN] {w}x{h} is not one of the 22 trained buckets and may"
                f" degrade noticeably. Nearest buckets: "
                + ", ".join(f"{b[1]}x{b[0]}" for b in near))
    if max(h, w) > RES_MAX:
        note += (f"  [WARN] beyond the training cap {RES_MAX}; output degrades"
                 f" further and runs slower.")
    return True, note



# --------------------------------------------------------------------------
# Noise schedule: identical to training (cosine, see aq_train.cosine_schedule)
# --------------------------------------------------------------------------
def cosine_schedule(T=1000, s=0.008):
    import math
    tt = torch.arange(T + 1, dtype=torch.float64) / T
    f = torch.cos((tt + s) / (1 + s) * math.pi / 2) ** 2
    abar = (f / f[0]).float().clamp(1e-8, 1.0 - 1e-8)
    return abar


# --------------------------------------------------------------------------
# Text encoding: exactly the contract used to precompute embeddings in training
# --------------------------------------------------------------------------
def load_te(device):
    """Load the text encoder, returns (tok, model).

    NOTE: **invariant** -- the text encoder must match exactly the one used to
    precompute caption embeddings during training (raw prompt +
    max_length=32 + padding="max_length" + truncation, take
    last_hidden_state, no chat template, no prefix). Swapping precision or
    weights makes the conditioning fall into a different distribution --
    **the image breaks and nothing is reported**. This was reverse engineered
    from measurements (cosine 0.99909); do not trust memory on it.

    MEASURED THREE TIMES ON 2026-10-03 -- note that the line below is NOT a
    necessary condition for images not to break.

    (a) `code/_t_te_precision_ladder.py`: ctx cos of each quantization scheme
          fp32 vs bf16   0.99919   <- **that 0.99909 line is exactly the
                                      bf16<->fp32 difference**
          int8 per-ch    0.99332, fp8 e4m3 0.98263, int4 per-ch 0.49291
    (b) `code/_t_te_tolerance_sweep.py` (decisive; overturns how (a) was used)
          Perturb the conditioning vector along the **real quantization error
          direction**:
            - cos drops all the way to **0.609** and images are **still fine**
              (CLIP 0.269 -> 0.270, still visibly a pizza)
            - a **random direction** at cos **0.996** already collapses into a
              **flat red block** (CLIP 0.269 -> 0.193)
          => **cos is not the criterion; the error direction is.** Using cos as
             a yardstick leads to a wrong "not feasible" conclusion.
    (c) `code/_t_te_quant_end2end.py` (really swaps the TE: 5 captions x
        20 DDIM steps)
            - **int8 per-channel: L1 ~ 1.9/255, CLIP matches bf16 => usable**
              (TE 1666 MB -> 833 MB, a 50% saving)
            - int4 per-channel: L1 12~22, CLIP drops 0.05 on some captions
              => naive RTN int4 is unusable (so the MLX nvfp4 route stays dead)

    => the **real meaning of `dtype=torch.bfloat16` here is a training
       contract**: caption embeddings were precomputed with a bf16 TE
       (`Aquarius_cloud/embeddings/`), so inference must land in the **same
       conditioning distribution**. **What must not be touched is swapping in
       a different set of weights / a different encoding protocol** (chat
       template, prefix, max_length, another model) -- that moves the
       conditioning into a completely different distribution and **breaks the
       image without any error**. Floating-point precision on the same weights
       is nearly insensitive, and low-bit quantization along the real error
       direction is far gentler than cos suggests.
    => to **save VRAM** use `AQ_TE_MODE=cache` (`Engine.te_plan()`), not dtype.
       To **save disk** int8 works (measured usable), but should be re-checked
       with more captions once training is finished.
    """
    # The delivery package's text_encoder/ is an **int8 slim build** (it has a
    # quant.json) => it must go through its own loader; transformers
    # from_pretrained cannot read those weights. The maths is **bit-identical**
    # to the bf16 original (see MEMORY.md section 7b); what it saves is disk and RAM.
    if (TE_DIR / "quant.json").is_file():
        try:
            from load_te import load_te as _load_slim           # same dir as aq_play.py
        except ImportError:                                      # fallback: load by path
            import importlib.util as _ilu
            _sp = _ilu.spec_from_file_location("_aq_slim_te", HERE / "load_te.py")
            _md = _ilu.module_from_spec(_sp)
            _sp.loader.exec_module(_md)
            _load_slim = _md.load_te
        _mode = os.environ.get("AQ_TE_PREC", "int8").lower()
        tok, model = _load_slim(str(TE_DIR), device="cpu", dtype=torch.bfloat16,
                                mode=_mode, verbose=True)
        if str(device) != "cpu":
            model = model.to(device)
        return tok, model
    from transformers import AutoTokenizer, AutoModel
    print(f"[te] loading {TE_DIR.name} (slow the first time, then reused and"
          f" no longer read from disk) ...", flush=True)
    tok = AutoTokenizer.from_pretrained(str(TE_DIR), trust_remote_code=True)
    model = AutoModel.from_pretrained(str(TE_DIR), trust_remote_code=True,
                                      dtype=torch.bfloat16).to(device).eval()
    return tok, model


def encode_texts(prompts, device, tok=None, model=None):
    """Encode text once.

    With model=None it loads the TE itself and unloads it afterwards (CLI
    path); passing tok/model reuses them (Engine path) -- **no longer re-reading
    1.67 GB from disk every time**.

    Input tensors go on the **device the model actually lives on** (never infer
    it from the argument), otherwise cache mode crashes with "CPU weights,
    cuda input".
    """
    own = model is None
    if own:
        tok, model = load_te(device)
    md = next(model.parameters()).device            # read the real device, do not guess
    out = []
    with torch.no_grad():
        for p in prompts:
            b = tok([p], return_tensors="pt", max_length=MAX_TOKENS,
                    truncation=True, padding="max_length").to(md)
            h = model(**b).last_hidden_state[0]        # (32, 1024)
            out.append(h.float().cpu())
    ctx = torch.stack(out).to(device)                  # (N, 32, 1024)
    print(f"[te] encoding done, ctx {tuple(ctx.shape)} (computed on {md})", flush=True)
    if own:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return ctx


def _step_of(p):
    """Parse the training step out of a filename so the newest export wins.
    Returns -1 when it cannot be parsed.

    Convention: every packed artifact carries a step suffix, e.g.
    `aquarius_ternary_step40000.safetensors` -- otherwise two days later nobody
    can tell which checkpoint a model came from (explicit user request,
    2026-10-02).
    """
    nums = re.findall(r"(\d{3,})", Path(p).stem)
    return max((int(x) for x in nums), default=-1)


def default_ckpt():
    """Default model file: **prefer the packed delivery model** -- the prebuilt runtime
    int4 (`diffusion_model/`) when it is there, otherwise the base-3 **master**
    (`portable/`), from which the runtime kernel is **rebuilt at load time, per
    device** (int4 fused kernel on CUDA, unpack-to-float elsewhere) -- then fall back
    to an fp16 slim checkpoint, then a full checkpoint.

    There may be **several exports at different steps**
    (`..._step40000.safetensors`, `..._step50000.safetensors`, ...); this
    **automatically picks the one with the highest step** -- so keeping older
    versions around in the same folder is harmless and they will not interfere.

    Both `Engine` (Gradio) and the CLI `main()` go through this single
    function -- on 2026-10-02 they each had their own copy, with the CLI
    hard-coding latest.pt, which produced exactly the inconsistency of "the UI
    uses the packed model while the command line uses a 1 GB checkpoint".
    """
    cands = []
    # Priority order: prebuilt runtime int4 first (zero-conversion load), then the
    # base-3 master -- from which the runtime kernel is rebuilt at load time, per
    # device.  The **first folder that actually holds an export wins**, so a package
    # shipping only `portable/` works out of the box.
    for d in (HERE / "diffusion_model", HERE / "portable", HERE / "model", CKPT_DIR):
        if d.is_dir():
            found = [p for p in d.glob("*.safetensors") if _step_of(p) > 0]
            if found:
                cands = found
                break
    if cands:
        return max(cands, key=_step_of)
    for cand in (CKPT_DIR / "latest.pt", CKPT_DIR / "final.pt"):
        if cand.is_file():
            return cand
    return CKPT_DIR / "latest.pt"


# --------------------------------------------------------------------------
# UNet loading (two sources: the ema in a training checkpoint, or a packed
# delivery model)
# --------------------------------------------------------------------------
def _add_code_path():
    """Put the folder holding aq_unet.py / aq_lowbit.py on sys.path (works for
    both the repo layout and the standalone package)."""
    for cand in (HERE, WS / "code", HERE.parent / "code"):
        if (cand / "aq_unet.py").is_file():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            return cand
    return None


def load_unet(ckpt_path, device, kernel=None):
    """Load the UNet. Two file kinds are auto-detected:

      *.safetensors  a packed delivery model (product of aq_pack.py /
                     aq_pack_int4.py) -- 153.0 MB (base-3) or 323.5 MB
                     (runtime int4). The weights are already the final
                     quantized values, so the forward pass **does not
                     re-quantize**
      *.pt           a training checkpoint; only its ema is read

    The two are mathematically equivalent (see the notes in aq_lowbit.py), but
    the packed version is slightly more accurate: the ema inside a checkpoint
    is stored in fp16, which flips 0.005% of the ternary codes, whereas the
    packed version is code-exact.

    `kernel` -- the sampling backend (**only meaningful for packed files**):
      None / "fp"   default. Unpack to floating-point weights and run
                    `F.conv2d / F.linear`. Saves disk, but **VRAM and
                    bandwidth are both floating point** (fp32 2223 MB /
                    bf16 1112 MB).
      "int4"        low-bit fused kernel: weights stay resident as **int4
                    (287 MB)** and unpacking happens inside the GEMM. Goes
                    through `aq_kernel.build_model_from_packed()` -- which
                    **builds straight from the packed file** without ever
                    pushing a bf16 UNet onto the card; measured build peak
                    1158 -> 362 MB and pixel-identical output.
                    NOTE: needs CUDA (the CPU version of `_weight_int4pack_mm`
                    is far slower and not production-viable).
    """
    _add_code_path()
    from aq_unet import AquariusUNet, set_quant_mode
    import aq_lowbit as lb

    ckpt_path = Path(ckpt_path)
    _kern = str(kernel or os.environ.get("AQ_KERNEL", "")).lower()

    if ckpt_path.suffix.lower() == ".safetensors" or lb.is_packed(ckpt_path):
        # **Let the file speak for itself**: a runtime int4 file
        # (format=aquarius-int4-cuda) carries its own "needs no conversion"
        # marker, so the caller does not have to pass kernel -- otherwise the
        # Gradio engine path (which passes no kernel) would trip over it.
        _is_int4_file = False
        if not _kern:
            try:
                _is_int4_file = (lb.read_meta(ckpt_path).get("format") == "aquarius-int4-cuda")
            except Exception:                                 # noqa: BLE001
                _is_int4_file = False
            if _is_int4_file:
                _kern = "int4"
        else:
            try:
                _is_int4_file = (lb.read_meta(ckpt_path).get("format") == "aquarius-int4-cuda")
            except Exception:                                 # noqa: BLE001
                _is_int4_file = False
        if _kern == "int4":
            import aq_kernel as AK
            # Automatic downgrade: the delivery file is **device independent**
            # (base-3 codes + fp16 scales), but the **runtime kernel is device
            # dependent**. Measured operator coverage:
            #   - `_weight_int4pack_mm` -> **CUDA only** (NotImplementedError on CPU)
            #   - `_weight_int8pack_mm` -> has a CPU version, but **per-channel
            #     only** (it forces a 1-D size N)
            # Our quantization is **group=128** => on CPU there is **no usable
            # low-bit fused kernel at all**, only unpacking to float and running
            # `F.conv2d/linear`. This is not something we skipped; torch's CPU
            # operators simply do not support group quantization.
            # => so **downgrade instead of raising**: the same file still runs
            #    on CPU, it just goes from 323 MB resident to about 1.1 GB.
            #    (`aq_kernel.backend_for()` could already compute this choice,
            #    it just was not wired into the loader.)
            if not str(device).startswith("cuda"):
                if _is_int4_file:
                    # The runtime int4 file contains **no** base-3 codes => the
                    # weights cannot be rebuilt on CPU. This is not a downgrade
                    # question, it is the scope of this particular file (see the
                    # notes in aq_pack_int4.py).
                    raise RuntimeError(
                        "This model is a 'runtime int4 layout' conversion-free file and can "
                        "**only be loaded on CUDA**: it stores the GPU int4pack layout and "
                        "contains no base-3 codes, so with no fused kernel on CPU "
                        "(_weight_int4pack_mm is CUDA-only) the weights cannot be rebuilt.\n"
                        "  - Have an NVIDIA card: add --device cuda (or let it auto-select)\n"
                        "  - Want CPU: use the base-3 delivery file instead (device "
                        "independent; CPU unpacks it to fp32 automatically)")
                print(f"[unet] kernel=int4 needs CUDA (got {device!r})"
                      f" -> automatically downgrading to the 'unpack to float' path."
                      f" No fused kernel on CPU: int4pack is CUDA-only, and the CPU"
                      f" int8pack only supports per-channel while we are group=128",
                      flush=True)
            else:
                model, st = AK.build_model_from_packed(
                    ckpt_path, device=device, dtype=torch.bfloat16, selfcheck=True,
                    prepacked=(ckpt_path if _is_int4_file else None))
                if st.get("bad"):
                    raise RuntimeError(f"int4 per-layer self-check failed on {len(st['bad'])} "
                                       f"layer(s): {st['bad'][:2]}")
                print(f"[unet] int4 fused kernel (direct build) - {st['linear']}L + {st['conv']}C - "
                      f"skipped {st['skipped']} - never pushed bf16 onto the card", flush=True)
                return model
        model = AquariusUNet(dict(block_out_channels=[256, 512, 1024, 1024],
                                  cross_attention_dim=1024, heads=8))
        info = lb.load_packed_into(model, ckpt_path)     # calls set_quant_mode("fp") internally
        model = model.to(device).eval()
        model._aq_ckpt = {"step": info["step"],
                          "weights": f"packed-{info['mode']} (bitpacked -> float forward)",
                          "path": str(ckpt_path),
                          "packed": True,
                          "bpw": info["bpw"]}
        return model

    set_quant_mode("ternary")
    model = AquariusUNet(dict(block_out_channels=[256, 512, 1024, 1024],
                              cross_attention_dim=1024, heads=8))
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    w = blob.get("ema") or blob.get("model")
    if not w:
        raise ValueError(f"{ckpt_path} contains neither ema nor model")
    model.load_state_dict(w)
    model = model.to(device).eval()
    # Hang checkpoint metadata on the instance (a plain attribute: not in
    # state_dict, not part of the forward pass)
    model._aq_ckpt = {"step": blob.get("step"),
                      "weights": "ema" if blob.get("ema") else "raw",
                      "path": str(ckpt_path),
                      "packed": False}
    print(f"[unet] {ckpt_path.name}  step={blob.get('step')}  "
          f"weights={'ema' if blob.get('ema') else 'raw'}", flush=True)
    return model



# --------------------------------------------------------------------------
# VAE loading -- optimization 2: decoding is the second biggest VRAM peak,
# right after "moving weights onto the card"
# --------------------------------------------------------------------------
# Measured (2026-10-04, 256^2 decode, ledger from `_t_bench_stage.py`):
#     UNet 20-step DDIM delta   **+95.4 MB**   <- tiny, left alone
#     VAE decode delta          **+503.3 MB**  <- 5x more expensive than a UNet
#                                                forward, and it is genuinely its
#                                                own forward pass
# The cause is the upsampling path in the VAE decoder holding intermediates on
# the order of 256x256x512.
#
# This model's VAE is the SD1.5 one, 83.65M parameters => fp32 **334.6 MB** /
# fp16 **167.3 MB**. The training and sampling pipelines are fp16/bf16
# throughout already (`aq_sample.py`, `Engine.load_vae`), so **fp16 is the
# default**; `AQ_VAE_DTYPE=fp32` reverts (useful for control experiments).
#
# `AQ_VAE_TILING=1` enables tiled decoding (diffusers `enable_tiling()`) --
# another step down in VRAM, but pointless at small resolutions and it can
# introduce seams => **off by default**, decided by the caller per resolution.
VAE_DTYPE = os.environ.get("AQ_VAE_DTYPE", "fp16")
VAE_TILING = os.environ.get("AQ_VAE_TILING", "0") == "1"
_TORCH_DTYPES = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}


def load_vae(device, dtype=None, tiling=None, verbose=True):
    """Load the VAE. `dtype` is fp16 (default) / fp32 / bf16; `tiling=True`
    enables tiled decoding."""
    from diffusers import AutoencoderKL
    _name = str(dtype or VAE_DTYPE).lower()
    dt = _TORCH_DTYPES[_name]
    if (VAE_DIR / "config.json").is_file():
        vae = AutoencoderKL.from_pretrained(str(VAE_DIR), torch_dtype=dt)
    else:
        # In the delivery package VAE/ holds **exactly one .safetensors** (no
        # config.json) => build the structure from the inline config.
        from safetensors.torch import load_file as _slf
        f = next((q for q in sorted(VAE_DIR.glob("*.safetensors"))), None)
        if f is None:
            raise FileNotFoundError(f"{VAE_DIR} has no .safetensors and no config.json")
        vae = AutoencoderKL.from_config(_VAE_CFG)
        # NOTE: **old-style key migration is mandatory**: this SD1.5 VAE uses
        #       the legacy attention names (`query/key/value/proj_attn`) while
        #       diffusers 0.40's AutoencoderKL expects
        #       `to_q/to_k/to_v/to_out.0`. `from_pretrained` normally does this
        #       for you; a direct `load_state_dict` reports 16 missing / 16
        #       unexpected keys (actually hit).
        _sd = {}
        for _k, _v in _slf(str(f)).items():
            _nk = _k
            for _a, _b in ((".query.", ".to_q."), (".key.", ".to_k."),
                           (".value.", ".to_v."), (".proj_attn.", ".to_out.0.")):
                if _a in _nk:
                    _nk = _nk.replace(_a, _b)
                    break
            _sd[_nk] = _v
        vae.load_state_dict(_sd, strict=True)
        vae = vae.to(dt)
    vae = vae.to(device).eval()
    tl = VAE_TILING if tiling is None else bool(tiling)
    if tl:
        vae.enable_tiling()
    if verbose:
        print(f"[vae] {_name} - tiling={tl} - weights "
              f"{sum(p.numel() * p.element_size() for p in vae.parameters()) / 1e6:.1f} MB",
              flush=True)
    return vae


# --------------------------------------------------------------------------
# Sampling: deterministic DDIM (eta=0), x0 prediction
# --------------------------------------------------------------------------
def ddim_prepare(abar, steps, device):
    """Do **all host-side / transfer** work of DDIM up front.

    **CUDA Graph capture must run this first**: during capture **no H2D copy is
    allowed at all**, and the failure message is the generic
    `operation failed due to a previous error during capture` (it does not say
    which operator). The two real culprits found by bisecting section by
    section on 2026-10-04:
      1. `abar[cpu_ts]` -- gathering a CUDA tensor with a **CPU index tensor**
         makes PyTorch do an H2D copy internally;
      2. `torch.tensor(1.0, device=device)` (only reached on the last
         iteration) -- another H2D copy.
      Also: `int(t)` inside the loop on a device tensor triggers a
      **device -> host sync**, equally forbidden.

    Returns a dict (`ts` is a **plain Python int list**; both coefficient
    tensors are already on the device).
    """
    ts_cpu = torch.linspace(0, 999, steps).round().long().flip(0)
    ts_list = [int(v) for v in ts_cpu]
    ab = abar.to(device)
    a_ts = ab[torch.tensor(ts_list, device=device)].to(torch.float32)   # [steps] (one gather)
    a_ps = torch.cat([a_ts[1:], torch.ones(1, device=device, dtype=torch.float32)])
    return dict(ts=ts_list, a_ts=a_ts, a_ps=a_ps)


def ddim(model, ctx, abar, steps, seed, device, h, w=None, x_init=None, prep=None):
    """Deterministic DDIM (eta=0), x0 prediction.

    h / w are **pixel** sizes (both divided by 8 to get the latent). w=None
    means w = h (square). The size constraint comes from
    `AquariusUNet.forward`: the latent's spatial dims must be multiples of 8,
    i.e. **pixel dims must be multiples of 64**. Legal sides are therefore
    256 / 320 / 384 / 448 / 512 / 576 / 640 ...

    `x_init` -- supply the initial latent noise directly (shape
    `[b,4,h/8,w/8]`), **bypassing the RNG**.
    `prep`   -- the product of `ddim_prepare()`. **Both are required when
                capturing a CUDA Graph**: the graph may contain no RNG, no H2D
                and no device->host sync.
    """
    if w is None:
        w = h
    h, w = int(h), int(w)
    lh, lw = h // 8, w // 8
    if lh % 8 or lw % 8:
        raise ValueError(
            f"resolution must be a multiple of 64 (the latent must divide by 8), got {w}x{h}"
            f" -> latent {lw}x{lh}")
    b = ctx.shape[0]
    if x_init is None:
        g = torch.Generator(device=device).manual_seed(seed)
        x = torch.randn(b, 4, lh, lw, device=device, generator=g)
    else:
        x = x_init
        if tuple(x.shape) != (b, 4, lh, lw):
            raise ValueError(f"x_init shape {tuple(x.shape)} != expected {(b, 4, lh, lw)}")
    if prep is None:
        prep = ddim_prepare(abar, steps, device)
    ts_list, a_ts, a_ps = prep["ts"], prep["a_ts"], prep["a_ps"]
    tb = torch.zeros(b, device=device, dtype=torch.long)     # reuse a buffer (no torch.full per step)
    x0 = None
    with torch.no_grad():
        for i, t in enumerate(ts_list):
            tb.fill_(t)                                       # plain Python int, no sync
            # NOTE: when not on CUDA **do not pass "cuda" to autocast at all**:
            #       torch's constructor **still** evaluates
            #       `is_bf16_supported()` even with `enabled=False`, which on a
            #       0-device machine raises `AssertionError: Invalid device id`
            #       (actually hit on 2026-10-04). => use nullcontext off CUDA.
            with (torch.autocast("cuda", dtype=torch.bfloat16)
                  if device == "cuda" else contextlib.nullcontext()):
                x0 = model(x, tb, ctx)
            x0 = x0.float()
            a_t, a_p = a_ts[i], a_ps[i]
            eps = (x - a_t.sqrt() * x0) / (1 - a_t).sqrt().clamp_min(1e-8)
            x = a_p.sqrt() * x0 + (1 - a_p).sqrt() * eps
    # NOTE: fixed 2026-10-03: this used to be clamp(-1,1); see the measured
    #       notes next to LAT_CLAMP at the top of the file.
    _lim = LAT_CLAMP
    return (x.clamp(-_lim, _lim),
            (x0.clamp(-_lim, _lim) if len(ts_list) == 1 else None))


def decode(latents, device, vae=None, dtype=None, tiling=None):
    """With vae=None it loads the VAE itself and unloads it afterwards (CLI
    path); pass an instance to reuse it (the resident Gradio path).

    Optimization 2: goes through `load_vae()` (**fp16 by default**, tunable via
    `dtype=` / `AQ_VAE_DTYPE`). This used to hard-code
    `torch_dtype=torch.float16` while `_t_bench_stage.py` loaded a separate
    fp32 copy -- that +503.3 MB peak in the ledger came from the fp32 one.
    """
    own = vae is None
    if own:
        vae = load_vae(device, dtype=dtype, tiling=tiling)
        print("[vae] decoding ...", flush=True)
    with torch.no_grad():
        z = (latents / SCALE).to(next(vae.parameters()).dtype)
        imgs = vae.decode(z).sample
    imgs = (imgs.float() / 2 + 0.5).clamp(0, 1)
    if own:
        del vae
        gc.collect()
        torch.cuda.empty_cache()
    return imgs


def slug(s, n=40):
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s.strip()).strip("_").lower()
    return (s[:n] or "prompt")


# ==========================================================================
# Resident engine -- reused by the Gradio UI (app.py). The CLI path never
# touches this.
# ==========================================================================
class Engine:
    """On-demand / optionally resident inference engine.

    VRAM ledger -- **planning estimate** (cache=True / te_mode=resident, the
    first choice on an 8 GB card):
        UNet 2.23 GB + VAE 0.32 GB + TE 1.60 GB  -- **all three resident**
        -> about 4.2 GB.  Note this is the **same** peak as the old design:
           even though the old design unloaded the TE right after use, it was
           anyway resident at the same time as the UNet, so keeping it resident
           costs no extra VRAM while removing the **1.67 GB disk re-read per
           generation** (10-20 s each, paid every single time in an interactive UI).

    The above assumes fp32/bf16 materialisation.  **Measured** on the shipped
    int4 build (RTX 5090, 512x512, 20 steps, `torch.cuda.max_memory_allocated`):
        te_mode=resident   1.34-1.44 s/image   peak 1.92 GB   <- default
        te_mode=cache      2.27-2.51 s/image   peak 1.34 GB
    So residency costs **+0.58 GB and buys ~40% wall-clock**.  Use `cache` only
    when under ~3 GB of free VRAM is available.

    te_mode="cache" (low VRAM): TE weights stay in **host RAM** and are moved
        to VRAM for encoding and back afterwards. Measured PCIe round trip is
        1.17-1.5 s, far better than re-reading from disk, but still ~0.9 s per
        image slower than just leaving it resident.
    cache=False: UNet/VAE are not resident either and reload per image -- only
        for cards below 6 GB.

    NOTE: **never run two instances at once**; VRAM would stack.
    """

    def __init__(self, device="auto", cache=True, ckpt=None, te_mode=None):
        if device in ("cpu", "cuda"):
            # NOTE: cuda explicitly requested but no usable CUDA on this machine
            #       (e.g. CUDA_VISIBLE_DEVICES="") => fall back to CPU right
            #       away with an explanation, instead of blowing up later in
            #       device_info() with `AssertionError: Invalid device id`
            #       (actually hit on 2026-10-04).
            self.device = device
            if device == "cuda" and not _cuda_ok():
                self.device = "cpu"
                print("[engine] cuda requested but _cuda_ok() is false -> falling back to CPU",
                      flush=True)
        else:
            self.device = "cuda" if _cuda_ok() else "cpu"

        self.cache = bool(cache)
        self.ckpt = Path(ckpt) if ckpt else default_ckpt()
        if not self.ckpt.is_file():
            raise FileNotFoundError(
                f"model file not found: {self.ckpt}\n"
                f"(expected a packed model in diffusion_model/ or portable/, or a "
                f"checkpoint in checkpoints/ternary/latest.pt)")

        self.abar = cosine_schedule()
        self._unet = None
        self._vae = None
        self._te = None
        self._tok = None
        # ---- decide once, then only read (never recompute per call) ---------
        # Counter-example in rework doc P0-1: using "free VRAM right now" as the
        # decision input makes the decision function contradict itself -- a
        # resident TE lowers free VRAM, so the next call decides it "should be
        # demoted".
        self.te_mode = te_mode or ("resident" if self.cache else "cache")
        self.loads = 0                      # actual UNet load count (for self-check)
        self.step = None
        self.oom_retries = 0                # OOM downgrade count (self-check / acceptance)
        if self.device == "cuda":
            apply_cudnn_policy()

    # ---- info ------------------------------------------------------------
    def device_info(self):
        if self.device == "cuda" and _cuda_ok():
            p = torch.cuda.get_device_properties(0)
            return (f"{torch.cuda.get_device_name(0)} "
                    f"{p.total_memory / 1e9:.1f} GB - torch {torch.__version__}")
        return f"CPU - torch {torch.__version__}"

    def vram(self):
        """Allocated VRAM in GB; None on CPU. Note it cannot see CUDA context
        overhead."""
        if self.device != "cuda" or not _cuda_ok():
            return None
        return torch.cuda.memory_allocated() / 1e9

    # ---- text encoder: load once, then only move device, never re-read disk --
    def ensure_te(self):
        """Make sure the TE is loaded (**exactly once**), returns (tok, model)."""
        if self._te is None:
            dev = "cuda" if (self.te_mode == "resident" and self.device == "cuda") else "cpu"
            self._tok, self._te = load_te(dev)           # module-level function
            print(f"[te] mode {self.te_mode}, weights on {dev}"
                  f"{' (resident, no more disk reads)' if dev == 'cuda' else ' (moved to VRAM for encoding)'}",
                  flush=True)
        return self._tok, self._te

    def te_state(self):
        """Answer "where is the text encoder **now**" -- must **read the real device**.

        NOTE: never answer a state question from te_mode / a decision function:
              that describes "what was intended", not "what is true now". The
              counter-example in the rework doc is exactly this -- the panel
              said "demoted to RAM cache" while the model was happily on
              cuda:0; both were "true" yet contradictory.
        """
        if self._te is None:
            return "not loaded"
        d = next(self._te.parameters()).device
        return f"resident {d}" if d.type == "cuda" else f"weights on {d} (moved to VRAM per encode)"

    def _te_to(self, dev):
        """Move the TE to a device (only when it actually changes)."""
        if self._te is None:
            return
        if next(self._te.parameters()).device.type == dev:
            return
        self._te = self._te.to(dev)
        gc.collect()
        if _cuda_ok():
            torch.cuda.empty_cache()

    def demote_te(self):
        """Move a VRAM-resident text encoder back to RAM, freeing about 1.6 GB.
        Returns whether it **actually** demoted.

        This is the core of OOM resilience: the worst allowed outcome is
        "slower", never "crash".
        """
        if self._te is not None and next(self._te.parameters()).device.type == "cuda":
            self._te.to("cpu")
            self.te_mode = "cache"
            gc.collect()
            torch.cuda.empty_cache()
            print("[te] out of VRAM -> demoted to RAM cache (a bit slower, but it will not crash)",
                  flush=True)
            return True
        return False

    def _oom_retry(self, fn, what):
        """Run fn(); on OOM demote once and retry. **Sampling and VAE decoding
        each blow up independently, so both call sites need this.**"""
        try:
            return fn()
        except RuntimeError as e:
            msg = str(e).lower()
            if not (isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in msg):
                raise
            print(f"[oom] {what} ran out of VRAM: {str(e).splitlines()[0][:140]}", flush=True)
        self.oom_retries += 1
        if not self.demote_te():
            raise RuntimeError(
                f"{what} ran out of VRAM and there is nothing left to demote (the text "
                f"encoder was not on VRAM to begin with). Lower the resolution, or tick "
                f"'low VRAM mode'.")
        return fn()

    def peek_ckpt(self):
        """Lightweight read of model metadata.

        .pt           mmap only, no read (5.6 GB costs no RAM; measured 0.15 s)
        .safetensors  reads only the header, even faster
        Returns (step, source description string)
        """
        p = self.ckpt
        if p.suffix.lower() == ".safetensors":
            try:
                _add_code_path()
                import aq_lowbit as lb
                m = lb.read_meta(p)
                st = m.get("step")
                self.step = int(st) if st not in (None, "None", "") else None
                return self.step, f"packed-{m.get('mode')} (bitpacked)"
            except Exception:                              # noqa: BLE001
                return None, "unknown"
        try:
            blob = torch.load(p, map_location="cpu",
                              weights_only=False, mmap=True)
            st = blob.get("step")
            ema = bool(blob.get("ema"))
            del blob
            self.step = int(st) if st is not None else None
            return self.step, ("ema (moving average, more stable)" if ema
                               else "raw model (or unknown)")
        except Exception:                                  # noqa: BLE001
            return None, "unknown"

    # ---- model lifecycle --------------------------------------------------
    def load_unet(self):
        if self._unet is None:
            self._unet = load_unet(self.ckpt, self.device)
            self.step = (self._unet._aq_ckpt or {}).get("step")
            self.loads += 1
        return self._unet

    def load_vae(self):
        if self._vae is None:
            # Shares the same loader as the CLI path (fp16 default; resident
            # reuse, no unload)
            self._vae = load_vae(self.device, verbose=False)
            print("[vae] VAE loaded (resident, reused)", flush=True)
        return self._vae

    def release(self):
        """Drop all resident models and hand VRAM back (the text encoder is
        moved to RAM too)."""
        dropped = [n for n, v in (("UNet", self._unet), ("VAE", self._vae),
                                  ("TE", self._te)) if v is not None]
        self._unet = None
        self._vae = None
        if self._te is not None:            # keep the TE instance (avoids re-reading disk),
            self._te.to("cpu")              # only move the weights back to RAM
            self.te_mode = "cache"
        gc.collect()
        if _cuda_ok():
            torch.cuda.empty_cache()
        return "released: " + (", ".join(dropped) if dropped else "(nothing was resident)"
                              ) + " (TE weights stay in RAM, so no disk re-read next time)"

    # ---- generation -------------------------------------------------------
    def generate(self, prompts, steps=DDIM_STEPS, h=RES, w=None, seed=3407,
                 per_prompt=1, low_vram=False, on_status=None):
        """Run a batch of prompts, returns [(prompt, seed, png_path), ...].
        on_status receives progress strings.

        h / w are **pixel** sizes, square by default. w=None means w = h.
        Both must be multiples of 64; the 22 buckets used in training are in
        TRAINED_BUCKETS.
        """
        say = on_status or (lambda _m: None)
        prompts = [p.strip() for p in prompts if p and p.strip()]
        if not prompts:
            raise ValueError("empty prompt")
        h = int(h)
        w = int(h if w is None else w)
        ok, note = check_res(h, w)
        if not ok:
            raise ValueError(note.strip())
        steps = max(1, int(steps))
        per_prompt = max(1, int(per_prompt))

        if seed is None or int(seed) < 0:
            seed = random.SystemRandom().randint(0, 2 ** 31 - 1)
            say(f"no seed given, picked random {seed}")
        seed = int(seed)

        # low_vram=True forces non-resident for this call only (does not modify
        # self.cache, so it cannot pollute the default behaviour)
        cache = self.cache and not low_vram
        out_dir = Path(OUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)

        say(f"device {self.device_info()} - cache={'on' if cache else 'off (low VRAM)'}")
        say(f"checkpoint {self.ckpt.name} - step={self.step}")
        say(f"size {w}x{h} (width x height) -> latent {w // 8}x{h // 8}"
            + (" (a trained bucket)" if (h, w) in TRAINED_BUCKETS else ""))
        if note:
            say(note.strip())

        from PIL import Image
        results = []
        t0 = time.time()
        for pi, p in enumerate(prompts):
            tag = f"[{pi + 1}/{len(prompts)}]"
            if not cache:
                self.release()
            # Text encoder: loaded once, then moved at most once, **never
            # re-read from disk**
            tok, te = self.ensure_te()
            if self.te_mode == "cache" and self.device == "cuda":
                self._te_to("cuda")
            say(f"{tag} encoding text (TE {self.te_state()}) ...")
            ctx = encode_texts([p], self.device, tok, te)
            if self.te_mode == "cache" and self.device == "cuda":
                self._te_to("cpu")            # give it back right away, leave VRAM for sampling
            model = self.load_unet()
            vae = self.load_vae()
            for k in range(per_prompt):
                sd = seed + k
                say(f"{tag} sampling {k + 1}/{per_prompt} - {steps} steps - seed {sd} - {w}x{h}")
                _t_img = time.time()
                # Sampling and VAE decoding **each blow up independently**, so
                # both call sites are wrapped
                x = self._oom_retry(
                    lambda: ddim(model, ctx, self.abar, steps, sd,
                                 self.device, h, w)[0], "sampling")
                say(f"{tag} decoding image ...")
                imgs = self._oom_retry(
                    lambda: decode(x, self.device, vae=vae), "VAE decode")
                arr = (imgs[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
                path = out_dir / f"{pi:03d}_{slug(p)}_s{sd}_{w}x{h}.png"
                Image.fromarray(arr).save(path)
                say(f"{tag} saved {path.name} ({time.time() - _t_img:.1f} s)")
                results.append((p, sd, str(path)))
            if not cache:
                self.release()

        # Times are reported in SECONDS (the old "x.x min" was unreadable for the
        # single-image case, which is the normal case in the demo UI).
        _el = time.time() - t0
        say(f"done, {len(results)} image(s) in {_el:.1f} s"
            + (f" ({_el / len(results):.1f} s/image)" if results else "")
            + (f" ({self.oom_retries} OOM downgrade(s) along the way)" if self.oom_retries else ""))
        return results


def main():
    ap = argparse.ArgumentParser(description="Aquarius ternary text-to-image inference")
    ap.add_argument("prompts", nargs="*", help="prompt(s), one or more")
    ap.add_argument("--prompts-file", type=str, default=None,
                    help="read prompts from a text file, one per line")
    ap.add_argument("--ckpt", type=str, default=None,
                    help=f"checkpoint path (default {CKPT_DIR}/latest.pt)")
    ap.add_argument("--out", type=str, default=str(WS / "play_out"))
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--steps", type=int, default=DDIM_STEPS)
    ap.add_argument("--res", type=str, default=str(RES),
                    help="output size. '512' = square; '512x640' = width 512, height 640 "
                         "(width first). Both sides must be multiples of 64, cap 640")
    ap.add_argument("--batch", type=int, default=2,
                    help="samples per batch; lower it if VRAM is short (default 2)")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                    help="auto (default) uses the GPU when available. Force cpu for smoke "
                         "tests while training holds the card")
    ap.add_argument("--te-mode", choices=["resident", "cache"], default=None,
                    help="text encoder policy: resident = keep in VRAM (default, fast); "
                         "cache = weights in RAM, moved to VRAM for encoding (saves VRAM, slower)")
    args = ap.parse_args()

    prompts = list(args.prompts)
    if args.prompts_file:
        prompts += [l.strip() for l in
                    Path(args.prompts_file).read_text(encoding="utf-8").splitlines()
                    if l.strip()]
    if not prompts:
        ap.error("no prompt given. Usage: python aq_play.py \"a cat on a chair\"")
    try:
        h, w = parse_res(args.res)
    except ValueError:
        ap.error(f"could not parse --res: '{args.res}', expected e.g. 512 or 512x640 (width x height)")
    ok, note = check_res(h, w)
    if not ok:
        ap.error(f"--res {args.res} is invalid: {note.strip()}")
    if note:
        print(f"[warn] {note.strip()}")

    ckpt = Path(args.ckpt) if args.ckpt else default_ckpt()
    if not ckpt.is_file():
        print(f"[abort] model file not found: {ckpt}")
        print(f"        expected diffusion_model/*.safetensors or portable/*.safetensors,"
              f" or {CKPT_DIR}/latest.pt")
        return 2

    if args.device == "cpu":
        dev = "cpu"
    elif args.device == "cuda":
        dev = "cuda"
    else:
        dev = "cuda" if _cuda_ok() else "cpu"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if dev == "cuda":
        apply_cudnn_policy()
        print(f"[env] {torch.cuda.get_device_name(0)}  "
              f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB  "
              f"torch {torch.__version__}", flush=True)

    t0 = time.time()
    abar = cosine_schedule()

    for i in range(0, len(prompts), args.batch):
        chunk = prompts[i:i + args.batch]
        print(f"\n=== batch {i//args.batch + 1} / "
              f"{(len(prompts)+args.batch-1)//args.batch} ===", flush=True)
        for p in chunk:
            print(f"  - {p}", flush=True)

        ctx = encode_texts(chunk, dev)                   # TE loaded and dropped again

        model = load_unet(ckpt, dev)
        x, _ = ddim(model, ctx, abar, args.steps, args.seed, dev, h, w)
        del model
        gc.collect()
        torch.cuda.empty_cache()

        imgs = decode(x, dev)

        from PIL import Image
        for j, p in enumerate(chunk):
            arr = (imgs[j].permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
            fn = f"{i+j:03d}_{slug(p)}_s{args.seed}_{w}x{h}.png"
            Image.fromarray(arr).save(out_dir / fn)
            print(f"[out] {fn}", flush=True)

    dt = time.time() - t0
    print(f"\ndone, {len(prompts)} image(s) in {dt:.1f} s "
          f"({dt / max(1, len(prompts)):.1f} s/image) -> {out_dir}", flush=True)
    print("Reminder: this checkpoint is step 260,000 = 28.70 epochs (deliberately stopped:"
          " clip_ratio saturated + rental budget); output is still blurry scene"
          " composition, which is the expected under-trained state.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
