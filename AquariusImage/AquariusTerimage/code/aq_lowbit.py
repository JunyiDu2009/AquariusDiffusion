# -*- coding: utf-8 -*-
"""Aquarius low-bit weight format (ternary 2-bit / binary 1-bit) with
packing/unpacking primitives.

The bit layout is **identical** to the old project's
`LowPrecisionQuantization/pack_weights.py` -- that script is the implementation
behind the **188.4 MB** figure in the production report (179.6 MiB in the old
notation). This module just carries over its four core functions
(`quant_index` / `pack_bits` / `unpack_bits` / `dequant`) and swaps the
project-specific parts for Aquarius.

----------------------------------------------------------------------
Why packing is "lossless"
----------------------------------------------------------------------
Training and inference go through `aq_unet.nlt_quantize(w)`:

    s   = mean(|w|) per group of 128 weights (clamp_min 1e-8)
    q   = round(clamp(w / s, -1, 1))        # ternary, q in {-1, 0, +1}
    w_q = q * s

This module stores **(q, s) exactly**, so the unpacked `w_q` is bit-equivalent
to the training forward pass. The only deviation comes from the STE wrapper in
`nlt_quantize`, `w + (w_q - w).detach()` -- in fp32 `w + (w_q - w)` differs from
`w_q` by at most 1 ULP (about 1e-7 relative); measured in `aq_pack.py --verify`.

----------------------------------------------------------------------
NOTE: after loading you **must** call set_quant_mode("fp")
----------------------------------------------------------------------
Never feed the unpacked `w_q` to the model as "ordinary weights" and let it
quantize again:

    mean(|q * s|) = mean(|q|) * s  !=  s          (measured mean(|q|) ~ 0.71)

Re-quantizing shrinks every group of weights to 71%, making the result entirely
wrong **without reporting any error**.
There is only one correct approach: **the weights are already the final
quantized values -- let the forward pass use them as-is.**
In `aq_unet`, when `_QUANT_MODE == "fp"`, `nlt_quantize` returns `w` directly,
which is why `load_packed_into()` ends by setting the mode to `"fp"` -- this is
not "degrading to full precision", it is "turning off repeated quantization".

----------------------------------------------------------------------
File format (safetensors)
----------------------------------------------------------------------
quantized layer weight -> still called `{layer}.weight` after uint8 bitpacking
                          (the shape becomes a 1-D byte array)
        group scale    -> `{layer}.gscale`, fp32, length = ceil(numel / 128)
non-quantized layers   -> kept under the original name, fp16
metadata (all values are str):
    format  "aquarius-lowbit-1"
    mode    "ternary" | "binary" | "fp"
    group   "128"
    step    training step
    shapes  JSON {quantized layer name: [original shape]}   <- the shape is lost
            by packing, so it must be stored
    n_quant / n_other / bpw
Loading distinguishes layers by **dtype**: uint8 = packed layer, anything else =
loaded directly.
"""
import json
from pathlib import Path

import torch

FORMAT = "aquarius-lowbit-1"      # old format: ternary 2-bit + fp32 scale
FORMAT_V2 = "aquarius-lowbit-2"   # new format: ternary base-3 (5 per byte) + fp16 scale
FORMATS = (FORMAT, FORMAT_V2)
PACKINGS = ("b2", "b3")           # b2 = 2 bit per ternary; b3 = base-3, 1.6 bit per ternary (3^5=243<=256)
SCALE_DTYPES = ("fp32", "fp16")
G = 128                       # weights per group, same as training; must not change
                              # (changing it breaks compatibility with the packer)


# ---------------------------------------------------------------------------
# base-3 packing (ternary only) -- store "3 states" at the information limit
# ---------------------------------------------------------------------------
# Why this deserves its own path: ternary has only 3 states, so the
# **theoretical floor is log2(3) = 1.585 bit**. Storing 2 bit wastes 26% of the
# codes (135.1 MB -> 107.1 MB).
# Approach: 3^5 = 243 <= 256 => one byte holds 5 ternary digits losslessly.
def pack_ternary3(code):
    """code in {0,1,2} (i.e. q+1) -> pack every 5 into 1 byte."""
    n = code.numel()
    pad = (-n) % 5
    if pad:
        code = torch.nn.functional.pad(code, (0, pad))
    c = code.to(torch.int32).view(-1, 5)
    p = (c[:, 0] + 3 * c[:, 1] + 9 * c[:, 2] + 27 * c[:, 3] + 81 * c[:, 4])
    assert int(p.max()) <= 242, "base-3 packing overflow (code must be in {0,1,2})"
    return p.to(torch.uint8)


def unpack_ternary3(packed, n):
    """base-3 unpack -> code uint8{0,1,2}, truncated to length n."""
    p = packed.to(torch.int32)
    parts = [((p // (3 ** i)) % 3).to(torch.uint8) for i in range(5)]
    return torch.stack(parts, dim=1).reshape(-1)[:n]


# ---------------------------------------------------------------------------
# Quantization + bit packing (line-for-line identical to the old pack_weights.py)
# ---------------------------------------------------------------------------
def quant_index(w, mode):
    """Quantize exactly like the training forward pass -> (code uint8, scale fp32[ng,1])

    ternary: q in {-1,0,1} -> code {0,1,2}
    binary : q in {-1,1}   -> code {0,1}
    """
    flat = w.reshape(-1)
    n = flat.numel()
    pad = (G - n % G) % G
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    g = flat.view(-1, G).float()
    s = g.abs().mean(dim=1, keepdim=True).clamp_min(1e-8)      # [ng,1] fp32
    if mode == "ternary":
        q = (g / s).round().clamp_(-1, 1)                     # {-1,0,1}
        code = (q + 1).to(torch.uint8)                        # {0,1,2}
    elif mode == "binary":
        q = (g >= 0).float() * 2 - 1                          # {-1,1}
        code = (q > 0).to(torch.uint8)                        # {0,1}
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return code.reshape(-1), s


def pack_bits(code, bits):
    """uint8 code (0..2^bits-1) -> bitpacked uint8 (low bit first)"""
    n = code.numel()
    pad = (-n) % (8 // bits)
    if pad:
        code = torch.nn.functional.pad(code, (0, pad))
    if bits == 2:                                             # 4 x 2bit -> 1 byte
        c = code.view(-1, 4)
        packed = (c[:, 0] | (c[:, 1] << 2) | (c[:, 2] << 4) | (c[:, 3] << 6)).to(torch.uint8)
    else:                                                     # 8 x 1bit -> 1 byte
        c = code.view(-1, 8)
        p = torch.zeros(c.shape[0], dtype=torch.uint8)
        for i in range(8):
            p |= (c[:, i] << i).to(torch.uint8)
        packed = p
    return packed


def unpack_bits(packed, n, bits):
    """bitpacked uint8 -> code uint8 (truncated to length n)"""
    if bits == 2:
        c0 = packed & 0b00000011
        c1 = (packed >> 2) & 0b11
        c2 = (packed >> 4) & 0b11
        c3 = (packed >> 6) & 0b11
        code = torch.stack([c0, c1, c2, c3], dim=1).reshape(-1)
    else:
        bits_t = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8)
        code = ((packed.unsqueeze(1) & bits_t.unsqueeze(0)) > 0).to(torch.uint8).reshape(-1)
    return code[:n]


def dequant(code, scale, mode, orig_shape, n_orig, group=G):
    """code + scale -> dequantized weights (equivalent to the w_q computed by the
    training forward pass).

    Two things must be compatible or it will definitely fail (2026-10-02 was
    sunk by the first one):
      1. On disk the scale is **1-D** `[ng]` (squeezed when saved) while the
         packer internally passes `[ng,1]`. Both must work -- otherwise
         broadcasting blows up.
      2. `code` has length **n_orig** (tail padding already trimmed), which may
         not be a multiple of group, so a direct `.view(-1, group)` fails. Pad
         with zeros first, view, then trim back.
    """
    q = code.reshape(-1).float()
    q = q - 1 if mode == "ternary" else q * 2 - 1
    pad = (group - q.numel() % group) % group
    if pad:
        q = torch.nn.functional.pad(q, (0, pad))
    sc = scale.reshape(-1, 1) if scale.dim() == 1 else scale
    wq = (q.view(-1, group) * sc).reshape(-1)[:n_orig]
    return wq.reshape(orig_shape)


def bits_of(mode):
    return 2 if mode == "ternary" else (1 if mode == "binary" else 0)


# ---------------------------------------------------------------------------
# Packing: state_dict -> flat tensor dict + metadata
# ---------------------------------------------------------------------------
def pack_state_dict(sd, q_names, mode, group=G, packing="b3", scale_dtype="fp16"):
    """Returns (tensors, meta, rows). Each entry of `rows` is
    (name, param count, bytes, code round-trip diff, dequant diff).

    packing     "b2" = 2 bit per ternary (old format, byte-compatible);
                "b3" = base-3, 5 per byte (new, default)
                NOTE: binary has only 2 states, i.e. inherently 1 bit, and is
                unaffected by packing.
    scale_dtype "fp32" (old) | "fp16" (new, default) -- halves the scale size.
                NOTE: fp16 rounds the **per-group scale** (codes untouched),
                the same nature as "bf16 weights" but smaller in magnitude.
    """
    from aq_unet import nlt_quantize, set_quant_mode

    bits = bits_of(mode)
    b3 = (packing == "b3" and mode == "ternary")
    scal_t = torch.float16 if scale_dtype == "fp16" else torch.float32
    tensors, shapes = {}, {}
    n_q = n_other = 0
    rows = []
    # Set the quant mode once (nlt_quantize reads a module-level global; do not
    # reset it inside the loop)
    if mode != "fp":
        set_quant_mode(mode)
    try:
        for name, t in sd.items():
            t = t.detach().cpu()
            if mode != "fp" and name in q_names and t.dtype.is_floating_point:
                n = t.numel()
                code, scale = quant_index(t, mode)
                packed = pack_ternary3(code) if b3 else pack_bits(code, bits)
                tensors[name] = packed
                base = name[: -len(".weight")] if name.endswith(".weight") else name
                tensors[base + ".gscale"] = scale.squeeze(1).to(scal_t).clone().contiguous()
                shapes[name] = list(t.shape)
                n_q += n
                # Round-trip check: 1) are the codes bit-exact 2) dequantized
                # value vs the w_q from the training forward pass
                code_back = (unpack_ternary3(packed, n) if b3
                             else unpack_bits(packed, n, bits))
                exact = (code.to(code_back.dtype) - code_back).abs().max().item()
                back = dequant(code_back, scale, mode, t.shape, n)
                wq = nlt_quantize(t)
                rows.append((name, n, int(packed.numel()) + scale.numel() * scal_t.itemsize,
                             exact, (back - wq).abs().max().item()))
            else:
                tensors[name] = t.half().contiguous()
                n_other += t.numel()
                rows.append((name, t.numel(), t.numel() * 2, 0.0, 0.0))
    finally:
        set_quant_mode("fp")

    total_bytes = sum(r[2] for r in rows)
    nparams = n_q + n_other
    meta = {
        "format": FORMAT_V2 if (b3 or scale_dtype == "fp16") else FORMAT,
        "mode": mode,
        "group": str(group),
        "packing": "b3" if b3 else "b2",
        "scale_dtype": scale_dtype,
        "shapes": json.dumps(shapes, separators=(",", ":")),
        "n_quant": str(n_q),
        "n_other": str(n_other),
        "bpw": f"{total_bytes * 8 / max(nparams, 1):.4f}",
        "predicted_bytes": str(total_bytes),
    }
    return tensors, meta, rows


# ---------------------------------------------------------------------------
# Format migration: old (b2/fp32) -> new (b3/fp16). **Only the container
# changes; the quantization does not.**
# ---------------------------------------------------------------------------
def reformat_packed(src, dst, packing="b3", scale_dtype="fp16", verbose=True):
    """Repack an already-packed file into a different container.

    Never run `quant_index` again on "unpacked weights": re-quantizing
    `w_q = q*s` yields `s' = mean(|q|)*s ~ 0.71*s` (measured mean(|q|) ~ 0.71),
    shrinking every group of weights to 71%, **without reporting any error**
    (this is exactly what the warning at the top of the module is about).
    The correct approach is to **carry (code, scale) over unchanged** and only
    change the packing scheme and the scale dtype.

    Returns a dict: src_format/packing/scale_dtype -> dst versions, plus whether
    the codes are bit-identical.
    """
    from safetensors.torch import load_file, save_file

    m0 = read_meta(src)
    if m0.get("format") not in FORMATS:
        raise ValueError(f"source file is not in this format: {m0.get('format')!r}")
    mode = m0.get("mode", "ternary")
    bits = bits_of(mode)
    p0 = m0.get("packing", "b2")
    b3_out = (packing == "b3" and mode == "ternary")
    scal_t = torch.float16 if scale_dtype == "fp16" else torch.float32
    shapes = json.loads(m0.get("shapes", "{}"))

    raw = load_file(str(src))
    out, n_q, n_other = {}, 0, 0
    worst = 0.0
    for k, t in raw.items():
        if k.endswith(".gscale"):
            continue
        if t.dtype == torch.uint8:
            shape = tuple(shapes[k])
            n = 1
            for d in shape:
                n *= d
            base = k[: -len(".weight")] if k.endswith(".weight") else k
            code = unpack_ternary3(t, n) if p0 == "b3" else unpack_bits(t, n, bits)
            sc_old = raw[base + ".gscale"]
            # Code round-trip self-check: changing container must lose no information
            back = (unpack_ternary3(pack_ternary3(code), n) if b3_out
                    else unpack_bits(pack_bits(code, bits), n, bits))
            worst = max(worst, float((code.to(back.dtype) - back).abs().max()))
            out[k] = pack_ternary3(code) if b3_out else pack_bits(code, bits)
            out[base + ".gscale"] = sc_old.to(scal_t).clone().contiguous()
            n_q += n
        else:
            out[k] = t
            n_other += t.numel()

    total = sum(v.numel() * v.element_size() for v in out.values())
    nparams = n_q + n_other
    meta = dict(m0)
    meta.update({
        "format": FORMAT_V2 if (b3_out or scale_dtype == "fp16") else FORMAT,
        "packing": "b3" if b3_out else "b2",
        "scale_dtype": scale_dtype,
        "bpw": f"{total * 8 / max(nparams, 1):.4f}",
        "predicted_bytes": str(total),
    })
    save_file(out, str(dst), metadata=meta)
    if verbose:
        print(f"[reformat] {Path(src).name} -> {Path(dst).name}")
        print(f"           {p0}/{m0.get('scale_dtype','fp32')} -> {meta['packing']}/{scale_dtype}"
              f" - max code diff {worst:.0f} (must be 0)"
              f" - bpw {m0.get('bpw')} -> {meta['bpw']}")
    return {"code_diff": worst, "bpw_before": m0.get("bpw"), "bpw_after": meta["bpw"],
            "bytes": total, "n_quant": n_q, "n_other": n_other}


# ---------------------------------------------------------------------------
# Runtime loading (used by aq_play.py)
# ---------------------------------------------------------------------------
def read_meta(path):
    from safetensors import safe_open
    with safe_open(str(path), framework="pt", device="cpu") as f:
        return dict(f.metadata() or {})


def is_packed(path):
    """Is this file in our format? (Looks at the safetensors metadata, not the
    extension. Both old and new formats are recognised.)"""
    try:
        m = read_meta(path)
    except Exception:                                       # noqa: BLE001
        return False
    return m.get("format") in FORMATS


def load_packed_into(model, path, verbose=True):
    """Load a packed file into an already-built AquariusUNet.

    **It ends by setting the quant mode to "fp"** -- because the weights are
    already the final quantized values and the forward pass must use them as-is,
    otherwise they would be quantized a second time (see the warning at the top
    of the module).

    Returns a dict: mode / group / step / n_quant / n_other / bpw / bytes
    """
    from safetensors.torch import load_file
    from aq_unet import set_quant_mode

    meta = read_meta(path)
    if meta.get("format") not in FORMATS:
        raise ValueError(f"not an Aquarius packed file (format={meta.get('format')!r})")
    mode = meta.get("mode", "ternary")
    bits = bits_of(mode)
    # Dispatch between old and new format: old files lack these two keys =>
    # default to b2 / fp32
    packing = meta.get("packing", "b2")
    scale_dtype = meta.get("scale_dtype", "fp32")
    shapes = json.loads(meta.get("shapes", "{}"))

    raw = load_file(str(path))
    sd = {}
    n_q = 0
    for k, t in raw.items():
        if k.endswith(".gscale"):
            continue
        if t.dtype == torch.uint8:                    # packed layer
            if k not in shapes:
                raise ValueError(f"packed layer {k} has no shapes metadata")
            shape = tuple(shapes[k])
            n = 1
            for d in shape:
                n *= d
            base = k[: -len(".weight")] if k.endswith(".weight") else k
            code = (unpack_ternary3(t, n) if packing == "b3"
                    else unpack_bits(t, n, bits))
            sd[k] = dequant(code, raw[base + ".gscale"], mode, shape, n)
            n_q += n
        else:                                          # non-quantized layer (fp16)
            sd[k] = t.float()

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise ValueError(f"state_dict mismatch: missing={list(missing)[:4]} "
                         f"unexpected={list(unexpected)[:4]}")

    # Turn off repeated quantization: the weights are already in quantized space
    set_quant_mode("fp")

    info = {"mode": mode, "group": int(meta.get("group", G)),
            "step": (int(meta["step"]) if meta.get("step") not in (None, "", "None")
                     else None),
            "n_quant": n_q,
            "n_other": int(meta.get("n_other", 0)),
            "bpw": float(meta.get("bpw", 0) or 0),
            "path": str(path)}
    if verbose:
        import os
        mb = os.path.getsize(path) / 1e6
        print(f"[pack] {os.path.basename(str(path))}  mode={mode}  "
              f"step={info['step']}  quantized params {n_q:,}  non-quantized {info['n_other']:,}  "
              f"{mb:.1f} MB  {info['bpw']:.3f} bit/weight", flush=True)
        print("[pack] set_quant_mode('fp') -- weights are the final quantized values, "
              "the forward pass will not re-quantize", flush=True)
    return info
