# -*- coding: utf-8 -*-
"""Loader shipped alongside the artifact. **transformers `from_pretrained`
cannot read int8 weights.**

This file is self-contained: it only needs `torch` + `transformers` +
`safetensors`, and does NOT depend on torchao / bitsandbytes.

    import sys; sys.path.insert(0, r"<artifact dir>")
    from load_te import load_te
    tok, model = load_te(r"<artifact dir>")               # default mode="int8": weights stay int8, saves RAM
    tok, model = load_te(r"<artifact dir>", mode="bf16")   # dequantize everything on load, saves CPU

    # Byte-for-byte the same contract as aq_play.encode_texts:
    b = tok(["a cheese pizza cut into many slices on a table"], return_tensors="pt",
            max_length=32, truncation=True, padding="max_length")
    ctx = model(**b).last_hidden_state[0]      # (32, 1024)

## The two modes

| | `mode="bf16"` | `mode="int8"` (default) |
|---|---|---|
| Weights in RAM | bf16 | **int8 + one f32 scale per row** |
| How it is built | normal build, fill **each tensor in place** | **empty shell on `torch.device("meta")`** + int8 module swap |
| Each matmul | plain bf16 | **temporarily** dequantize that layer, then multiply |
| Best for | plenty of RAM, lowest CPU cost | **tight RAM / machines with a spinning disk** |

**The embedding table is the single biggest win**: 248,320 x 1024, which is
31% of the whole file. In `int8` mode only **the ~32 rows that are actually
gathered** get dequantized, not the whole table -> almost zero cost for
roughly 260 MB saved.

NOTE: the two modes are **mathematically identical** (both are
"int8 -> dtype, then multiply"); measured ctx cos is about 1.0.

## Pitfalls already hit (do not repeat them)

1. **`from_config + load_state_dict` does no prefix mapping**: the checkpoint
   keys are `model.language_model.*`, while `Qwen3_5Model.state_dict()` uses
   `language_model.*`. `from_pretrained` strips the prefix automatically,
   `from_config` does **not** -> every key misses and the **model silently
   becomes randomly initialized** (cos drops from 0.995 to -0.012, with no
   error at all).
2. **The "no meta tensors left" check must run AFTER `model.visual = None`**:
   we intentionally do not ship the vision tower, so it stays on meta forever;
   checked too early it gets misreported as a failure.
3. **Do not build the model on the real device and then swap in int8**: peak
   memory first rises to the full bf16 copy, which defeats the whole point.
"""
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


# --------------------------------------------------------------------------- #
class _Int8Linear(nn.Module):
    """Weights stay int8; the forward pass dequantizes **this layer only** to
    the activation dtype and then multiplies."""

    def __init__(self, q, s, bias=None):
        super().__init__()
        self.register_buffer("q", q, persistent=False)     # int8 [out, in]
        self.register_buffer("s", s, persistent=False)     # fp32 [out]
        self.bias = bias
        self.out_features, self.in_features = q.shape

    def forward(self, x):
        # NOTE: **multiply in fp32**, exactly like `_dequant` -> bit-for-bit the
        #       same numbers as mode="bf16". Multiplying in bf16 (the scale only
        #       has 8 mantissa bits) produces a measurable drift (measured std
        #       4.1746 vs 4.1772).
        w = (self.q.to(torch.float32) * self.s.unsqueeze(-1)).to(x.dtype)
        return F.linear(x, w, self.bias)

    @property
    def weight(self):
        """Compatibility: model code may introspect `mod.weight`.
        Note that every access dequantizes a fresh copy, so it is only safe for
        introspection -- keep it out of hot paths."""
        return (self.q.to(torch.float32) * self.s.unsqueeze(-1)).to(self.s.dtype)


class _Int8Embedding(nn.Module):
    """Like `_Int8Linear`, but **only the gathered rows** are dequantized --
    the single biggest win of this scheme."""

    def __init__(self, q, s, dtype):
        super().__init__()
        self.register_buffer("q", q, persistent=False)
        self.register_buffer("s", s, persistent=False)
        self.num_embeddings, self.embedding_dim = q.shape
        self.out_dtype = dtype

    def forward(self, idx):
        return (self.q[idx].to(torch.float32) * self.s[idx].unsqueeze(-1)).to(self.out_dtype)

    @property
    def weight(self):
        """Compatibility: see the note on `_Int8Linear.weight`."""
        return (self.q.to(torch.float32) * self.s.unsqueeze(-1)).to(self.s.dtype)


# --------------------------------------------------------------------------- #
def _dequant(w, s, dtype, chunk_rows=8192):
    """int8 + per-row scale -> dtype. Done **in chunks** so a big tensor does
    not create a whole extra fp32 temporary."""
    if w.dtype != torch.int8:
        return w.to(dtype)
    sr = s.reshape(-1, *([1] * (w.dim() - 1)))
    out = torch.empty(w.shape, dtype=dtype)
    for i in range(0, w.shape[0], chunk_rows):
        out[i:i + chunk_rows] = (w[i:i + chunk_rows].to(torch.float32)
                                 * sr[i:i + chunk_rows]).to(dtype)
    return out


def load_te(directory, device="cpu", dtype=torch.bfloat16, mode="int8",
            verbose=True):
    """Load the slimmed (+quantized) text encoder, returns (tokenizer, model).

    mode="int8" -- weights stay int8 (saves RAM, default)
    mode="bf16" -- dequantize everything on load (saves CPU, costs RAM)
    """
    t_start = time.time()
    d = Path(directory)
    qj = json.loads((d / "quant.json").read_text(encoding="utf-8"))
    suffix = qj.get("scale_suffix", ".__scale__")
    if qj.get("bits", 0) == 0:
        mode = "bf16"
    wf = next((p for p in (d / "model_int8.safetensors", d / "model.safetensors")
               if p.is_file()), None)
    if wf is None:
        raise FileNotFoundError(f"{d} contains no model_int8.safetensors / model.safetensors")

    # ---- read weights (int8 resident, small; this is the only full copy) ----
    from safetensors import safe_open
    sd, scales = {}, {}
    with safe_open(str(wf), framework="pt") as h:
        for k in h.keys():
            if k.endswith(suffix):
                scales[k[: -len(suffix)]] = h.get_tensor(k)
            else:
                sd[k] = h.get_tensor(k)
    t_read = time.time() - t_start

    from transformers import AutoConfig, AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(d), trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(str(d), trust_remote_code=True)

    # ---- prefix alignment (pitfall 1) ----
    with torch.device("meta"):
        probe = AutoModel.from_config(cfg, dtype=dtype)
    expected = set(probe.state_dict().keys())
    del probe
    dropped = ""
    if not (set(sd) & expected):
        for pre in ("model.", "base_model.model.", "model.model."):
            cand = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
            if cand and (set(cand) & expected):
                sd = cand
                scales = {k[len(pre):]: v for k, v in scales.items()
                          if k.startswith(pre)}
                dropped = pre
                break
    quant = set(scales)
    if verbose:
        print(f"[load_te] {d.name}: {len(sd)} tensors, {len(quant)} int8, "
              f"bits={qj.get('bits')}, vision_tower={qj.get('vision_tower')}, "
              f"mtp={qj.get('mtp')}"
              + (f", prefix aligned (stripped '{dropped}')" if dropped else ""))

    # ---- build the model: **both paths build on meta** ----
    # NOTE: why meta is mandatory: `AutoModel.from_config()` on CPU performs a
    #       random init of 853M params, which **measured 99.6 seconds** (meta
    #       takes 1.6 s) -- that alone is a 60x difference.
    with torch.device("meta"):
        model = AutoModel.from_config(cfg, dtype=dtype)

    if mode == "bf16":
        full = {k: (_dequant(v, scales[k], dtype) if k in quant else v.to(dtype))
                for k, v in sd.items()}
        sd.clear()
        missing_names, _unexpected = model.load_state_dict(full, strict=False, assign=True)
        del full
    else:
        rest = {k: v.to(dtype) for k, v in sd.items() if k not in quant}
        missing_names, _unexpected = model.load_state_dict(rest, strict=False, assign=True)
        del rest
        mods = dict(model.named_modules())
        swapped = conv_bf16 = 0
        for name in list(quant):
            if not name.endswith(".weight"):
                continue
            path = name[: -len(".weight")]
            m = mods.get(path)
            parent_path, _, leaf = path.rpartition(".")
            parent = mods.get(parent_path) if parent_path else model
            if parent is None or leaf not in dict(parent.named_children()):
                continue
            bq = sd.get(path + ".bias")
            if isinstance(m, nn.Embedding):
                setattr(parent, leaf, _Int8Embedding(sd[name], scales[name], dtype))
                swapped += 1
            elif isinstance(m, nn.Linear):
                bias = (None if bq is None
                        else nn.Parameter(bq.to(dtype), requires_grad=False))
                setattr(parent, leaf, _Int8Linear(sd[name], scales[name], bias))
                swapped += 1
            else:
                # NOTE: for things like 3D convs, **replace only the module's
                #       `.weight`, never the module itself** -- the model code
                #       accesses `self.conv1d.weight`, and swapping the module
                #       for a bare Parameter crashes with
                #       "'Parameter' object has no attribute 'weight'".
                #       There are very few of these (18 layers x 24,576) and
                #       they are tiny -> just dequantize them to bf16.
                setattr(m, "weight",
                        nn.Parameter(_dequant(sd[name], scales[name], dtype),
                                     requires_grad=False))
                conv_bf16 += 1
        if verbose:
            print(f"[load_te] int8 resident: swapped {swapped} Linear/Embedding"
                  f" ({conv_bf16} 3D convs dequantized to bf16)")

    # NOTE: **drop the vision tower before checking for meta tensors** (pitfall
    #       hit before): we intentionally do not ship the vision tower, so it
    #       stays on meta forever and would otherwise be misreported as failure.
    if qj.get("vision_tower") == "dropped" and hasattr(model, "visual"):
        model.visual = None

    # ---- fill in the meta "computed at init, not in the checkpoint" buffers ----
    # Typical case: RoPE's `inv_freq` / `original_inv_freq` (non-persistent
    # buffers -> invisible in state_dict, which is why the earlier check missed
    # them and only `.to()` blew up). `__init__` derives them deterministically
    # from config, so rebuild the module once on the real device with
    # `type(mod)(mod.config)` and copy the buffers back (verified to work).
    fixed = 0
    for mname, mod in list(model.named_modules()):
        bs = [(n, b) for n, b in mod.named_buffers(recurse=False)]
        if not any(b.is_meta for _, b in bs):
            continue
        mcfg = getattr(mod, "config", None)
        if mcfg is None:
            bad = [n for n, b in bs if b.is_meta]
            raise RuntimeError(f"[load_te] {mname} has meta buffer(s) {bad} but no config, "
                               f"cannot rebuild")
        fresh = dict(type(mod)(mcfg).named_buffers(recurse=False))
        for bn, b in bs:
            if b.is_meta:
                if bn not in fresh:
                    raise RuntimeError(f"[load_te] rebuild of {mname} still has no buffer '{bn}'")
                mod._buffers[bn] = fresh[bn].clone()
        fixed += 1
    if verbose and fixed:
        print(f"[load_te] rebuilt {fixed} module(s) with computed-at-init buffers (like RoPE inv_freq)")

    # Final check: nothing on the text path should still be on meta
    left = [k for k, v in list(model.named_parameters()) + list(model.named_buffers())
            if v.is_meta]
    if left:
        raise RuntimeError(f"[load_te] {len(left)} meta tensors still unfilled (first 3: {left[:3]})")

    # Two kinds of "missing" are **normal** and must not be treated as failure:
    #   1) `rotary_emb.inv_freq` / `original_inv_freq` -- computed at init,
    #      **never in the checkpoint** (the original weight file lacks them too).
    #   2) in `int8` mode, the `.weight` tensors we replaced with
    #      `_Int8Linear/_Int8Embedding` -- they bypassed `load_state_dict` but
    #      are already owned by the modules.
    IGNORE = ("inv_freq",)
    real_missing = [k for k in missing_names
                    if k not in quant and not any(t in k for t in IGNORE)]
    fatal = [k for k in real_missing if "language_model" in k or "embed_tokens" in k]
    if fatal:
        raise RuntimeError(f"[load_te] text tower is missing {len(fatal)} weights (first 5: {fatal[:5]})"
                           f" -- the prefix or the quantized file is broken; refusing to silently "
                           f"return a randomly initialized model")
    if verbose:
        if real_missing:
            print(f"[load_te] {len(real_missing)} missing (should all belong to the vision tower, "
                  f"which is off the text path): {real_missing[:3]} ...")
        if missing_names:
            print(f"[load_te] another {len(missing_names) - len(real_missing)} are normally missing"
                  f" (computed inv_freq / already owned by int8 modules)")
    model = model.to(device).eval()
    if verbose:
        print(f"[load_te] read weights {t_read:.1f}s, total {time.time() - t_start:.1f}s")
    return tok, model


if __name__ == "__main__":
    import os
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent)
    md = sys.argv[2] if len(sys.argv) > 2 else "int8"
    try:
        import psutil
        # decimal MB (1 MB = 1e6 bytes), per the project-wide unit convention
        mb = lambda: psutil.Process(os.getpid()).memory_info().rss / 1e6
    except Exception:                                      # noqa: BLE001
        mb = lambda: float("nan")
    print(f"  start RSS {mb():.0f} MB")
    tok, model = load_te(d, mode=md)
    print(f"  [{md}] load complete, RSS {mb():.0f} MB")
    cap = "a cheese pizza cut into many slices on a table"
    b = tok([cap], return_tensors="pt", max_length=32, truncation=True,
            padding="max_length").to(next(model.parameters()).device)
    t0 = time.time()
    with torch.no_grad():
        ctx = model(**b).last_hidden_state[0]
    print(f"  [{md}] ctx {tuple(ctx.shape)} {ctx.dtype} std {float(ctx.float().std()):.4f}"
          f", encoded in {time.time() - t0:.2f}s")
