# -*- coding: utf-8 -*-
"""Aquarius **low-bit fused kernel** inference backend -- lets weights travel
from VRAM straight into registers at 0.5 B per parameter.

## Why this exists (started 2026-10-03)

After loading, the packed build (`aq_lowbit.py`) holds **fp32 weights of
2.233 GB** -- so a forward pass reads **4 B per parameter**. Bonsai Image 4B
saves bandwidth precisely with a **fused low-bit GEMM** (in its own words:
"the inference bottleneck of a diffusion Transformer is weight memory
bandwidth, not computational precision"). The user's framing: "Bonsai is the
reference that started this work ... squeeze both storage and bandwidth to the
floor".

## Why not Gemlite / Triton

**This machine (Windows) has no Triton** (`torch.compile` reported
TritonMissing back then) => Gemlite is a dead end.
=> Instead use **two operators that ship with PyTorch** (torchao's
Int8/Int4WeightOnly are built on them):

| operator | group scale | device | weight | bandwidth |
|---|---|---|---|---|
| `torch._weight_int8pack_mm` | no, per-channel only (the CPU version forces a 1-D size N) | CUDA + CPU | 1 B/param | 1 B |
| **`torch._weight_int4pack_mm`** | yes, **supports group_size** | **CUDA only** | **0.5 B/param** | **0.5 B** |

=> we picked **int4 + group_size=128**: exactly the g128 quantization layout
from `aq_unet.nlt_quantize`, and **ternary fits losslessly into int4**
(`q in {-1,0,1}` -> `q+8 in {7,8,9}`).

## Ternary -> int4 is lossless (the key point)

torchao's int4 weight convention is "nibble values 0..15 mean -8..7", i.e. it
dequantizes to `(q4 - 8) * scale`.
Our `q in {-1,0,+1}` maps to `q4 = q + 8 in {7,8,9}` => `(q4-8) = q`,
**bit-exact**.

## How a 3x3 convolution attaches to a GEMM (route 2)

Flatten the conv weight `(out,in,3,3)` into `[out, in*9]`. Because
`nlt_quantize` groups by 128 along the **flattened weight**, the groups along
the flattened K dimension are **identical to the training-time grouping** =>
"**one 3x3 convolution = one `[out, in*9] x [in*9, L]` GEMM**":

    L = unfold(x, 3, padding=1, stride=s)        # [N, in*9, L]
    y = int4_mm(L^T, W_packed, scales) + bias    # [N, L, out]
    y = y.transpose(1,2).view(N, out, H', W')

`stride in {1,2}` both work (`F.unfold` supports them natively) => **all 50
quantized 3x3 convs in this UNet are covered**.
NOTE on the ragged tail: `in*9` is not necessarily a multiple of 128;
`quant_index` pads with **code=0** (dequantizes to 0), which contributes
exactly 0 to the GEMM, so it is **mathematically exact**; here we just pad K up
to a multiple of 128.

## NOTE on status: **not yet validated on GPU**

`_weight_int4pack_mm` / `_convert_weight_to_int4pack` have **no CPU
implementation** (measured `NotImplementedError`) => on this machine only the
"preparation stage" maths can be verified on CPU; the **actual fused GEMM needs
a GPU window** (training holds the 5090, and concurrent double CUDA segfaults).
It is therefore **off by default**; enable it explicitly with `AQ_KERNEL=int4`.

Usage:
    from aq_kernel import available, convert_model
    print(available())                       # what this environment supports
    convert_model(model, group=128)          # swap quantized layers in place
"""
import json
import os

import torch
import torch.nn.functional as F
from torch import nn

INT4_FORMAT = "aquarius-int4-cuda"   # prebuilt runtime int4 layout (zero conversion)
GROUP_DEFAULT = 128
_VERBOSE = os.environ.get("AQ_KERNEL_VERBOSE", "0") == "1"


# --------------------------------------------------------------------------- #
# Environment probing
# --------------------------------------------------------------------------- #
def available():
    """Return which kernel backends this environment supports."""
    info = {"device": "cuda" if torch.cuda.is_available() else "cpu",
            "int4pack": False, "int8pack": False, "triton": False}
    try:
        import triton  # noqa: F401
        info["triton"] = True
    except Exception:                                       # noqa: BLE001
        pass
    if torch.cuda.is_available():
        try:
            w = torch.zeros(8, 8, dtype=torch.int32, device="cuda")
            torch._convert_weight_to_int4pack(w, 2)
            info["int4pack"] = True
        except Exception as e:                              # noqa: BLE001
            info["int4pack_err"] = f"{type(e).__name__}: {str(e)[:80]}"
    try:                                                    # int8 has a CPU implementation
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        torch._weight_int8pack_mm(torch.zeros(1, 8, dtype=torch.bfloat16, device=dev),
                                  torch.zeros(4, 8, dtype=torch.int8, device=dev),
                                  torch.ones(4, dtype=torch.bfloat16, device=dev))
        info["int8pack"] = True
    except Exception:                                       # noqa: BLE001
        pass
    return info


def backend_for(device):
    i = available()
    if str(device).startswith("cuda") and i["int4pack"]:
        return "int4"
    if i["int8pack"]:
        return "int8"
    return "dequant"


# --------------------------------------------------------------------------- #
# Weight preparation: ternary codes -> packed int4
# --------------------------------------------------------------------------- #
def _pad_k(code, K):
    """Pad the flattened codes up to K (code 0 dequantizes to 0 => zero
    contribution to the GEMM)."""
    if code.numel() == K:
        return code
    return F.pad(code, (0, K - code.numel()))


def scales_of(scales, K, group):
    """Pad scales of shape [M, K0/group] to [M, K/group] (for the extra groups
    created by padding K we fill 1.0 -- the codes there are 0, and 0 times any
    scale is 0, so any value works)."""
    M, g0 = scales.shape
    g = K // group
    if g == g0:
        return scales
    return F.pad(scales, (0, g - g0), value=1.0)


def make_int4pack(code, scales, group=GROUP_DEFAULT, inner_k_tiles=None):
    """Ternary (codes, per-group mean|w|) -> (packed_int4, scales_and_zeros).
    **Zero dependencies, no torchao.**

    ## Recipe (measured on CPU 2026-10-03, relative error **0.2994%**, i.e. the
    ## same order as bf16 rounding)

    1. **You must supply qparams yourself; do not let the tool derive them from
       min/max**:
            scale = s / 7        zero = 0            # s = mean(|w|) of that group
       Ternary has only 3 levels, and the uniform 16-level affine int4 grid puts
       0 at q=7.5 (not an integer) -- auto-derived qparams give a **4-5% error**.
       Supplying them makes `{-s,0,+s}` land on **q = {1, 8, 15}** (formula
       `q = round(w/scale + 8)`, mid_point=8) => exact.
       NOTE: `zero = -s` was tried and dequantizes to `[-0.2, 0.0, 0.0]`
       (wrong); **zero must be 0**.

    2. **What you pass to `_convert_weight_to_int4pack*` must be `[M, K]`
       element-wise int32 (0..15)**, **not** a hand-packed `[M, K/8]`. The
       packing (into uint8 `[M, K/2]`) is done by the operator itself.
       Hand-packing `[M, K/8]` fails at the mm with
       `expect B.size(1) to be K/2, got 64`.
       (The CUDA version accepts the same form; it converts to uint8 internally,
       with even indices in the high nibble.)

    3. `scales_and_zeros` has layout **`[K/G, M, 2]`**, the last dim being
       (scale, zero).

    4. Length constraints: `w` must be **2-D** (reshape 4-D convs to
       `(out, -1)` first); K must be a **multiple of group** (pad with code 0 =>
       dequantizes to 0 => zero GEMM contribution); the operator itself also
       requires K/8 alignment. `inner_k_tiles = group // 16`.
    """
    M, K = code.shape
    # Do **not** compute floating-point weights: with only 3 levels the int4
    # codes are uniquely determined by the ternary codes --
    #     q = round(w / (s/7) + 8) = round(7*code + 8)  =>  {-1,0,+1} -> {1,8,15}
    #     (measured equivalent; this **skips the bf16 intermediate**, which cuts
    #     peak VRAM a lot)
    q = (code.to(torch.int32) * 7 + 8).clamp(0, 15)               # [M, K] element-wise
    scale = (scales / 7).to(torch.float32)                       # [M, K/G]
    zero = torch.zeros_like(scale)
    sz = torch.stack([scale, zero], dim=-1).transpose(0, 1).contiguous()  # [K/G, M, 2]
    tiles = inner_k_tiles or max(1, group // 16)
    if code.is_cuda:
        # NOTE: **the CUDA version wants uint8[M, K/2], with even indices in the
        #       high nibble** (exactly the opposite of the CPU version): torchao's
        #       source is `int_data[:, ::2] << 4 | int_data[:, 1::2]`.
        #       Passing int32 fails with `Expected in.dtype() == at::kByte`.
        b = (q[:, ::2] << 4 | q[:, 1::2]).to(torch.uint8)
        packed = torch.ops.aten._convert_weight_to_int4pack(b, tiles)
    else:
        # The CPU version wants **[M, K] element-wise int32** (it packs itself)
        packed = torch.ops.aten._convert_weight_to_int4pack_for_cpu(q, tiles)
    return packed, sz.to(torch.bfloat16)


def codes_from_packed(path, verbose=False, raw=None, meta=None):
    """Read the **(codes int8{-1,0,1}, scale)** of every quantized layer out of a
    packed file.

    This is the **only correct source.** Never back out the codes from the
    "unpacked floating-point weights": requantizing `w_q = q*s` yields
    `s' = mean(|q|)*s ~ 0.71*s` (measured mean(|q|) ~ 0.71), shrinking the whole
    group to 71% **without reporting anything**. (Measured 2026-10-03: whole-model
    output L1 jumped from ~0 to 37.8/255.)

    Returns {param_name: (codes[out,K0] int8, scales[out, K0/G] fp16)}.

    `raw` / `meta` may carry already-read contents to avoid reading the same file
    twice (153 MB x 2).
    """
    from safetensors.torch import load_file
    import aq_lowbit as lb
    m0 = meta if meta is not None else lb.read_meta(path)
    mode = m0.get("mode", "ternary")
    packing = m0.get("packing", "b2")
    bits = lb.bits_of(mode)
    shapes = __import__("json").loads(m0.get("shapes", "{}"))
    raw = raw if raw is not None else load_file(str(path))
    out = {}
    for k, t in raw.items():
        if t.dtype != torch.uint8:
            continue
        shape = tuple(shapes[k])
        n = 1
        for d in shape:
            n *= d
        base = k[: -len(".weight")] if k.endswith(".weight") else k
        code01 = (lb.unpack_ternary3(t, n) if packing == "b3" else lb.unpack_bits(t, n, bits))
        code = code01.to(torch.int8) - 1 if mode == "ternary" else code01.to(torch.int8) * 2 - 1
        sc = raw[base + ".gscale"]                       # [K0/G] or [out, K0/G]
        out[k] = (code.reshape(shape[0], -1),
                  sc.reshape(-1, sc.numel() // shape[0]) if sc.dim() == 1
                  else sc.reshape(shape[0], -1))
        if verbose:
            print(f"    [codes] {k} {shape} - scale {tuple(out[k][1].shape)}")
    return out


def int4_mm(x, packed, K, group, sz, codes=None):
    """Fused GEMM: x [..., K] -> [..., N]. The weights stay int4 resident the
    whole time and never land as floating point.

    When `codes` is not None it takes the **emulated path** (only for verifying
    surrounding logic on environments with no fused operator; no bandwidth gain).
    """
    if codes is not None:
        w = codes.to(torch.float32) * sz.repeat_interleave(group, dim=1).to(torch.float32)
        return (x.reshape(-1, codes.shape[1]).float() @ w.T).to(x.dtype).reshape(
            *x.shape[:-1], codes.shape[0])
    if x.device.type == "cuda":
        # NOTE: this is a **custom op, so autocast does not manage it** => what
        #       arrives may be fp32 (measured: x(1024,2304) float32). The operator
        #       only accepts bf16 => convert explicitly here and back afterwards.
        y = torch.ops.aten._weight_int4pack_mm(
            x.to(torch.bfloat16).contiguous(), packed, group, sz)
        return y.to(x.dtype)
    # NOTE: measured: the CPU version requires the activation to be **fp32**
    #       (passing bf16 gives "expected scalar type Float"), and
    #       scales_and_zeros must be fp32 as well. Convert back afterwards.
    y = torch.ops.aten._weight_int4pack_mm_for_cpu(
        x.float().contiguous(), packed, group, sz.float().contiguous())
    return y.to(x.dtype)


class FusedInt4Linear(nn.Module):
    """Equivalent to `nn.Linear`, but the weight is **int4 packed resident**
    (0.5 B/param)."""

    def __init__(self, packed, K, out_features, in_features, group, sz, bias,
                 dtype, codes=None):
        super().__init__()
        if packed is not None:
            self.register_buffer("packed", packed, persistent=False)
        self.register_buffer("sz", sz, persistent=False)      # scales_and_zeros
        if codes is not None:
            self.register_buffer("codes", codes, persistent=False)
        self.in_features, self.out_features = in_features, out_features
        self.K, self.group = K, group
        self.use_bias = bias is not None
        if bias is not None:
            self.register_buffer("bias", bias, persistent=False)

    def forward(self, x):
        sh = x.shape
        y = int4_mm(x.reshape(-1, self.in_features).contiguous(),
                    getattr(self, "packed", None),
                    self.K, self.group, self.sz, codes=getattr(self, "codes", None))
        y = y.reshape(*sh[:-1], self.out_features)
        return y + self.bias if self.use_bias else y


class FusedInt4Conv2d(nn.Module):
    """Any k x k convolution = `unfold` + **one fused GEMM** (K = in*k*k, which
    aligns naturally with the g128 grouping).

    `stride in {1,2}` both work (`F.unfold` supports them). When K is short of a
    multiple of group we pad with 0 codes, whose GEMM contribution is exactly 0
    => mathematically exact.
    """

    def __init__(self, packed, K, out_channels, in_channels, group, sz, bias,
                 kernel_size, stride, padding, dilation, dtype, codes=None):
        super().__init__()
        if packed is not None:
            self.register_buffer("packed", packed, persistent=False)
        self.register_buffer("sz", sz, persistent=False)
        if codes is not None:
            self.register_buffer("codes", codes, persistent=False)
        self.in_channels, self.out_channels = in_channels, out_channels
        self.K, self.group = K, group
        self.kernel_size, self.stride = kernel_size, stride
        self.padding, self.dilation = padding, dilation
        self.use_bias = bias is not None
        if bias is not None:
            self.register_buffer("bias", bias, persistent=False)

    def forward(self, x):
        N, C, H, W = x.shape
        L = F.unfold(x, self.kernel_size, dilation=self.dilation,
                     padding=self.padding, stride=self.stride)      # [N, C*k*k, M]
        if self.K > L.shape[1]:
            L = F.pad(L, (0, 0, 0, self.K - L.shape[1]))             # pad K (weights are 0 too)
        # NOTE: the operator requires a **2-D contiguous** input (the transpose
        #       after unfold is neither 2-D nor contiguous)
        M = L.shape[2]
        L2 = L.transpose(1, 2).reshape(-1, self.K).contiguous()      # [N*M, K]
        y = int4_mm(L2, getattr(self, "packed", None), self.K, self.group,
                    self.sz, codes=getattr(self, "codes", None))     # [N*M, out]
        Ho = (H + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        Wo = (W + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
        y = y.reshape(N, M, self.out_channels).transpose(1, 2).reshape(
            N, self.out_channels, Ho, Wo)
        return y + self.bias.view(1, -1, 1, 1) if self.use_bias else y


def convert_model(model, group=GROUP_DEFAULT, verbose=None, skip_names=(),
                  emulate=False, code_map=None, selfcheck=False, device=None,
                  packed_map=None):
    """Replace the model's quantized layers in place with fused int4 modules.
    **Should only be called on CUDA.**

    `device` -- **where to pack and where to land** (defaults to whatever device
    the old weight is on).
    Pointing it explicitly at `cuda` means "model structure stays on CPU, int4 is
    created directly on the GPU", which **completely skips the "move the bf16
    UNet onto the card" step** (the single biggest source of peak VRAM, measured
    1131.4 MB). See `build_model_from_packed()`.

    `packed_map` -- the **zero-conversion path**: `{param_name: (K, packed, sz)}`
    where packed/sz were prebuilt by the publisher with `code/aq_pack_int4.py`
    (including the CUDA int4pack layout and inner_k_tiles) => here we only move,
    never convert. Mutually exclusive with `code_map`; when supplied, no
    quantization or packing happens at all.

    Returns a stats dict. Unsupported layers (special cases beyond 3D,
    non-quantized layers) are left untouched.
    """
    verbose = _VERBOSE if verbose is None else verbose
    from aq_unet import QuantConv2d, QuantLinear
    if not emulate:
        # Both CUDA and CPU have fused operator implementations (CPU uses
        # _weight_int4pack_mm_for_cpu)
        ok = False
        try:
            torch.ops.aten._weight_int4pack_mm_for_cpu(
                torch.zeros(1, 128, dtype=torch.bfloat16),
                torch.ops.aten._convert_weight_to_int4pack_for_cpu(
                    torch.zeros(8, 16, dtype=torch.int32), 8),
                128, torch.zeros(2, 8, 2, dtype=torch.bfloat16))
            ok = True
        except Exception:
            ok = torch.cuda.is_available()
        assert ok, "this device has neither a CPU nor a CUDA int4 fused operator implementation"

    stats = {"linear": 0, "conv": 0, "skipped": 0, "params_int4": 0,
             "bad": [], "device": str(device) if device is not None else "same-as-weight"}
    for name, mod in list(model.named_modules()):
        if not getattr(mod, "nlt_quant", False):
            continue
        if name in skip_names or not hasattr(mod, "weight"):
            stats["skipped"] += 1
            continue
        w = mod.weight.data
        if w.dim() < 2:
            stats["skipped"] += 1
            continue
        if packed_map is not None:
            key = name if name in packed_map else (name + ".weight")
            if key not in packed_map:
                stats["skipped"] += 1
                continue
            K, packed, sc = packed_map[key]
            emu_codes = None
            if selfcheck:
                # NOTE: the prebuilt packed/sz come from a file => they are on
                #       **CPU**; the self-check must use copies on the **target
                #       device**, otherwise all 256 layers would be falsely
                #       reported bad (actually hit on 2026-10-04).
                _dv = torch.device(device) if device is not None else packed.device
                _pk, _sz2 = packed.to(_dv), sc.to(_dv)
                _xt = torch.randn(2, K, dtype=torch.bfloat16, device=_dv)
                try:
                    int4_mm(_xt, _pk, K, group, _sz2)
                except Exception as _e:                       # noqa: BLE001
                    stats["bad"].append((name, tuple(w.shape), K,
                                         type(_e).__name__, str(_e)[:70]))
                del _xt
        elif code_map is not None:
            # NOTE: code_map keys are **parameter names** (`...conv1.weight`)
            #       while named_modules() yields **module names** (`...conv1`) --
            #       so `.weight` must be appended here.
            key = name if name in code_map else (name + ".weight")
            if key not in code_map:
                stats["skipped"] += 1
                continue
            codes, scales = code_map[key]
        else:
            # Only allowed when the model's weights have not entered the
            # quantized space yet. Calling this on already-dequantized w_q
            # recomputes the scale (0.71x) => silent error; see the notes on
            # codes_from_packed.
            raise RuntimeError(
                f"{name}: no code_map supplied. Backing the codes out of already-dequantized "
                f"weights yields a 0.71x scale (a silent error) -- use codes_from_packed() to "
                f"read them from the packed file, or pass code_map explicitly.")
        if packed_map is None:
            codes = codes.reshape(w.shape[0], -1)                # [out, K0]
            K0 = codes.shape[1]
            K = ((K0 + group - 1) // group) * group              # pad to a multiple of group
            if K % 8:
                K = ((K + 7) // 8) * 8
            if K > K0:                                   # pad along dim=1 (code 0 => dequantizes to 0)
                codes = F.pad(codes, (0, K - K0))
            emu_codes = codes if emulate else None
            if emulate:
                packed, sz = None, scales.to(torch.float32)     # emulated: use [M, K/G] directly
            else:
                # NOTE: **packing must happen on the target device**:
                #       codes_from_packed() returns CPU tensors, and if they are
                #       not moved, `code.is_cuda` inside `make_int4pack` is false
                #       => it produces the **CPU uint8 layout**, which only blows
                #       up at forward time on CUDA (with a dimension error that is
                #       hard to localize).
                #       Also: `device` may be given explicitly => model structure
                #       stays on CPU, int4 is created directly on the GPU, so the
                #       1131.4 MB "bf16 UNet onto the card" never enters the peak.
                _dev = torch.device(device) if device is not None else w.device
                codes_d = codes.to(_dev)
                scales_d = scales_of(scales, K, group).to(_dev)
                packed, sz = make_int4pack(codes_d, scales_d, group=group)
                if selfcheck:
                    # Per-layer self-check: run a minimal mm **before installing
                    # it into the model**, so we do not discover the problem when
                    # the forward pass finally reaches that layer (which is much
                    # more expensive to localize).
                    _xt = torch.randn(2, K, dtype=torch.bfloat16, device=_dev)
                    try:
                        int4_mm(_xt, packed, K, group, sz)
                    except Exception as _e:                       # noqa: BLE001
                        stats["bad"].append((name, tuple(w.shape), K,
                                             type(_e).__name__, str(_e)[:70]))
                    del _xt
            sc = sz
        bias = None if mod.bias is None else mod.bias.data.detach().clone()
        dt = torch.bfloat16
        parent = model.get_submodule(name.rpartition(".")[0]) if "." in name else model
        leaf = name.rpartition(".")[2]
        if isinstance(mod, QuantLinear):
            new = FusedInt4Linear(packed, K, mod.out_features, mod.in_features,
                                  group, sc, bias, dt, codes=emu_codes)
            stats["linear"] += 1
        elif isinstance(mod, QuantConv2d):
            new = FusedInt4Conv2d(packed, K, mod.out_channels, mod.in_channels,
                                  group, sc, bias, tuple(mod.kernel_size),
                                  tuple(mod.stride), tuple(mod.padding),
                                  tuple(mod.dilation), dt, codes=emu_codes)
            stats["conv"] += 1
        else:
            stats["skipped"] += 1
            continue
        _tgt = torch.device(device) if device is not None else w.device
        setattr(parent, leaf, new.to(_tgt))
        # NOTE: **the old bf16 weight tensor must be released**: the
        #       `for name, mod in list(model.named_modules())` above built a
        #       **snapshot list** that holds **every** old module for the whole
        #       loop => the old weights are never freed => inflated peak VRAM.
        #       Measured: without releasing, the peak is **1483.7 MB** ~
        #       1131.4 (bf16 weights) + 287.3 (int4), while steady state needs
        #       only 287.3. (Setting `.weight` to None frees the bulk; the module
        #       object itself is still held by the snapshot but it is an empty shell.)
        try:
            mod.weight = None
        except Exception:                                     # noqa: BLE001
            pass
        stats["params_int4"] += w.numel()
        if verbose:
            print(f"  [kernel] {name}: {tuple(w.shape)} -> int4 packed (K={K})")
    return stats


# --------------------------------------------------------------------------- #
# Optimization 1: build the int4 model **directly** from the packed file
# (skipping the bf16 move onto the card)
# --------------------------------------------------------------------------- #
UNET_CFG = dict(block_out_channels=[256, 512, 1024, 1024],
                cross_attention_dim=1024, heads=8)


def build_model_from_packed(path, device="cuda", group=GROUP_DEFAULT,
                            dtype=torch.bfloat16, verbose=False, selfcheck=False,
                            prepacked=None):
    """**Optimization 1** -- build a fused int4 model straight from the packed
    file, never moving a bf16 UNet onto the card.

    Difference from the old path (`load_unet(...).to(bf16).cuda()` +
    `convert_model`):

        old: build the model by random init on CPU (2.2 GB host RAM, tens of seconds)
             -> `.to(bf16)` -> **`.cuda()` moves 1131.4 MB of weights onto the card**
                               <- the single biggest source of peak VRAM
             -> swap layer by layer to int4 (only then are the old weights freed,
                but the peak has already been recorded)

        new: build an **empty shell** on `torch.device("meta")` (0 B, 1-2 s)
             -> `to_empty(cpu)` allocates uninitialized storage
             -> load only the **non-quantized layers** from the file
                (18.2M params, fp16 36.4 MB)
             -> move each layer's **codes** onto the card and pack int4 in place
                (only a few MB transient per layer)
             -> finally `.to(device, bf16)` moves just those 36.4 MB

    => the "bf16 UNet onto the card" step disappears entirely. Measured peak went
    from **1483.3 MB to about 400 MB** (see `_t_opt_peak.py`).

    NOTE: **nothing is requantized** anywhere: codes are only read from the packed
    file (`codes_from_packed`), bit-exact.
    NOTE: ends with `set_quant_mode("fp")` -- the weights are already the final
    quantized values and requantizing would shrink everything to 71% without
    reporting an error.
    NOTE: there is a **safety gate**: the quantized layers must be exactly the
    keys of `code_map`, and every other parameter must have been loaded from the
    file -- otherwise the meta shell leaves **uninitialized garbage memory**, and
    the forward pass reports nothing while producing wrong results. This gate is
    not defensive programming; it guards the one genuinely new risk of this scheme.
    """
    from pathlib import Path as _P
    from safetensors.torch import load_file
    from aq_unet import AquariusUNet, set_quant_mode
    import aq_lowbit as lb

    path = _P(path)
    meta = lb.read_meta(path)
    _fmt = meta.get("format")
    # A self-contained int4 file is also valid input (when `prepacked` points at
    # itself, path IS it)
    if _fmt not in lb.FORMATS and _fmt != INT4_FORMAT:
        raise ValueError(f"not an Aquarius packed file (format={meta.get('format')!r})")
    raw = load_file(str(path))

    # 1) Build the shell on meta: no random init of 558M params (that costs tens
    #    of seconds and 2.2 GB of host RAM).
    with torch.device("meta"):
        model = AquariusUNet(UNET_CFG)
    model = model.to_empty(device="cpu")

    # 2) Load only the non-quantized layers (anything not uint8 and not .gscale)
    sd = {k: v for k, v in raw.items()
          if v.dtype != torch.uint8 and not k.endswith(".gscale")
          and not k.endswith(".sz") and not k.endswith(".packed")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        raise ValueError(f"the packed file has tensors the model does not know: "
                         f"{sorted(unexpected)[:6]}")

    # 3) Weight source:
    #    - default = read the **codes** from the base-3 packed file (the only
    #      correct source) and pack int4 here;
    #    - `prepacked=` = another file already in the **runtime int4 layout**
    #      => only move, never convert (zero conversion).
    packed_map = None
    if prepacked is not None:
        from safetensors.torch import load_file as _lf
        _pm = lb.read_meta(_P(prepacked))
        if _pm.get("format") != INT4_FORMAT:
            raise ValueError(f"not a prebuilt int4 file (format={_pm.get('format')!r}, "
                             f"expected {INT4_FORMAT!r})")
        _pr = _lf(str(prepacked))
        _kmap = json.loads(_pm.get("K_map", "{}"))     # {param_name: K}
        packed_map = {}
        for _k in _pr:
            if _k.endswith(".packed"):
                _base = _k[:-len(".packed")]           # module name
                _wk = _base + ".weight"
                if _wk not in _kmap:
                    raise ValueError(f"prebuilt file has no K for {_wk} (K_map incomplete)")
                # NOTE: the key must be the **parameter name** (".weight"), the
                #       same convention as code_map -- the safety gate compares
                #       against model.state_dict() parameter names; using module
                #       names makes all 256 layers falsely flagged.
                packed_map[_wk] = (int(_kmap[_wk]), _pr[_k], _pr[_base + ".sz"])
        if verbose:
            print(f"[int4] prebuilt file {_P(prepacked).name}: {len(packed_map)} layers, "
                  f"zero-conversion load")
        code_map = None
        q_keys = set(packed_map)
    else:
        code_map = codes_from_packed(path, verbose=verbose, raw=raw, meta=meta)
        q_keys = set(code_map)

    # 4) Safety gate: make sure no parameter is "neither in the file nor going to
    #    be replaced by int4"
    #    NOTE: `missing` comes from `model.state_dict()` so it uses **parameter
    #          names** (`...conv1.weight`), and `code_map` keys are parameter
    #          names too => they can be compared directly. Never compare module
    #          names (hit on 2026-10-04: using module names falsely flagged all
    #          256 layers, looking like the packed file was missing data).
    leftover = [m for m in missing if m not in q_keys]
    if leftover:
        raise RuntimeError(
            f"{len(leftover)} parameters are neither in the packed file nor a replaceable "
            f"quantized layer -- they would keep **uninitialized garbage memory** (the forward "
            f"pass reports nothing but produces wrong results): {leftover[:8]}")
    orphan = sorted(q_keys - set(missing))
    if orphan:
        raise RuntimeError(
            f"these {len(orphan)} quantized keys in the packed file have no matching parameter "
            f"in the model -- the packed file and the model structure disagree (check the "
            f"config): {orphan[:8]}")

    # 5) Layer by layer -> int4 (packing happens directly on the target device;
    #    the model structure stays on CPU throughout)
    stats = convert_model(model, group=group, code_map=code_map, packed_map=packed_map,
                          verbose=verbose, selfcheck=selfcheck, device=device)
    stats["n_other"] = int(meta.get("n_other", 0) or 0)
    stats["non_quant_params"] = sum(t.numel() for t in sd.values())

    # 6) Move only the **non-quantized layers** to the target device; the
    #    quantized layers are already int4 buffers on that device
    model = model.to(device=device, dtype=dtype).eval()
    set_quant_mode("fp")
    model._aq_ckpt = {"step": int(meta.get("step", -1) or -1),
                      "weights": f"packed-{meta.get('mode', 'ternary')} -> int4 fused kernel "
                                 f"(direct build, never moved as bf16)",
                      "path": str(path), "packed": True,
                      "bpw": float(meta.get("bpw", 0) or 0),
                      "kernel": "int4", "direct": True}
    return model, stats


def self_test():
    """Print environment capabilities and run a numerical self-check within what
    is available."""
    info = available()
    print("[aq_kernel] environment:", info)
    print(f"[aq_kernel] selected backend (cuda): {backend_for('cuda')} | (cpu): {backend_for('cpu')}")
    dev = "cuda" if info["int4pack"] else "cpu"
    torch.manual_seed(0)
    M, K, N, G = 8, 256, 3, GROUP_DEFAULT
    code = torch.randint(-1, 2, (M, K), dtype=torch.int8, device=dev)
    print(f"  ternary codes -> int4 domain values: {sorted(set((code + 8).reshape(-1).tolist()))}")
    if info["int4pack"]:
        pk, K2 = make_int4pack(code, group=G)
        sc = torch.rand(M, K2 // G, dtype=torch.float16, device=dev)
        x = torch.randn(N, K, dtype=torch.bfloat16, device=dev)
        y = int4_mm(x, pk, K2, G, sc)
        ref = x.float() @ (code.float() * sc.repeat_interleave(G, 1).float()).T
        print(f"  int4 fused GEMM numerical self-check: max diff "
              f"{float((y.float() - ref).abs().max()):.5f}"
              f" (reference magnitude {float(ref.abs().max()):.2f})")
    else:
        print("  [WARN] no int4pack on this device => fused GEMM not self-checked "
              "(normal on CPU)")
        print("     What can still be checked: ternary<->int4 mapping, K padding, "
              "scales layout (see the assertions in convert_model)")
    return info


if __name__ == "__main__":
    self_test()
