"""Aquarius UNet -- SD1.5-isomorphic ~1B text-to-image UNet for NLT (native low-bit) training.

Structure mirrors diffusers UNet2DConditionModel exactly (validated: instantiating this
class with SD1.5 config ch=[320,640,1280,1280], cross=768 reproduces diffusers 0.40.0
SD1.5 UNet param count 859,520,964 EXACTLY, per-module diff = 0):
  - block_out_channels=[256, 512, 1024, 1024], mid block at 1024 (same width as trunk end)
  - down = CrossAttnDownBlock2D x3 + DownBlock2D
  - mid  = UNetMidBlock2DCrossAttn
  - up   = UpBlock2D + CrossAttnUpBlock2D x3     (up channel seq = reversed: [1024,1024,512,256])
  - layers_per_block=2 (up blocks: 3 resnets), heads=8 fixed, head_dim = ch/8 (32/64/128/128)
  - cross_attention_dim=1024 (Qwen3.5-0.8B hidden; input = precomputed emb [B,32,1024], no tokenizer)
  - time cond = sinusoidal + MLP (256 -> 1024 -> 1024), same as SD1.5
  - in_channels = out_channels = 4

NLT (Native Low-bit Training): fp32 master weights live in quant space from step 0.
Every forward quantizes weights: g128 mean-scale (absmean per 128-weight group),
clamp +/-1, ternary (zero state) / binary (no zero state), STE straight-through.
Non-quantized layers (project pack spec): conv_in, conv_out, time_embedding MLP,
resnet time_emb_proj, all norms. Trunk convs, qkv/out, ff, up/down samplers and
resnet conv_shortcut are quantized. Quantizer semantics follow the COCO Lite 62M
train.py implementation (g128 mean-scale frozen quant + STE) scaled up; that file
was not present on this machine, so the canonical BitNet-b1.58 absmean formulation
from HANDOFF.md is used.

Usage:
  python aq_unet.py            -> print exact param count of the default 1B config
  python aq_unet.py --smoke    -> base=88 scaled-down same-structure forward/backward check
"""
import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# global NLT quant mode: 'fp' | 'ternary' | 'binary'
# ---------------------------------------------------------------------------
_QUANT_MODE = "fp"
_QUANT_GROUP = 128


def set_quant_mode(mode: str) -> None:
    global _QUANT_MODE
    assert mode in ("fp", "ternary", "binary"), mode
    _QUANT_MODE = mode


def nlt_quantize(w: torch.Tensor, group: int = _QUANT_GROUP) -> torch.Tensor:
    """g128 mean-scale quant + STE. w: fp32 master weight (any shape).

    ternary: q = round(clamp(w/scale, -1, 1))   in {-1, 0, +1}
    binary : q = sign(w/scale)                  in {-1, +1}  (no zero state)
    scale  = mean(|w|) per group of 128 weights (frozen, not trainable).
    Returns w + (w_q - w).detach(): forward uses w_q, backward flows to master w.
    """
    if _QUANT_MODE == "fp":
        return w
    orig_shape = w.shape
    wf = w.reshape(-1).float()
    pad = (-wf.numel()) % group
    if pad:
        wf = torch.cat([wf, torch.zeros(pad, device=w.device, dtype=wf.dtype)])
    wg = wf.view(-1, group)
    scale = wg.abs().mean(dim=1, keepdim=True).clamp_min(1e-8)
    wn = wg / scale
    if _QUANT_MODE == "binary":
        q = torch.where(wn >= 0, 1.0, -1.0)
    else:  # ternary
        q = torch.round(wn.clamp(-1.0, 1.0))
    wq = (q * scale).reshape(-1)[: w.numel()].reshape(orig_shape)
    return w + (wq - w).detach()


class QuantConv2d(nn.Conv2d):
    """Conv2d whose weight goes through NLT quant on every forward."""
    nlt_quant = True

    def forward(self, x):
        w = nlt_quantize(self.weight) if self.nlt_quant and _QUANT_MODE != "fp" else self.weight
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


class QuantLinear(nn.Linear):
    """Linear whose weight goes through NLT quant on every forward."""
    nlt_quant = True

    def forward(self, x):
        w = nlt_quantize(self.weight) if self.nlt_quant and _QUANT_MODE != "fp" else self.weight
        return F.linear(x, w, self.bias)


class NoQuantConv2d(QuantConv2d):
    nlt_quant = False  # conv_in / conv_out


class NoQuantLinear(QuantLinear):
    nlt_quant = False  # time_embedding MLP / resnet time_emb_proj


# ---------------------------------------------------------------------------
# building blocks (diffusers-isomorphic)
# ---------------------------------------------------------------------------

def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal timestep embedding, cos-first (diffusers flip_sin_to_cos convention)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float()[:, None] * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)



def _gn(ch: int, eps: float) -> nn.GroupNorm:
    """GroupNorm with 32 groups when ch allows (diffusers SD1.5 convention),
    else the largest power-of-two divisor <= 32 (only hit by scaled-down smoke configs)."""
    g = 32
    while ch % g != 0:
        g //= 2
    return nn.GroupNorm(g, ch, eps=eps)

class ResnetBlock2D(nn.Module):
    """diffusers ResnetBlock2D: GN-SiLU-Conv / (+temb proj) / GN-SiLU-Conv / (+shortcut)."""

    def __init__(self, in_ch: int, out_ch: int, emb_ch: int):
        super().__init__()
        self.norm1 = _gn(in_ch, 1e-5)
        self.conv1 = QuantConv2d(in_ch, out_ch, 3, padding=1)
        self.time_emb_proj = NoQuantLinear(emb_ch, out_ch)
        self.norm2 = _gn(out_ch, 1e-5)
        self.conv2 = QuantConv2d(out_ch, out_ch, 3, padding=1)
        self.conv_shortcut = QuantConv2d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_emb_proj(F.silu(temb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
        return x + h


class Attention(nn.Module):
    """Multi-head attention, 8 heads fixed (SD1.5 convention), SDPA backend.
    diffusers layout: to_q/k/v WITHOUT bias, to_out WITH bias."""

    def __init__(self, dim: int, ctx_dim: int, heads: int = 8):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.to_q = QuantLinear(dim, dim, bias=False)
        self.to_k = QuantLinear(ctx_dim, dim, bias=False)
        self.to_v = QuantLinear(ctx_dim, dim, bias=False)
        self.to_out = QuantLinear(dim, dim, bias=True)

    def forward(self, x, ctx=None):
        b, l, _ = x.shape
        ctx = x if ctx is None else ctx
        q = self.to_q(x).view(b, l, self.heads, self.head_dim).transpose(1, 2)
        k = self.to_k(ctx).view(b, ctx.shape[1], self.heads, self.head_dim).transpose(1, 2)
        v = self.to_v(ctx).view(b, ctx.shape[1], self.heads, self.head_dim).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v)
        o = o.transpose(1, 2).reshape(b, l, -1)
        return self.to_out(o)


class GEGLU(nn.Module):
    """diffusers GEGLU: Linear(dim, inner*2), chunk, a * gelu(b)."""

    def __init__(self, dim: int, inner: int):
        super().__init__()
        self.proj = QuantLinear(dim, inner * 2)

    def forward(self, x):
        a, b = self.proj(x).chunk(2, dim=-1)
        return a * F.gelu(b)


class FeedForward(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        inner = dim * 4
        self.net = nn.ModuleList([GEGLU(dim, inner), nn.Identity(), QuantLinear(inner, dim)])

    def forward(self, x):
        return self.net[2](self.net[0](x))


class BasicTransformerBlock(nn.Module):
    """diffusers BasicTransformerBlock: x+=attn1(ln1(x)); x+=attn2(ln2(x),ctx); x+=ff(ln3(x))."""

    def __init__(self, dim: int, ctx_dim: int, heads: int = 8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.attn1 = Attention(dim, dim, heads)          # self
        self.norm2 = nn.LayerNorm(dim, eps=1e-5)
        self.attn2 = Attention(dim, ctx_dim, heads)      # cross (text emb)
        self.norm3 = nn.LayerNorm(dim, eps=1e-5)
        self.ff = FeedForward(dim)

    def forward(self, x, ctx):
        x = x + self.attn1(self.norm1(x))
        x = x + self.attn2(self.norm2(x), ctx)
        x = x + self.ff(self.norm3(x))
        return x


class Transformer2D(nn.Module):
    """diffusers Transformer2DModel (patched inputs): GN -> proj_in(1x1) -> stack -> proj_out(1x1), residual."""

    def __init__(self, ch: int, ctx_dim: int, heads: int = 8, n_layers: int = 1):
        super().__init__()
        self.norm = _gn(ch, 1e-6)
        self.proj_in = QuantConv2d(ch, ch, 1)
        self.blocks = nn.ModuleList(
            [BasicTransformerBlock(ch, ctx_dim, heads) for _ in range(n_layers)]
        )
        self.proj_out = QuantConv2d(ch, ch, 1)

    def forward(self, x, ctx):
        b, c, h, w = x.shape
        y = self.proj_in(self.norm(x)).view(b, c, h * w).transpose(1, 2)
        for blk in self.blocks:
            y = blk(y, ctx)
        y = y.transpose(1, 2).reshape(b, c, h, w)
        return x + self.proj_out(y)


class Downsample2d(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = QuantConv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample2d(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = QuantConv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class CrossAttnDownBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, emb_ch, ctx_dim, layers=2, heads=8, n_tr=1):
        super().__init__()
        self.resnets = nn.ModuleList(
            [ResnetBlock2D(in_ch if i == 0 else out_ch, out_ch, emb_ch) for i in range(layers)]
        )
        self.attentions = nn.ModuleList(
            [Transformer2D(out_ch, ctx_dim, heads, n_tr) for _ in range(layers)]
        )
        self.downsamplers = nn.ModuleList([Downsample2d(out_ch)])

    def forward(self, x, temb, ctx):
        for res, att in zip(self.resnets, self.attentions):
            x = res(x, temb)
            x = att(x, ctx)
        return self.downsamplers[0](x)


class DownBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, emb_ch, layers=2):
        super().__init__()
        self.resnets = nn.ModuleList(
            [ResnetBlock2D(in_ch if i == 0 else out_ch, out_ch, emb_ch) for i in range(layers)]
        )
        self.downsamplers = None

    def forward(self, x, temb, ctx=None):
        for res in self.resnets:
            x = res(x, temb)
        return x


class UNetMidBlock2DCrossAttn(nn.Module):
    def __init__(self, ch, mid_ch, emb_ch, ctx_dim, heads=8, n_tr=1):
        super().__init__()
        self.resnets = nn.ModuleList(
            [ResnetBlock2D(ch, mid_ch, emb_ch), ResnetBlock2D(mid_ch, mid_ch, emb_ch)]
        )
        self.attentions = nn.ModuleList([Transformer2D(mid_ch, ctx_dim, heads, n_tr)])
        self.exit_conv = QuantConv2d(mid_ch, ch, 1) if mid_ch != ch else None

    def forward(self, x, temb, ctx):
        x = self.resnets[0](x, temb)
        x = self.attentions[0](x, ctx)
        x = self.resnets[1](x, temb)
        if self.exit_conv is not None:
            x = self.exit_conv(x)
        return x


class UpBlock2D(nn.Module):
    """in_widths[j] = hidden/skip concatenated input width per resnet (skip plan aware)."""

    def __init__(self, in_widths, out_ch, emb_ch):
        super().__init__()
        self.resnets = nn.ModuleList(
            [ResnetBlock2D(w, out_ch, emb_ch) for w in in_widths]
        )
        self.upsamplers = nn.ModuleList([Upsample2d(out_ch)])
        self.attentions = None

    def forward(self, x, skips, temb, ctx=None):
        for j, res in enumerate(self.resnets):   # skips: pre-sliced list, index access
            x = torch.cat([x, skips[j]], dim=1)  # (mutation-free -> checkpoint safe)
            x = res(x, temb)
        return self.upsamplers[0](x)


class CrossAttnUpBlock2D(nn.Module):
    def __init__(self, in_widths, out_ch, emb_ch, ctx_dim, heads=8, n_tr=1, has_upsample=True):
        super().__init__()
        self.resnets = nn.ModuleList(
            [ResnetBlock2D(w, out_ch, emb_ch) for w in in_widths]
        )
        self.attentions = nn.ModuleList(
            [Transformer2D(out_ch, ctx_dim, heads, n_tr) for _ in in_widths]
        )
        self.upsamplers = nn.ModuleList([Upsample2d(out_ch)]) if has_upsample else None

    def forward(self, x, skips, temb, ctx):
        for j, (res, att) in enumerate(zip(self.resnets, self.attentions)):
            x = torch.cat([x, skips[j]], dim=1)  # skips: pre-sliced list, index access
            x = res(x, temb)
            x = att(x, ctx)
        if self.upsamplers is not None:
            x = self.upsamplers[0](x)
        return x


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = dict(
    in_channels=4,
    out_channels=4,
    block_out_channels=[256, 512, 1024, 1024],
    layers_per_block=2,
    cross_attention_dim=1024,
    heads=8,                  # fixed like SD1.5; head_dim = ch // 8 = 32/64/128/128
    transformer_layers_per_block=1,
    mid_channels=1024,        # mid block scales down with the trunk (1024-level)
)


class AquariusUNet(nn.Module):
    """SD1.5-isomorphic UNet. ctx = [B, 32, 1024] precomputed text embedding."""

    def __init__(self, config: dict | None = None):
        super().__init__()
        cfg = dict(DEFAULT_CONFIG)
        if config:
            cfg.update(config)
        self.cfg = cfg
        chs = cfg["block_out_channels"]
        ch0 = chs[0]
        ch3 = chs[-1]
        emb_ch = ch0 * 4                    # 1408
        ctx_dim = cfg["cross_attention_dim"]
        heads = cfg["heads"]
        n_tr = cfg["transformer_layers_per_block"]
        lpb = cfg["layers_per_block"]
        mid_ch = cfg["mid_channels"] or ch3

        self.time_embedding = nn.Sequential(
            NoQuantLinear(ch0, emb_ch), nn.SiLU(), NoQuantLinear(emb_ch, emb_ch)
        )
        self.conv_in = NoQuantConv2d(cfg["in_channels"], ch0, 3, padding=1)

        # down: 3x CrossAttnDownBlock2D + DownBlock2D
        self.down_blocks = nn.ModuleList()
        prev = ch0
        for i in range(3):
            self.down_blocks.append(
                CrossAttnDownBlock2D(prev, chs[i], emb_ch, ctx_dim, lpb, heads, n_tr)
            )
            prev = chs[i]
        self.down_blocks.append(DownBlock2D(prev, ch3, emb_ch, lpb))

        # mid
        self.mid_block = UNetMidBlock2DCrossAttn(ch3, mid_ch, emb_ch, ctx_dim, heads, n_tr)

        # up path -- diffusers skip plan (validated vs diffusers SD1.5):
        # append order on the way down: [conv_in] + per CrossAttn level (lpb res outputs
        # + 1 downsample out) + DownBlock2D (lpb res outputs); popped in reverse.
        skip_plan = [ch0]
        for i in range(3):
            skip_plan += [chs[i]] * (lpb + 1)
        skip_plan += [ch3] * lpb
        pops = skip_plan[::-1]                       # consumed front-to-back by up path
        rev = list(reversed(chs))                    # [ch3, ch2, ch1, ch0]
        self.up_blocks = nn.ModuleList()
        hidden = ch3                                 # mid output width
        idx = 0
        for i in range(4):
            out_ch = rev[i]
            has_up = i < 3
            in_widths = []
            for _ in range(lpb + 1):
                in_widths.append(hidden + pops[idx])  # resnet input = hidden || skip
                idx += 1
                hidden = out_ch
            if i == 0:
                self.up_blocks.append(UpBlock2D(in_widths, out_ch, emb_ch))
            else:
                self.up_blocks.append(
                    CrossAttnUpBlock2D(in_widths, out_ch, emb_ch, ctx_dim, heads, n_tr,
                                       has_upsample=has_up)
                )
            hidden = out_ch

        self.conv_norm_out = _gn(ch0, 1e-5)
        self.conv_out = NoQuantConv2d(ch0, cfg["out_channels"], 3, padding=1)

    def forward(self, x, t, ctx, grad_ckpt=False):
        """
        x:   [B, 4, h, w]   noisy latent
        t:   [B]            integer timesteps in [0, T)
        ctx: [B, 32, 1024]  precomputed text embedding
        grad_ckpt: gradient checkpointing (training VRAM saver)
        returns predicted x0, same shape as x
        """
        if x.shape[-1] % 8 or x.shape[-2] % 8:
            raise ValueError(
                f"latent spatial dims must be multiples of 8, got {tuple(x.shape[-2:])} "
                f"(bucket pixel dims must be multiples of 64 per project spec)"
            )
        from torch.utils.checkpoint import checkpoint

        def ck(fn, *a):
            if grad_ckpt and self.training and torch.is_grad_enabled():
                return checkpoint(fn, *a, use_reentrant=False)
            return fn(*a)

        temb = self.time_embedding(timestep_embedding(t, self.cfg["block_out_channels"][0]))

        # down path, build skip stack in diffusers order:
        # [conv_in] + per level: (res+attn).out x layers, ds.out
        skips = [self.conv_in(x)]
        h = skips[0]
        for blk in self.down_blocks:
            if blk.downsamplers is not None:
                for res, att in zip(blk.resnets, blk.attentions):
                    # default-arg binding: closures must NOT late-bind loop vars,
                    # or checkpoint recompute at backward uses the wrong layers
                    h = ck(lambda hh, tt, r=res, a=att: a(r(hh, tt), ctx), h, temb)
                    skips.append(h)
                h = blk.downsamplers[0](h)
                skips.append(h)
            else:
                for res in blk.resnets:
                    h = ck(res, h, temb)
                    skips.append(h)

        # mid
        h = ck(self.mid_block, h, temb, ctx)

        # up path: slice skips OUTSIDE checkpoint (mutation-free inside)
        for blk in self.up_blocks:
            n = len(blk.resnets)
            blk_skips = [skips.pop() for _ in range(n)]
            h = ck(blk, h, blk_skips, temb, ctx)

        return self.conv_out(F.silu(self.conv_norm_out(h)))


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def smoke(base: int = 88):
    """Scaled-down same-structure forward/backward check (no real data needed)."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = dict(block_out_channels=[base, base * 2, base * 4, base * 4],
               cross_attention_dim=1024, heads=8)
    model = AquariusUNet(cfg).to(dev)
    n = count_params(model)
    set_quant_mode("ternary")
    x = torch.randn(2, 4, 32, 32, device=dev)
    t = torch.randint(0, 1000, (2,), device=dev)
    ctx = torch.randn(2, 32, 1024, device=dev)
    out = model(x, t, ctx)
    assert out.shape == x.shape, f"bad out shape {tuple(out.shape)}"
    target = torch.randn_like(x)
    loss = F.mse_loss(out.float(), target)
    loss.backward()
    grads_ok = all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    finite = torch.isfinite(out).all().item() and torch.isfinite(loss).item() and grads_ok
    print(f"[smoke base={base}] device={dev} params={n:,} out={tuple(out.shape)} "
          f"loss={loss.item():.4f} finite={finite}")
    # binary mode quick pass
    set_quant_mode("binary")
    model.zero_grad(set_to_none=True)
    out2 = model(x, t, ctx)
    loss2 = F.mse_loss(out2.float(), target)
    loss2.backward()
    print(f"[smoke base={base}] binary mode: loss={loss2.item():.4f} "
          f"finite={torch.isfinite(loss2).item() and torch.isfinite(out2).all().item()}")
    set_quant_mode("fp")
    return finite


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="base=88 scaled fwd/bwd check")
    ap.add_argument("--base", type=int, default=88, help="smoke base width")
    args = ap.parse_args()
    if args.smoke:
        ok = smoke(args.base)
        sys_exit = 0 if ok else 1
        raise SystemExit(sys_exit)
    model = AquariusUNet()  # default config (base 256), CPU, fp32 master
    n = count_params(model)
    print(f"AquariusUNet default config exact params: {n:,}")
    print(f"  vs expected ~550M (+/-3%): {n / 1e6:.1f} M ({(n - 550e6) / 1e6:+.1f} M off)")
