"""AquariusCodaExp UNet 一步去噪器（像素空间，直接预测 x0）。

结构按计划书 §4.1：
- 输入 3 通道带噪 RGB（[-1,1]），无 VAE
- 下采样 4 层 64→128→256→512，每层 2 个 ResBlock；瓶颈 512 通道 2 个 ResBlock
- 上采样对称 4 层 + skip connection；输出头 1×1 卷积直接输出 x0
- 条件：sinusoidal(p) → MLP → AdaGN 注入每个 ResBlock
- Self-Attention 只放在最深两级（256² 输入时为 32² 与 16²；512² 输入时为 64²/32²，见 §4.2）
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(ch: int) -> int:
    for g in (32, 16, 8, 4, 2, 1):
        if ch % g == 0:
            return g
    return 1


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, 2 * out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self._ckpt = False

    def _body(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb_proj(emb).chunk(2, dim=-1)
        h = self.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(F.silu(h))
        return h + self.skip(x)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        if self._ckpt and self.training and x.requires_grad:
            return torch.utils.checkpoint.checkpoint(self._body, x, emb, use_reentrant=False)
        return self._body(x, emb)


class AttnBlock(nn.Module):
    def __init__(self, ch: int, heads: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(_groups(ch), ch)
        self.qkv = nn.Conv2d(ch, 3 * ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        self.heads = heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, C, H * W)
        hd = C // self.heads
        q = qkv[:, 0].reshape(B, self.heads, hd, H * W).transpose(1, 2)
        k = qkv[:, 1].reshape(B, self.heads, hd, H * W).transpose(1, 2)
        v = qkv[:, 2].reshape(B, self.heads, hd, H * W).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, C, H, W)
        return x + self.proj(out)


class UNetDenoiser(nn.Module):
    def __init__(self, base: int = 64, emb_dim: int = 256):
        super().__init__()
        self.emb_dim = emb_dim
        self.conv_in = nn.Conv2d(3, base, 3, padding=1)

        self.p_embed = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        chs = [base, base * 2, base * 4, base * 8]
        # encoder：每层 2 个 ResBlock + stride2 卷积下采样（共 4 次下采样）
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        prev = base
        for c in chs:
            self.enc_blocks.append(nn.ModuleList([ResBlock(prev, c, emb_dim), ResBlock(c, c, emb_dim)]))
            self.downs.append(nn.Conv2d(c, c, 3, stride=2, padding=1))
            prev = c

        self.enc3_attn = AttnBlock(chs[2])
        self.mid1 = ResBlock(chs[-1], chs[-1], emb_dim)
        self.mid_attn = AttnBlock(chs[-1])
        self.mid2 = ResBlock(chs[-1], chs[-1], emb_dim)

        # decoder：每层 2 个 ResBlock（输入 = 深侧上采样 + 同层 skip，两者通道不同）
        # 第 i 级：输入 chs[4-i] + chs[3-i]，输出 chs[3-i]（i=0 时深侧为 mid 输出 chs[3]）
        self.dec_blocks = nn.ModuleList()
        self.ups = nn.ModuleList()
        for i in range(4):
            skip_c = chs[3 - i]
            deeper_c = chs[3] if i == 0 else chs[4 - i]
            self.dec_blocks.append(
                nn.ModuleList([ResBlock(deeper_c + skip_c, skip_c, emb_dim), ResBlock(skip_c, skip_c, emb_dim)])
            )
            self.ups.append(nn.Conv2d(skip_c, skip_c, 3, padding=1))

        self.norm_out = nn.GroupNorm(_groups(base), base)
        self.conv_out = nn.Conv2d(base, 3, 3, padding=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def time_embedding(self, p: torch.Tensor) -> torch.Tensor:
        half = self.emb_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=p.device, dtype=torch.float32) / half
        )
        args = p[:, None].float() * 1000.0 * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.p_embed(emb)

    def forward(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        emb = self.time_embedding(p)
        h = self.conv_in(x)
        skips = []
        for i, blocks in enumerate(self.enc_blocks):
            for blk in blocks:
                h = blk(h, emb)
            skips.append(h)
            h = self.downs[i](h)
            if i == 2:
                h = self.enc3_attn(h)
        h = self.mid1(h, emb)
        h = self.mid_attn(h)
        h = self.mid2(h, emb)
        for i, blocks in enumerate(self.dec_blocks):
            skip = skips[3 - i]
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            for blk in blocks:
                h = blk(h, emb)
            if i < 3:
                h = F.interpolate(h, scale_factor=2, mode="nearest")
                h = self.ups[i](h)
        h = self.conv_out(F.silu(self.norm_out(h)))
        return h

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def enable_gradient_checkpointing(self) -> None:
        """开启 ResBlock 梯度检查点（计划书 §7.1 显存优化，8GB 硬约束必需）。

        以约 +30% 计算时间换取激活显存 3-5 倍下降；推理（eval 模式）不受影响。
        """
        for m in self.modules():
            if isinstance(m, ResBlock):
                m._ckpt = True
