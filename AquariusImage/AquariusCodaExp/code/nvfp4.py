"""NVFP4 权重编解码：E2M1 数据 + per-16 E4M3 块 scale + FP32 per-tensor scale。

数值格式与计划书 §8.2 及 torchao BlockWise1x16 一致：
- E2M1 共 15 个有效值：0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6（4 bit，2 个值/字节）
- 每 16 个连续元素（展平序）共享一个 E4M3 块 scale
- 每个张量一个 FP32 scale，把块 scale 归一化进 E4M3 动态范围（≤448）
有效约 4.5 bit/值。

首层/输出头及全部 1-D 参数（norm/bias/embedding）按精度保留策略走 FP16，
仅 2-D 以上的 conv/linear 主体权重进 NVFP4（计划书 §8.2 精度保留策略）。
"""
import torch

GROUP = 16
MAX_E2M1 = 6.0
MAX_E4M3 = 448.0

# 幅度 → 4bit 码：0.0→0, 0.5→1, 1.0→2, 1.5→3, 2.0→4, 3.0→5, 4.0→6, 6.0→7；负数 = 码 + 8
_MAG_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _encode_e2m1(x: torch.Tensor) -> torch.Tensor:
    """最近邻舍入到 E2M1，返回 4bit 码（uint8, 0-15）。"""
    levels = _MAG_LEVELS.to(x.device, torch.float32)
    sign = (x < 0).to(torch.uint8)
    ax = x.abs().clamp(max=MAX_E2M1)
    idx = (ax.unsqueeze(-1) - levels).abs().argmin(dim=-1).to(torch.uint8)
    return idx + sign * 8


def _decode_e2m1(codes: torch.Tensor) -> torch.Tensor:
    """4bit 码 → 浮点值。"""
    levels = _MAG_LEVELS.to(codes.device, torch.float32)
    mag = levels[(codes & 7).long()]
    return torch.where(codes >= 8, -mag, mag)


def _encode_e4m3(x: torch.Tensor):
    """FP32 → E4M3。优先 torch 原生 float8 cast；失败则退回 FP32 存储（manifest 记录）。"""
    try:
        return x.to(torch.float8_e4m3fn), "e4m3"
    except Exception:
        return x, "fp32"


def _decode_e4m3(t: torch.Tensor) -> torch.Tensor:
    if t.dtype == torch.float8_e4m3fn:
        return t.float()
    return t.float()


def quantize_weight(w: torch.Tensor, group: int = GROUP) -> dict:
    """单个权重张量 → NVFP4 组件。"""
    orig_shape = tuple(w.shape)
    flat = w.detach().float().reshape(-1).cpu()
    n = flat.numel()
    pad_n = (-n) % group
    if pad_n:
        flat = torch.cat([flat, torch.zeros(pad_n)])
    blocks = flat.reshape(-1, group)
    amax = blocks.abs().amax(dim=1)
    block_scale = amax / MAX_E2M1
    tmax = block_scale.max()
    if tmax > 0:
        tensor_scale = (tmax / MAX_E4M3).clamp(min=1e-12)
    else:
        tensor_scale = torch.ones(())
    bs = (block_scale / tensor_scale).clamp(max=MAX_E4M3)
    bs_q, scales_dtype = _encode_e4m3(bs)
    bs_f = _decode_e4m3(bs_q)
    data = _encode_e2m1(blocks / (bs_f[:, None] * tensor_scale)).reshape(-1)
    if pad_n:
        data = data[:-pad_n]
    if data.numel() % 2:
        data = torch.cat([data, torch.zeros(1, dtype=torch.uint8)])
    packed = (data[0::2] | (data[1::2] << 4)).to(torch.uint8)
    return {
        "codes": packed,
        "block_scales": bs_q,
        "tensor_scale": tensor_scale,
        "shape": orig_shape,
        "n": n,
    }


def dequantize_weight(d: dict, group: int = GROUP) -> torch.Tensor:
    """NVFP4 组件 → FP16 权重。"""
    codes = d["codes"]
    lo = codes & 0xF
    hi = codes >> 4
    all_codes = torch.stack([lo, hi], dim=-1).reshape(-1)[: d["n"]]
    vals = _decode_e2m1(all_codes)
    bs = _decode_e4m3(d["block_scales"])
    ts = float(d["tensor_scale"])
    pad_n = (-d["n"]) % group
    if pad_n:
        vals = torch.cat([vals, torch.zeros(pad_n)])
    blocks = vals.reshape(-1, group) * bs[:, None] * ts
    out = blocks.reshape(-1)[: d["n"]]
    return out.reshape(d["shape"]).to(torch.float16)


def should_quantize(name: str, param: torch.Tensor) -> bool:
    """精度保留策略：仅 2-D 以上主体权重进 NVFP4；1-D（norm/bias/embedding）与
    首层/输出头保持 FP16（计划书 §8.2）。"""
    if param.ndim < 2:
        return False
    if "conv_in" in name or "conv_out" in name:
        return False
    return True


def export_model_nvfp4(state_dict: dict, group: int = GROUP) -> dict:
    """state_dict → NVFP4 导出包（含 FP16 保留层）。"""
    nvfp4_state, fp16_state = {}, {}
    total_params = quant_params = 0
    for name, p in state_dict.items():
        p = p.detach().cpu()
        total_params += p.numel()
        if should_quantize(name, p):
            nvfp4_state[name] = quantize_weight(p, group)
            quant_params += p.numel()
        else:
            fp16_state[name] = p.to(torch.float16)
    scales_dtype = "fp32"
    if nvfp4_state:
        scales_dtype = (
            "e4m3" if next(iter(nvfp4_state.values()))["block_scales"].dtype == torch.float8_e4m3fn else "fp32"
        )
    return {
        "format": "nvfp4-e2m1-e4m3g16",
        "group": group,
        "scales_dtype": scales_dtype,
        "fp16_state": fp16_state,
        "nvfp4_state": nvfp4_state,
        "stats": {
            "total_params": total_params,
            "nvfp4_params": quant_params,
            "fp16_params": total_params - quant_params,
        },
    }


def dequantize_model(pkg: dict) -> dict:
    """NVFP4 导出包 → FP16 state_dict。"""
    out = {k: v.clone() for k, v in pkg["fp16_state"].items()}
    for name, q in pkg["nvfp4_state"].items():
        out[name] = dequantize_weight(q, pkg.get("group", GROUP))
    return out
