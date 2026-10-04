# -*- coding: utf-8 -*-
"""把 base-3 交付件转成**运行时 int4 布局**的自包含免转换交付件（~324 MB）。

## 为什么要这个（2026-10-04 用户要求）

用户设备有 CUDA，希望**冷启动零转换** —— 直接把自己预先打好的 int4 运行时布局拷进显存。
所以本脚本把 `torch._convert_weight_to_int4pack` 的结果**预先算好存进文件**，
加载时只搬不转（见 `aq_kernel.build_model_from_packed(..., prepacked=...)`）。

## ⚠️ 代价（必须知道，别当成纯赚）

- **体积从 152.99 MB 涨到约 324 MB（2.12×）** —— 因为 int4 槽是 4 bit，而权重信息量只有 1.585 bit。
- **int4pack 布局是 CUDA 专属且是 torch 内部实现细节**：
  `_weight_int4pack_mm` **在 CPU 上根本没实现**（实测 `NotImplementedError`），
  且它绑 torch 版本。⇒ **本文件只能在 CUDA 上用**，不能像 base-3 件那样跨平台。
- 省的只有**加载时那一次转码**（实测 CUDA 直建 3.3 s）。
⇒ 判据是「额外下载字节 vs 一次性转码秒数」。base-3 件是**正本**（设备无关），
   本文件是**为 CUDA 冷启动做的派生件**。两个都留着才对。

## 产物是**自包含**的

同时含：量化层的 `.packed`(uint8) + `.sz`(bf16) + `.bias`，以及**非量化层的 fp16 权重与 bias**。
⇒ 加载时只需要这一个文件，不需要同时带 base-3 件。

## 用法

    <aquarius py> code/aq_pack_int4.py --src out/aquarius_ternary_step249000.safetensors
    <aquarius py> code/aq_pack_int4.py --src ... --out ... --no-verify

⚠️ **需要 CUDA**（`_convert_weight_to_int4pack` 是 CUDA-only），会短暂占用 GPU。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

WS = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(WS / "code"), str(WS / "play")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="base-3 打包件（*.safetensors）")
    ap.add_argument("--out", default=None, help="输出路径；默认 out/<src名>_int4cuda.safetensors")
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--no-verify", action="store_true",
                    help="跳过「重新加载并与 base-3 路径比对输出」的端到端校验（不建议）")
    a = ap.parse_args()

    src = Path(a.src)
    if not src.is_absolute():
        src = WS / src
    if not src.is_file():
        raise SystemExit(f"找不到 {src}")
    out = Path(a.out) if a.out else (WS / "out" / (src.stem + "_int4cuda.safetensors"))
    if not out.is_absolute():
        out = WS / out

    import aq_kernel as AK

    print(f"[1/4] 从 base-3 件建 int4 模型（直建，含逐层自检）…")
    t0 = time.time()
    model, st = AK.build_model_from_packed(str(src), device="cuda",
                                          dtype=torch.bfloat16, selfcheck=True)
    if st.get("bad"):
        raise SystemExit(f"逐层自检失败 {len(st['bad'])} 层：{st['bad'][:2]}")
    print(f"      ✅ {st['linear']} Linear + {st['conv']} Conv · 跳过 {st['skipped']} · "
          f"{time.time() - t0:.1f}s")

    print(f"[2/4] 抽出预打好的 int4 张量…")
    tensors, kmap = {}, {}
    for name, mod in model.named_modules():
        if not isinstance(mod, (AK.FusedInt4Linear, AK.FusedInt4Conv2d)):
            continue
        tensors[name + ".packed"] = mod.packed.detach().to("cpu").contiguous()
        tensors[name + ".sz"] = mod.sz.detach().to("cpu").contiguous()
        kmap[name + ".weight"] = int(mod.K)          # 含补到 group 倍数的部分
        if getattr(mod, "use_bias", False):
            tensors[name + ".bias"] = mod.bias.detach().to("cpu").contiguous()
    n_quant = len(kmap)
    del model
    torch.cuda.empty_cache()

    src_raw = load_file(str(src))
    n_other = 0
    for k, v in src_raw.items():
        if k.endswith((".packed", ".sz", ".gscale")) or v.dtype == torch.uint8:
            continue                                  # 量化层的 base-3 码位/gscale 不要
        if k in kmap:
            continue
        tensors[k] = v.contiguous()                   # 非量化层原样（fp16 weight / bias）
        n_other += 1
    print(f"      ✅ 量化层 {n_quant} · 非量化张量 {n_other}")

    src_meta = {}
    try:
        with open(src, "rb") as f:
            import struct
            n = struct.unpack("<Q", f.read(8))[0]
            src_meta = json.loads(f.read(n).decode("utf-8")).get("__metadata__", {}) or {}
    except Exception:                                 # noqa: BLE001
        pass

    meta = {
        "format": AK.INT4_FORMAT,
        "packing": "int4cuda",
        "group": str(a.group),
        "K_map": json.dumps(kmap),
        "n_quant": str(n_quant),
        "n_other": str(src_meta.get("n_other", "")),
        "mode": src_meta.get("mode", "ternary"),
        "step": str(src_meta.get("step", "")),
        "source": src.name,
        "torch": torch.__version__,
        "cuda_arch": f"sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}",
        "note": "runtime int4 layout; CUDA only; zero-conversion load",
    }
    save_file(tensors, str(out), metadata=meta)
    mb = out.stat().st_size / 1e6
    print(f"[3/4] 已写出 {out.name}  {mb:.2f} MB  "
          f"（base-3 件 {src.stat().st_size / 1e6:.2f} MB，比值 {mb / (src.stat().st_size / 1e6):.2f}×）")

    if a.no_verify:
        print("[4/4] 已跳过校验")
        return 0
    print(f"[4/4] 端到端校验：用 `prepacked=` 重新加载，与 base-3 路径比对输出…")
    m_ref, _ = AK.build_model_from_packed(str(src), device="cuda", dtype=torch.bfloat16)
    m_pre, _ = AK.build_model_from_packed(str(out), device="cuda", dtype=torch.bfloat16,
                                         prepacked=str(out))
    torch.manual_seed(1234)
    x = torch.randn(1, 4, 32, 32, device="cuda", dtype=torch.bfloat16)
    t = torch.full((1,), 500, device="cuda", dtype=torch.long)
    c = torch.randn(1, 32, 1024, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        y_ref = m_ref(x, t, c).float()
        y_pre = m_pre(x, t, c).float()
    d = float((y_ref - y_pre).abs().max())
    scale = float(y_ref.abs().mean())
    print(f"      base-3 路径 vs 免转换路径：最大差 {d:.3e}（参考量级 {scale:.3f}）"
          f"  {'✅ 逐位一致' if d == 0 else ('✅ 差异可忽略' if d < 1e-3 else '❌ 差异过大')}")
    if d > 1e-3:
        return 1
    print(f"\n  ⚠️ 提醒：本文件**只能在 CUDA + 同族 torch 上零转换加载**；跨平台/长期正本仍用 base-3 件。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
