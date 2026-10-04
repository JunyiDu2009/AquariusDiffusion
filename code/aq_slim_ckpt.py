"""把训练检查点压成「纯推理」检查点。

训练检查点（`latest.pt`，5.604 GB）里装了三份东西：

    model  fp32  2.233 GB   ← 训练用的高精度主权重（STE 更新累加需要）
    ema    fp32  2.233 GB   ← 滑动平均影子
    opt    bnb8bit 1.136 GB ← AdamW 的一阶/二阶矩

**推理只读 `ema`。** model 与 opt 只服务于「继续训练」，推理路径上一个字节都用不到
（`aq_unet` 的前向会把权重重新量化成 {-1,0,+1}×scale，看的就是 ema）。

所以：

    只留 ema（fp32）  → 2.233 GB
    只留 ema（fp16）  → 1.117 GB   ← 默认。见下面的精度论证

**为什么 ema 降到 fp16 几乎无损：**
  1. 量化层的权重会被 `nlt_quantize` 重新量化成三值。翻不翻转只取决于
     `w / scale` 与 ±0.5 的比较，而 fp16 的相对误差 ~5e-4 远小于三值的量化间距，
     会翻转的码位应当极少（`--verify` 给出精确计数，**不达标就别用这份精简检查点**）。
  2. 非量化层（conv_in/conv_out/time_embedding/norm）保留 fp16。注意推理时
     GPU 走 `torch.autocast(bf16)`，算子输入本来就被降到 **bf16（8 位尾数）**——
     fp16 有 10 位尾数，**比推理路径已有的精度还高**。

**⚠️ 一个必须避开的度量陷阱**（第一版 `--verify` 就栽在这里）：
不要比较 `nlt_quantize(w)` 的**反量化输出** `w_q = code × scale`。
fp16 存储会让 `scale` 自身微移 ~5e-4，于是每个**非零**码位的数值都不相等，
比较结果会退化成「非零码位占比」（实测量到 70.9%，纯属假象）。
必须直接比较**整数码位** `round(clamp(w/scale, -1, 1))`。

用法：

    python code/aq_slim_ckpt.py <in.pt> <out.pt>              # ema → fp16
    python code/aq_slim_ckpt.py <in.pt> <out.pt> --keep-fp32  # ema → fp32
    python code/aq_slim_ckpt.py <in.pt> <out.pt> --verify     # 附三值翻转统计

产物可直接喂给 `play/aq_play.py` / `play/app.py`（它们读 `blob["ema"]`）。
不要拿它 `--resume` 训练：没有 optimizer 状态，无法续跑。
"""
import argparse
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def human(n):
    return f"{n / 1e9:.2f} GB"


def convert(src, dst, keep_fp32=False):
    t0 = time.time()
    print(f"[read] {src}")
    blob = torch.load(src, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict) or not (blob.get("ema") or blob.get("model")):
        raise SystemExit(f"[abort] {src} 不像 Aquarius 检查点（顶层键：{list(blob)[:8]}）")

    use_ema = bool(blob.get("ema"))
    w = blob["ema"] if use_ema else blob["model"]
    dtype = torch.float32 if keep_fp32 else torch.float16

    print(f"[info] 权重来源 = {'ema' if use_ema else 'model'} · step = {blob.get('step')}")
    print(f"[info] 张量 {len(w)} 个，目标 dtype = {str(dtype).replace('torch.', '')}")

    slim = {}
    nparam = 0
    for k, v in w.items():
        v = v.detach()
        slim[k] = v.to(dtype) if v.dtype.is_floating_point else v.clone()
        nparam += v.numel()

    out = {"ema": slim, "step": blob.get("step"), "slim": True,
           "ema_dtype": str(dtype).replace("torch.", ""),
           "source": os.path.basename(str(src)),
           "params": nparam,
           "note": "inference-only: optimizer state and fp32 master weights stripped"}
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dst)

    a, b = Path(src).stat().st_size, dst.stat().st_size
    print(f"[done] {human(a)} → {human(b)}"
          f"（{a / b:.2f}× 缩小，省 {human(a - b)}）耗时 {time.time() - t0:.1f}s")
    print(f"[out ] {dst}")
    del blob, w, slim, out
    return dst


def _codes(w, group=128, mode="ternary"):
    """复刻 aq_unet.nlt_quantize 的**整数码位**。

    刻意不返回反量化值 `code * scale`：fp16 存储会让 scale 自身漂移，
    那样比较会退化成「非零码位占比」，量到的全是不相等的假象。
    """
    wf = w.reshape(-1).float()
    pad = (-wf.numel()) % group
    if pad:
        wf = torch.cat([wf, torch.zeros(pad, dtype=wf.dtype)])
    wg = wf.view(-1, group)
    scale = wg.abs().mean(dim=1, keepdim=True).clamp_min(1e-8)
    wn = wg / scale
    if mode == "binary":
        return torch.where(wn >= 0, 1.0, -1.0)
    return torch.round(wn.clamp(-1.0, 1.0))


def verify(src, dst):
    """精确统计：fp16 存储让多少个三值码位发生翻转。"""
    from aq_unet import AquariusUNet, set_quant_mode

    set_quant_mode("ternary")
    ref = AquariusUNet(dict(block_out_channels=[256, 512, 1024, 1024],
                            cross_attention_dim=1024, heads=8))
    q_names = {n + ".weight" for n, m in ref.named_modules()
               if getattr(m, "nlt_quant", False) is True}
    del ref

    full = torch.load(src, map_location="cpu", weights_only=False)
    slim = torch.load(dst, map_location="cpu", weights_only=False)
    fa = full.get("ema") or full.get("model")
    fb = slim["ema"]

    tot = flips = 0
    zeros = 0
    worst = []
    for name in sorted(q_names):
        if name not in fa:
            continue
        with torch.no_grad():
            ca = _codes(fa[name])
            cb = _codes(fb[name])
        n = ca.numel()
        f = int((ca != cb).sum().item())
        tot += n
        flips += f
        zeros += int((ca == 0).sum().item())
        if f:
            worst.append((f / n, f, n, name))

    print("\n=== 三值码位翻转统计（fp32 ema → fp16 ema）===")
    print(f"量化层权重总数      {tot:,}")
    print(f"其中零码位          {zeros:,}（{100.0 * zeros / max(tot, 1):.2f}%）")
    print(f"翻转的码位          {flips:,}")
    print(f"翻转比例            {100.0 * flips / max(tot, 1):.6f}%")
    if worst:
        worst.sort(reverse=True)
        print("翻转最多的 5 层：")
        for r, f, n, name in worst[:5]:
            print(f"  {100 * r:8.5f}%  {f:>8,}/{n:<10,} {name}")
    else:
        print("无任何翻转。")

    # 非量化层（fp16 保留）：绝对误差与 RMS 相对误差
    # 不要用「逐元素最大相对误差」——权重里接近 0 的元素会让它虚高几个数量级。
    maxabs = 0.0
    num = den = 0.0
    for k, v in fa.items():
        if k in q_names or not v.dtype.is_floating_point:
            continue
        d = fb[k].float() - v.float()
        maxabs = max(maxabs, d.abs().max().item())
        num += d.double().pow(2).sum().item()
        den += v.double().pow(2).sum().item()
    print(f"\n非量化层（{len(fa) - len(q_names)} 个张量，fp16 保留）：")
    print(f"  最大绝对误差      {maxabs:.3e}")
    print(f"  RMS 相对误差      {(num / max(den, 1e-30)) ** 0.5:.3e}")
    print(f"  对照：bf16 理论相对误差 2^-8 = {2 ** -8:.3e}，"
          f"fp16 = 2^-11 = {2 ** -11:.3e}")

    del full, slim, fa, fb
    return flips, tot


def main():
    ap = argparse.ArgumentParser(description="训练检查点 → 纯推理检查点")
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--keep-fp32", action="store_true", help="ema 保持 fp32")
    ap.add_argument("--verify", action="store_true", help="转换后附三值翻转统计")
    args = ap.parse_args()

    out = convert(args.src, args.dst, keep_fp32=args.keep_fp32)
    if args.verify:
        verify(args.src, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
