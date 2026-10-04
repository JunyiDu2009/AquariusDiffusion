# -*- coding: utf-8 -*-
"""Aquarius 权重打包：训练检查点 -> 可交付的低比特模型文件。

这是旧项目 `LowPrecisionQuantization/code/pack_weights.py` 的 Aquarius 适配版。
**位打包/反量化代码逐行沿用原脚本**（那份实现就是报告里 188.4 MB（旧写法 179.6 MiB）的来源），
只替换掉 5 处项目耦合：

    ROOT = r"E:/AI_Library/Projects/LowPrecisionQuantization"   -> 本仓库路径
    importlib 加载旧 train.py 建模型                            -> aq_unet.AquariusUNet
    build_model(base=128, mults=(1,2,4), img_size=64)           -> base=256, ch=[256,512,1024,1024]
    量化层标记 getattr(m, "quantizable")                        -> getattr(m, "nlt_quant")
    量化函数 mod.ste_quant(t, mode)                             -> aq_unet.nlt_quantize(t)

产物形态（详见 aq_lowbit.py 的模块文档）：

    ternary  -> 量化层 base-3 打包（1.585 bit/权重）+ g128 fp16 scale；非量化层 fp16 ≈ 153.0 MB
    binary   -> 量化层 1-bit 打包 + g128 fp16 scale；非量化层 fp16        ≈ 112.4 MB
    fp       -> 全部 fp16（对照基准，零压缩）                          ≈ 1116.7 MB

用法：

    python code/aq_pack.py                                   # ternary，输出 out/
    python code/aq_pack.py --ckpt checkpoints/ternary/ckpt_040000.pt
    python code/aq_pack.py --mode binary --out out/bin.safetensors
    python code/aq_pack.py --sizes-only                      # 只按结构算体积，不读检查点

**产物文件名默认带训练步数后缀**（`aquarius_ternary_step40000.safetensors`），
这样一眼就能看出玩的是哪次检查点；多份导出可共存，`aq_play.default_ckpt()`
会自动挑步数最大的那份。

产物可直接被 play/aq_play.py / play/app.py 加载（`--ckpt xxx.safetensors`）。
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
WS = HERE.parent
sys.path.insert(0, str(HERE))

import aq_lowbit as lb                                          # noqa: E402
import aq_paths                                                 # noqa: E402
from aq_unet import AquariusUNet, count_params, set_quant_mode   # noqa: E402

CFG = dict(block_out_channels=[256, 512, 1024, 1024],
           cross_attention_dim=1024, heads=8)

# aq_export_size.py 的预测口径：量化层 2bit/1bit 打包 + 每组 fp32 scale，其余 fp16
G = 128

# 用户约定（2026-10-02）：「以后你直接给我新的量化完的检查点文件即可，
# 直接放在这个自动备份的文件夹内，我直接替换」。
#
# ⚠️ 目录**自动探测**（`aq_paths.sync_root()`），不写死 —— 本机是
# `C:\Users\ajifang\...`，目标机用户名是 `Du`。写死会导致在新机器上
# **在错误的路径建出一棵新目录树**。
#
# ⚠️ 放**子目录**而不是 SyncDisk 根：
#   root/Aquarius/        是 aq_archive.py 的轮转树（其 prune 会删 ckpt_*.pt）
#   root/Aquarius_models/ 不在归档计划的任何一条里 → 绝不可能被 prune 碰到
# ⚠️ 相对同步根的路径。2026-10-04 归档重组后模型投放目录搬到了这里。
SYNC_SUBDIR = "Aquarius20261004/04_models"


def build_model():
    set_quant_mode("fp")
    return AquariusUNet(CFG)


def quantized_names(model):
    """量化层的 `.weight` 名字集合。标记是 `nlt_quant`（NoQuant* 覆写为 False）。"""
    return {n + ".weight" for n, m in model.named_modules()
            if getattr(m, "nlt_quant", False) is True}


def predict_bytes(sd, q_names, mode, packing="b3", scale_dtype="fp16"):
    """按结构预测导出字节数（与 code/aq_export_size.py 同一口径）。

    packing     "b2" = 2 bit/三值（旧）· "b3" = base-3，5 位/字节 = 1.6 bit/三值（新默认）
                ⚠️ 用 log2(3)=1.585 做预测，与 pack_ternary3 的实际字节数一致（都向上取整到字节）。
    scale_dtype "fp32" = 4 B/组（旧）· "fp16" = 2 B/组（新默认）
    """
    n_q = n_f = 0
    b_q = b_f = 0
    b3 = (packing == "b3" and mode == "ternary")
    sb = 2 if scale_dtype == "fp16" else 4
    for name, t in sd.items():
        if mode != "fp" and name in q_names and t.dtype.is_floating_point:
            n = t.numel()
            ng = (n + G - 1) // G
            if b3:
                b_q += math.ceil(n / 5) + ng * sb          # 5 个三值位/字节
            else:
                bits = 2 if mode == "ternary" else 1
                b_q += n * bits / 8 + ng * sb
            n_q += n
        else:
            b_f += t.numel() * 2
            n_f += t.numel()
    return b_q + b_f, n_q, n_f


def load_ema(ckpt, model):
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict) or not (blob.get("ema") or blob.get("model")):
        raise SystemExit(f"[abort] {ckpt} 不像 Aquarius 检查点")
    w = blob.get("ema") or blob.get("model")
    model.load_state_dict(w)
    src = "ema" if blob.get("ema") else "model"
    return blob.get("step"), src


def default_sync_dir():
    """自动探测同步目录下的投放位置；探测不到返回 None（调用方跳过并说明）。"""
    root = aq_paths.sync_root()
    return (root / SYNC_SUBDIR) if root is not None else None


def sync_to_baidu(local, mode, sync_dir, dry=False):
    """把打包好的量化模型复制到百度网盘同步目录，供用户直接替换。

    只同步**量化成品**（ternary / binary）。`mode=fp` 是 1116.7MB 的未压缩对照基准，
    不是交付物，同步它只会白占上传流量。

    返回 (是否成功, 说明文字)。
    """
    if mode == "fp":
        return False, "mode=fp 是未压缩对照基准，不同步（只同步量化成品）"
    if sync_dir is None:
        return False, ("未找到百度云同步目录，已跳过投放"
                       "（可用 AQ_SYNC_DIR 显式指定，或 --sync-dir）")
    d = Path(sync_dir)
    if not d.parent.is_dir():
        return False, f"同步目录的上级不存在，跳过：{d.parent}（换机器时正常）"
    d.mkdir(parents=True, exist_ok=True)
    dst = d / Path(local).name
    if dry:
        return True, f"[dry-run] 会复制到 {dst}"
    import shutil
    shutil.copy2(local, dst)
    mb = dst.stat().st_size / 1e6
    # 同名老文件会被覆盖 —— 这正是用户要的"我直接替换"
    return True, f"已同步到百度云同步目录：{dst}（{mb:.1f} MB）"


def main():
    ap = argparse.ArgumentParser(description="Aquarius 权重打包（ternary 2-bit / binary 1-bit）")
    ap.add_argument("--ckpt", default=str(WS / "checkpoints" / "ternary" / "ckpt_040000.pt"),
                    help="训练检查点（用编号快照，别用 latest.pt —— 训练会边跑边重写它）")
    ap.add_argument("--mode", choices=["ternary", "binary", "fp"], default="ternary")
    ap.add_argument("--packing", choices=["b2", "b3"], default="b3",
                    help="b3（新默认）= base-3，三值 5 位/字节，省 21%%；b2 = 旧的 2 bit/三值")
    ap.add_argument("--scale-dtype", choices=["fp32", "fp16"], default="fp16",
                    help="fp16（新默认）省一半 scale；fp32 = 旧格式")
    ap.add_argument("--out", default=None,
                    help="输出路径（默认 out/aquarius_{mode}_step{步数}.safetensors，"
                         "文件名自动带训练步数）")
    ap.add_argument("--sync-dir", default=None,
                    help="打包后自动复制到该目录供用户替换"
                         "（默认自动探测 <百度云同步目录>/Aquarius_models）")
    ap.add_argument("--no-sync", action="store_true", help="不复制到同步目录")
    ap.add_argument("--sizes-only", action="store_true",
                    help="只按模型结构预测体积，不读检查点、不写文件")
    ap.add_argument("--no-verify", action="store_true", help="跳过逐层往返校验（快，但别这么用）")
    args = ap.parse_args()

    model = build_model()
    sd = model.state_dict()
    q_names = quantized_names(model)
    print(f"[model] 参数 {count_params(model):,}  "
          f"量化层 {len(q_names)} 个  "
          f"量化参数 {sum(sd[n].numel() for n in q_names if n in sd):,}", flush=True)

    pred, n_q, n_f = predict_bytes(sd, q_names, args.mode, packing=args.packing, scale_dtype=args.scale_dtype)
    print(f"[pred ] 预测导出体积 {pred / 1e6:.2f} MB "
          f"（量化 {n_q / sum(t.numel() for t in sd.values()) * 100:.1f}% 参数，"
          f"其余 fp16）", flush=True)
    if args.sizes_only:
        return 0

    ckpt = Path(args.ckpt)
    if not ckpt.is_file():
        raise SystemExit(f"[abort] 找不到检查点 {ckpt}")
    t0 = time.time()
    step, src = load_ema(ckpt, model)
    print(f"[ckpt ] {ckpt.name}  step={step}  权重来源={src}  "
          f"读取耗时 {time.time() - t0:.1f}s", flush=True)

    print(f"[pack ] mode={args.mode} ...", flush=True)
    t0 = time.time()
    tensors, meta, rows = lb.pack_state_dict(sd, q_names, args.mode,
                                            packing=args.packing,
                                            scale_dtype=args.scale_dtype)
    meta["step"] = str(step)
    meta["source"] = ckpt.name
    dt = time.time() - t0

    worst_code = max(r[3] for r in rows)
    worst_deq = max(r[4] for r in rows)
    print(f"[pack ] {len(tensors)} 个张量，耗时 {dt:.1f}s", flush=True)
    print(f"[check] 码位往返最大差  {worst_code:.1e}   （必须 0 —— 位打包无损）")
    print(f"[check] 反量化 vs 训练前向 w_q 最大差 {worst_deq:.3e}"
          f"   （STE 包装的 fp32 舍入，量级 1e-7）", flush=True)

    # 默认文件名**带训练步数后缀** —— 用户 2026-10-02 明确要求：
    # 「用那个步数作为模型的后缀，不然我每次都不知道玩的是哪次检查点的模型」。
    # 多份不同步数的导出可以共存，aq_play.default_ckpt() 会自动挑步数最大的。
    stem = f"aquarius_{args.mode}" + (f"_step{step}" if step is not None else "")
    out = Path(args.out) if args.out else (WS / "out" / f"{stem}.safetensors")
    out.parent.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import load_file, save_file
    save_file(tensors, str(out), metadata=meta)

    # 从磁盘回读，逐张量比对（抓序列化/截断问题）
    back = load_file(str(out))
    same = (len(back) == len(tensors)
            and all(torch.equal(back[k], v) for k, v in tensors.items()))

    # ⚠️ 关键：用**真正的加载器**把文件装回一个空模型，再和源权重比。
    # 内部一致性校验（上面的 code/dequant 比对）不能替代这一步 ——
    # 2026-10-02 就因为只测了打包器内部路径、没测加载器，
    # 漏掉了「盘上 scale 是 1 维、打包器内部是 2 维」的广播 bug。
    #
    # 两类层要**分开看阈值**，混在一起比必然误报：
    #   量化层      解包后应逐位还原，与 nlt_quantize(w) 差 ~1e-8（STE 的 fp32 舍入）
    #   非量化层    **按格式规定存 fp16**，与 fp32 相差至多 2^-11 = 4.883e-04 ——
    #               这是格式本身的性质，不是缺陷（实测最大差恰好就是这个数）
    q_diff = f_diff = None
    if not args.no_verify:
        probe = build_model()
        from aq_unet import nlt_quantize as _nq, set_quant_mode as _sqm
        lb.load_packed_into(probe, out, verbose=False)
        got = probe.state_dict()
        _sqm(args.mode if args.mode != "fp" else "fp")
        q_diff = f_diff = 0.0
        for name in q_names:
            if name not in sd or name not in got:
                continue
            ref = _nq(sd[name].detach().cpu()) if args.mode != "fp" else sd[name]
            q_diff = max(q_diff, (got[name].float() - ref.float()).abs().max().item())
        for name, t in sd.items():
            if name in q_names or not t.dtype.is_floating_point:
                continue
            f_diff = max(f_diff, (got[name].float() - t.float()).abs().max().item())
        _sqm("fp")
        fp16_bound = 2 ** -11 * max(1.0, max(abs(t.max().item())
                                            for n, t in sd.items()
                                            if n not in q_names
                                            and t.dtype.is_floating_point))
        # ⚠️ 量化层的上界**必须随 scale_dtype 变**（2026-10-04 修）：
        #   scale_dtype="fp32" → scale 精确，上界就是 STE 的 fp32 舍入 ~1e-8，用 1e-6 判。
        #   scale_dtype="fp16" → **每组 scale 本身被舍入到 fp16**（相对 2^-11），
        #     反量化值 w=code*scale 就有 |Δw| ≲ max|scale|·2^-11 的地板误差。
        #     实测 step205000：max|scale|=0.09857 → 上界 4.813e-05，实测 3.000e-05 ✅。
        #   旧代码对量化层硬编码 1e-6，是 fp32-scale 时代的遗产 ⇒ fp16 默认下**必然误报**
        #   （打包本身没问题：码位 0 差异、回读逐张量一致，只是 scale 的 fp16 舍入被算进差值）。
        if args.scale_dtype == "fp16":
            max_scale = max(tensors[n2[:-len(".weight")] + ".gscale"].float().abs().max().item()
                            for n2 in q_names
                            if n2.endswith(".weight") and n2[:-len(".weight")] + ".gscale" in tensors)
            q_bound = max_scale * 2 ** -11 * 1.001
        else:
            q_bound = 1e-6
        ok_q = q_diff <= q_bound
        ok_f = f_diff <= fp16_bound * 1.001
        print(f"[check] 加载器回装 · 量化层 最大差 {q_diff:.3e}  "
              f"{'✅' if ok_q else '❌'}（上限 {q_bound:.3e}"
              f"{' = max|scale|·2^-11（fp16 scale）' if args.scale_dtype == 'fp16' else '（fp32 scale）'}）",
              flush=True)
        print(f"[check] 加载器回装 · 非量化层 最大差 {f_diff:.3e}  "
              f"{'✅' if ok_f else '❌'}（fp16 上限 {fp16_bound:.3e} = 2^-11）", flush=True)

    disk = os.path.getsize(out)
    nparams = sum(r[1] for r in rows)
    print()
    print("=== 结果 ===")
    print(f"  文件      {out}")
    print(f"  磁盘体积  {disk / 1e6:.2f} MB   （{disk / 1e9:.3f} GB）")
    print(f"  预测体积  {pred / 1e6:.2f} MB   "
          f"差值 {(disk - pred) / 1024:.1f} KiB（safetensors 头部）")
    print(f"  bit/权重  {disk * 8 / nparams:.3f}")
    print(f"  参数量    {nparams:,}（量化 {n_q:,} / 非量化 {n_f:,}）")
    print(f"  fp16 等价 {nparams * 2 / 1e6:.1f} MB  →  压缩 {nparams * 2 / disk:.2f}×")
    print(f"  回读校验  {'✅ 逐张量一致' if same else '❌ 不一致'}")
    print(f"  码位无损  {'✅ 0 差异' if worst_code == 0 else f'❌ {worst_code:.1e}'}")
    if q_diff is not None:
        print(f"  加载器    {'✅ 量化层在格式上限内' if ok_q else '❌ 量化层有偏'}"
              f"（{q_diff:.3e} ≤ {q_bound:.3e}，fp16 scale 舍入地板）"
              f" + 非量化层 fp16（{f_diff:.3e} = 2^-11 量级，符合格式）")
    print()

    # ---- 复制到百度云同步目录（用户约定：直接给他文件，他自己替换）--------
    if args.no_sync:
        sdir = None
    else:
        sdir = Path(args.sync_dir) if args.sync_dir else default_sync_dir()
    synced, note = sync_to_baidu(out, args.mode, sdir)
    print(f"  {'☁️ ' if synced else '   '}{note}")
    print(f"  用运行时加载：python play/aq_play.py \"sunset over the ocean\" --ckpt \"{out}\"")
    json.dump(meta, open(out.with_suffix(".meta.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    ok = same and worst_code == 0 and (q_diff is None or ok_q)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
