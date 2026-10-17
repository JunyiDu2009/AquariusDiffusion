# -*- coding: utf-8 -*-
"""Aquarius 客观质量指标 —— 基于**潜向量分布距离**，不是像素差。

为什么需要它（2026-10-03 实测动机）：
  跨图 L1 只能检测**塌缩**，不是质量分。实测三个端点：
  塌缩 0.00 · 当前模型 60.98 · **纯噪声 85.35（最高）** ⇒ L1 越高不代表越好。
  它落在 0~85 之间时，**分不清"在学习"还是"在滑向塌缩"**。
  本脚本改用**分布距离**：生成潜向量的分布 与 真实 COCO 潜向量的分布 差多远。

----------------------------------------------------------------------
第一版（2026-10-03 上午）踩到的坑 —— 全部写在这里，别重犯
----------------------------------------------------------------------
第一版把「潜向量自适应池化到 4×4 再算对角高斯 Fréchet」当成质量分，
结果拿到一个**反直觉且错误**的读数：

    step  9,058（灰糊拼贴）score = 1.975   ← 号称"与真实分布几乎不可区分"
    step 116,000（草地+物体）score = 7.159  ← 号称"离真实分布很远"

但看图，后者**明显好得多**。三个原因，逐个修：

**坑 1 · 均值项在奖励"灰糊"。**
真实 COCO 池逐通道均值 ≈ 0（大量不同场景平均掉），所以**一张灰糊图**
（潜向量 ≈ 0）的逐格均值也 ≈ 0 ⇒ 均值项 ≈ 0 ⇒ 得分极好。
模型一旦学会输出"草地/天空/地平线"，逐格均值反而**偏离** 0 ⇒ 被罚。
⇒ 修法：特征必须包含**高通分量**（去掉 4×4 块均值后的残差），
  灰糊图的高通能量 ≈ 0，会被正确判成"离真实很远"。

**坑 2 · 样本量不对等，分数系统性偏低。**
旧实现：`F(gen[24], real[192])`，而噪声地板是 `F(real[24], real[24])`。
有限样本会让 F 偏大，两侧偏大的程度不同 ⇒ 分子被低估 ⇒ **分数偏乐观**。
⇒ 修法：分子分母**用同样的样本量**，并且各自**重复取中位**。

**坑 3 · "真实池"根本不是随机的。**
旧实现只从**第一个 shard** 顺序取前 n 张 ⇒ COCO id 连续、内容高度相关。
基线一偏，所有分数跟着偏。⇒ 修法：跨 shard 按比例随机取连续块。

另外两条**被实测推翻**的直觉，也记在这里：

**✘ "锐度越高越好" —— 在本项目当前阶段是错的。**
像素梯度幅值实测：step 9,058（糊）**17.97** · step 38,000 **19.24** ·
step 66,700 **15.17** · step 87,800 **14.56** · step 113,100（好）**14.93**。
**越早期的糊模型反而越"锐"** —— 因为它输出的是硬边拼贴块，梯度天然大。
所以本脚本里的 `锐度` 只作**诊断读数**，方向是 **→ 真实值**（不是越高越好），
真正的判据是「高通能量比」和分布距离。

**✘ "Fréchet 越低模型越好" —— 只在这个特征集上、且必须配合塌缩判据才成立。**

----------------------------------------------------------------------
方法
----------------------------------------------------------------------
1. **同桶比较**：潜向量统计量随分辨率变化，生成与真实必须取**同一个桶**。
   默认 `512x640`（真实样本 35,372 张；`512x512` 不到 300 张，不能用）。
2. **特征 = 低频 ⊕ 高频**，各 4×G×G 维：
     LF = avgpool(x, G)                        ← 布局 / 色彩 / 亮度
     HF = avgpool(|x − upsample(avgpool(x,G))|, G)  ← 纹理 / 边缘能量图
   合并 2·4·G·G 维（G=4 → 128 维）。
   **HF 是这次修复的核心**：它让"灰糊"无法再靠均值项蒙混过关。
3. **对角高斯 Fréchet 距离**（对对角协方差是精确解）：
       F = Σ_d [ (μ_g[d] − μ_r[d])² + (σ_g[d] + σ_r[d] − 2√(σ_g[d]·σ_r[d])) ]
   用对角形式是刻意的：十几个样本估不出 128×128 完整协方差，强行求逆得到垃圾。
4. **⭐ 有限样本的噪声必须校准掉**：
   从真实池里抽**两个不相交的、同样本量子集**，算 `F_floor = F(realA, realB)`。
   它代表"同一分布、同样本量下，这个指标本来就有多大读数"。
   最终看 **`score = F_gen_real / F_floor`**：
       score ≈ 1 → 生成分布与真实分布在该样本量下**不可区分**
       score 越小越接近；越大越远
   分子分母**样本量相同**、且各自重复 16 次取中位 ⇒ 跨天可比。
   ⚠️ **但 score 只在同样的 `--n` 下才可比**：n 越小，地板（有限样本噪声）越大，
   比值就越小。实测同一个模型 `n=6 → score 1.03`、`n=24 → 7.16`（旧公式）。
   **晨报固定 `--n 16`；历史 JSON 里 n 不同就不要直接比较。**
5. **塌缩诊断**：逐维方差比 + `跨图 L1`。真正的塌缩是**所有提示词出同一张图**。
6. **条件注入比**（`--cond-k` > 0 时）：同一初始噪声、换提示词 的图间距离
   ÷ 同一提示词、换初始噪声 的图间距离。≈1 说明**文本条件被忽略**（塌缩的精确机制）。

全程 **CPU**（训练占着 GPU，本机硬约定同一时刻只允许 1 个 CUDA 进程）。

用法
----
    python code/aq_metrics.py                          # 最新可用检查点，512x640
    python code/aq_metrics.py --ckpt <path> --n 16
    python code/aq_metrics.py --bucket 448x640 --n 24
    python code/aq_metrics.py --dry-run                # 只算真实侧基线（几秒）
    python code/aq_metrics.py --cond-k 4               # 加测「条件注入比」（多 8 张）
    python code/aq_metrics.py --budget-min 40
"""
import argparse
import json
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as Fn

WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "code"))
sys.path.insert(0, str(WS / "play"))
import aq_paths                                                   # noqa: E402
import aq_play as ap                                              # noqa: E402
from aq_unet import AquariusUNet                                  # noqa: E402

LATENTS = WS / "Aquarius_cloud" / "latents"
CAPTIONS = WS / "Aquarius_cloud" / "captions" / "captions_train2017.json"
CKPT_DIR = WS / "checkpoints" / "ternary"
SYNC_ROOT = aq_paths.sync_root()
OUT_ROOT = aq_paths.handover_dir() or (WS / "deliverables")

FEAT_GRID = 4            # 每侧 4ch × G × G（G=4 → 每侧 64 维，合计 128 维）
REAL_SEED = 20261003     # 真实池抽样种子（固定 ⇒ 跨天可比）
N_REPEAT = 16            # 分子/分母各重复取中位的次数
REAL_POOL = 512          # 真实池大小（必须 ≥ 2·n·若干，否则地板估不稳）


# ==========================================================================
# 真实侧
# ==========================================================================
def bucket_counts():
    """各桶的真实样本数（只读 safetensors 头部，很快）。"""
    from safetensors import safe_open
    cnt = {}
    for f in LATENTS.glob("latents_*.safetensors"):
        m = re.match(r"latents_(\d+)x(\d+)_", f.name)
        if not m:
            continue
        try:
            with safe_open(str(f), framework="pt") as h:
                n = h.get_slice("latents").get_shape()[0]
        except Exception:                                      # noqa: BLE001
            continue
        k = (int(m.group(1)), int(m.group(2)))
        cnt[k] = cnt.get(k, 0) + n
    return cnt


def parse_bucket(s):
    a, b = str(s).lower().split("x")
    return int(a), int(b)          # (H, W) —— 与 latents_ 文件名一致


def load_real_latents(bucket, n, seed=REAL_SEED):
    """跨 shard **随机**抽 n 张真实潜向量。

    ⚠️ 第一版只从第一个 shard 顺序取前 n 张 —— 那批 COCO id 连续、内容相关，
    **不是随机样本**，会让基线系统性偏移。这里按 shard 大小分摊配额，
    每个 shard 内随机取一**连续块**（够随机，且比逐行读快得多）。
    """
    from safetensors import safe_open
    H, W = bucket
    files = sorted(LATENTS.glob(f"latents_{H}x{W}_shard*.safetensors"))
    if not files:
        raise SystemExit(f"找不到桶 {H}x{W} 的潜数据（{LATENTS}）")
    sizes = []
    for f in files:
        with safe_open(str(f), framework="pt") as h:
            sizes.append(h.get_slice("latents").get_shape()[0])
    total = sum(sizes)
    g = torch.Generator().manual_seed(seed)
    quota = [max(1, round(n * s / total)) for s in sizes]
    while sum(quota) < n:                       # 四舍五入会少读几行，补够
        quota[max(range(len(sizes)), key=lambda i: sizes[i])] += 1
    chunks = []
    for f, tot, q in zip(files, sizes, quota):
        q = min(q, tot)
        if q <= 0:
            continue
        start = int(torch.randint(0, max(1, tot - q + 1), (1,), generator=g))
        with safe_open(str(f), framework="pt") as h:
            chunks.append(h.get_slice("latents")[start:start + q].float())
    lat = torch.cat(chunks, 0)
    idx = torch.randperm(lat.shape[0], generator=g)[:min(n, lat.shape[0])]
    return lat[idx]


def load_prompts(n, seed=REAL_SEED):
    """取 n 条互不相同的 COCO 英文提示词（确定性 ⇒ 跨检查点/跨天完全一致）。"""
    try:
        anns = json.loads(CAPTIONS.read_text(encoding="utf-8")).get("annotations", [])
    except Exception as e:                                     # noqa: BLE001
        raise SystemExit(f"读不到 captions：{e}")
    seen, out = set(), []
    rng = random.Random(seed)
    rng.shuffle(anns)
    for a in anns:
        c = str(a.get("caption", "")).strip()
        if c and c not in seen:
            seen.add(c)
            out.append((int(a.get("image_id", 0)), c))
        if len(out) >= n:
            break
    return out


# ==========================================================================
# 特征
# ==========================================================================
def feats_split(lat, grid=FEAT_GRID):
    """(N,4,h,w) -> (LF (N,4gg), HF (N,4gg))。

    LF = 块均值 → 布局/色彩/亮度
    HF = 减去块均值后的**残差幅值**再池化 → 纹理/边缘能量图
    """
    x = lat.float()
    low = Fn.adaptive_avg_pool2d(x, grid)
    up = Fn.interpolate(low, size=x.shape[-2:], mode="nearest")
    hf = (x - up).abs()
    return low.reshape(x.shape[0], -1), Fn.adaptive_avg_pool2d(hf, grid).reshape(x.shape[0], -1)


def feats_joined(lat, grid=FEAT_GRID):
    a, b = feats_split(lat, grid)
    return torch.cat([a, b], 1)


def hf_energy(lat):
    """高通能量（标量）：模型有没有纹理/边缘，还是糊成一片。"""
    x = lat.float()
    low = Fn.adaptive_avg_pool2d(x, 4)
    up = Fn.interpolate(low, size=x.shape[-2:], mode="nearest")
    return float((x - up).abs().mean())


ACF_LAGS = (2, 4, 8)


def acf(lat, lags=ACF_LAGS):
    """归一化空间自相关（对角方向，按窗口内方差归一）。返回每个 lag 的均值。

    为什么加它（2026-10-03 实测）：**它能把「碎块拼贴」和「有结构的图」分开**，
    而 HF / 高频比做不到 —— 9,058 的硬边拼贴高频统计量**恰好接近真实**，
    所以那些指标反而给它好评。实测：

        lag         L2      L4      L8
        真实       +0.373  +0.279  +0.158
        step 9,058 +0.232  +0.175  +0.090   ← 大尺度上"散架"
        step 117000+0.287  +0.221  +0.109   ← 明显更接近真实
    """
    g = lat.float()
    g = g - g.mean(dim=(2, 3), keepdim=True)
    v = g.var(dim=(2, 3), unbiased=False).mean().clamp_min(1e-12)
    return [float((g[..., L:, L:] * g[..., :-L, :-L]).mean() / v) for L in lags]


def acf_ratio(gen, real, lags=ACF_LAGS):
    """生成/真实 的自相关比值（逐 lag 平均）。**越接近 1 越好**（<1 = 大尺度偏"散架"）。"""
    ag, ar = acf(gen, lags), acf(real, lags)
    return float(sum(a / b for a, b in zip(ag, ar) if abs(b) > 1e-9) / len(lags))


def sharpness(lat):
    """平均梯度幅值 —— **诊断读数，方向是"越接近真实值越好"，不是越高越好**。

    实测（像素空间，历史 PNG）：step 9,058（差）= 17.97 · 113,100（好）= 14.93。
    **早期糊模型的硬边拼贴反而更"锐"**，所以这一项不能当质量分，只能当参考。
    """
    g = lat.float()
    dx = (g[..., :, 1:] - g[..., :, :-1]).abs().mean()
    dy = (g[..., 1:, :] - g[..., :-1, :]).abs().mean()
    return float((dx + dy) / 2)


def frechet_diag(A, B):
    """对角高斯的 Fréchet 距离（对对角协方差是精确解）。A/B: (N,D) fp32。"""
    m1, m2 = A.mean(0), B.mean(0)
    s1 = A.var(0, unbiased=True)
    s2 = B.var(0, unbiased=True)
    mean_term = ((m1 - m2) ** 2).sum().item()
    cov_term = (s1 + s2 - 2.0 * torch.sqrt(s1.clamp_min(1e-12) *
                                           s2.clamp_min(1e-12))).sum().item()
    return mean_term + cov_term, mean_term, cov_term


def pair_draws(pool, n, k, seed):
    """把池子随机切成 2k 个**不相交**的 n 块，返回 k 组 (A,B)。不够就自动降 k。"""
    N = pool.shape[0]
    k = max(1, min(k, N // (2 * n)))
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(N, generator=g)
    out = []
    for i in range(k):
        A = pool[idx[i * n:(i + 1) * n]]
        B = pool[idx[(k + i) * n:(k + i + 1) * n]]
        out.append((A, B))
    return out


def floor_estimate(pool, n, seed=REAL_SEED, k=N_REPEAT):
    """真实池抽两个不相交、同样本量子集算 F —— **有限样本噪声地板**。取中位。"""
    vals = sorted(frechet_diag(A, B)[0] for A, B in pair_draws(pool, n, k, seed + 1))
    return vals[len(vals) // 2], vals[0], vals[-1]


def val_estimate(gen_feat, pool, n, seed=REAL_SEED, k=N_REPEAT):
    """F(gen, real_subset) —— 子集大小 = n，**与地板同样本量**。取中位。"""
    N = pool.shape[0]
    k = max(1, min(k, N // n))
    g = torch.Generator().manual_seed(seed + 2)
    idx = torch.randperm(N, generator=g)
    vals = sorted(frechet_diag(gen_feat, pool[idx[i * n:(i + 1) * n]])[0]
                  for i in range(k))
    return vals[len(vals) // 2], vals[0], vals[-1]


# ==========================================================================
# 采样
# ==========================================================================
def ddim_local(model, ctx, abar, steps, seed, device, h, w):
    """**直接复用 `aq_play.ddim`** —— 不要再在这里复制一份采样循环。

    2026-10-03 的教训：这里原来抄了一份、并把末尾的 `clamp(-1,1)` 换成 ±6 守卫，
    而 `aq_play.ddim` 那边仍是 `clamp(-1,1)` ⇒ **指标量的是"没夹"的输出，
    晨报出的图是"夹过"的**，两边根本不是同一个模型行为。
    现在 `aq_play.LAT_CLAMP` 已统一为 ±6σ（真实潜向量 std ≈ 0.835、|x| > 1 占 22%，
    夹 ±1 等于夹 ±1.2σ，会削掉 26% 的数值），所以两边口径一致。
    """
    return ap.ddim(model, ctx, abar, steps, seed, device, h, w)


def sample_latents(model, tok, te, prompts, bucket, steps, on_status=None,
                   budget_min=None, seed_base=100000):
    """CPU 采样。每个提示词用不同种子（**同种子会共享初始噪声、污染统计量**）。"""
    say = on_status or (lambda _m: None)
    H, W = bucket
    abar = ap.cosine_schedule()
    out, t0 = [], time.time()
    for i, (_iid, p) in enumerate(prompts):
        if budget_min and (time.time() - t0) / 60 > budget_min:
            say(f"⏱ 预算 {budget_min:.0f} 分钟用完，已采 {len(out)}/{len(prompts)}")
            break
        ctx = ap.encode_texts([p], "cpu", tok, te)
        out.append(ddim_local(model, ctx, abar, steps, seed_base + i,
                              "cpu", H, W)[0].float().cpu())
        if (i + 1) % 4 == 0 or i == 0:
            say(f"[{i + 1}/{len(prompts)}] {(time.time() - t0) / 60:.1f} min")
    return torch.stack(out) if out else torch.zeros(0)


def cond_probe(model, tok, te, prompts, bucket, steps, k, on_status=None):
    """条件注入比：同一初始噪声换提示词 ÷ 同一提示词换初始噪声。

    ≈1 或更低 ⇒ 文本条件被忽略（这正是 2026-10-02 binary 塌缩的精确机制）。
    """
    say = on_status or (lambda _m: None)
    H, W = bucket
    abar = ap.cosine_schedule()
    use = prompts[:k]
    base, alt = [], []          # base: 全部用同一噪声 777（只有提示词不同）
    t0 = time.time()            # alt : 同提示词、换成噪声 888
    for i, (_iid, p) in enumerate(use):
        ctx = ap.encode_texts([p], "cpu", tok, te)
        base.append(ddim_local(model, ctx, abar, steps, 777, "cpu", H, W)[0].float())
        alt.append(ddim_local(model, ctx, abar, steps, 888, "cpu", H, W)[0].float())
        say(f"cond [{i + 1}/{k}] {(time.time() - t0) / 60:.1f} min")
    import itertools
    # 同噪声、换提示词 ⇒ 距离应当大（说明文本条件生效）
    dp = [(a - b).abs().mean().item() for a, b in itertools.combinations(base, 2)]
    d_prompt = sum(dp) / len(dp) if dp else float("nan")
    # 同提示词、换噪声 ⇒ 距离反映"与条件无关的固有变异"
    d_seed = sum((a - b).abs().mean().item() for a, b in zip(base, alt)) / len(base)
    return d_prompt, d_seed, (d_prompt / d_seed if d_seed > 1e-9 else float("nan"))


# ==========================================================================
def main():
    ap_arg = argparse.ArgumentParser(description="Aquarius 潜向量分布质量指标")
    ap_arg.add_argument("--mode", default="ternary")
    ap_arg.add_argument("--ckpt", default=None, help="默认取最新可用的编号快照")
    ap_arg.add_argument("--bucket", default="512x640", help="(H x W)，须有充足真实样本")
    ap_arg.add_argument("--n", type=int, default=16, help="生成的样本数")
    ap_arg.add_argument("--steps", type=int, default=20)
    ap_arg.add_argument("--pool", type=int, default=REAL_POOL, help="真实池大小")
    ap_arg.add_argument("--grid", type=int, default=FEAT_GRID, help="特征网格 G")
    ap_arg.add_argument("--budget-min", type=float, default=60.0)
    ap_arg.add_argument("--cond-k", type=int, default=0,
                        help=">0 时额外测「条件注入比」（多采 2k 张）")
    ap_arg.add_argument("--dry-run", action="store_true",
                        help="只算真实侧的基线（噪声地板），不加载模型、不采样")
    ap_arg.add_argument("--json-out", default=None)
    ap_arg.add_argument("--lat-out", default=None)
    args = ap_arg.parse_args()

    bucket = parse_bucket(args.bucket)
    H, W = bucket
    counts = bucket_counts()
    have = counts.get(bucket, 0)
    print("=" * 74)
    print(f"桶 {H}x{W} · 真实样本 {have:,} 张 · 特征 2×4×{args.grid}² "
          f"={2 * 4 * args.grid ** 2} 维（低频⊕高频） · 生成 {args.n} 张 · "
          f"{args.steps} 步")
    if have < 200:
        print(f"  ⚠️ 该桶真实样本只有 {have} 张，统计量会不稳。"
              f"建议改用样本多的桶（448x640 40,306 / 512x640 35,372）")
    print("=" * 74)

    # ---- 真实池 -----------------------------------------------------------
    need_pool = max(args.pool, 4 * args.n)
    real = load_real_latents(bucket, need_pool)
    print(f"真实池 {tuple(real.shape)}（跨 shard 随机抽取）")
    LF_r, HF_r = feats_split(real, args.grid)
    Rf = torch.cat([LF_r, HF_r], 1)
    n_LF = LF_r.shape[1]
    print(f"  高通能量（真实）= {hf_energy(real):.4f} · "
          f"潜向量锐度（真实）= {sharpness(real):.4f}")

    if args.dry_run:
        fl, lo, hi = floor_estimate(Rf, args.n)
        print(f"⭐ 噪声地板 F(realA,realB) 中位 {fl:.4f}（区间 {lo:.4f}~{hi:.4f}，"
              f"同样本量 {args.n}）")
        print("\n--dry-run：到此为止（真实侧已就绪）")
        return 0

    # ---- 检查点 ----------------------------------------------------------
    ckpt = Path(args.ckpt) if args.ckpt else None
    if ckpt is None:
        p, why = aq_paths.newest_good_ckpt(CKPT_DIR)
        if p is None:
            raise SystemExit(f"没有可用检查点：{why}")
        ckpt = p
        print(f"检查点 {p.name}（{why}）")
    else:
        ok, why = aq_paths.ckpt_ok(ckpt)
        print(f"检查点 {ckpt.name}：{'✅' if ok else '❌'} {why}")
        if not ok:
            raise SystemExit("指定的检查点不可用")

    # ⚠️ 用 `ap.load_unet` 而不是自己 torch.load：
    # 训练检查点（*.pt）读 ema，**打包交付文件（*.safetensors，153.0 MB）读位打包权重**。
    # 两者数学等价但**不完全相同**（检查点路径的 ema 存 fp16，会让 0.005% 的三值码位翻转），
    # 所以「验收最终交付物」时**必须能直接评打包文件**。
    model = ap.load_unet(ckpt, "cpu")
    step = (model._aq_ckpt or {}).get("step")
    tok, te = ap.load_te("cpu")
    print(f"模型 step={step} 已在 CPU 就绪"
          f"（{ '位打包' if (model._aq_ckpt or {}).get('packed') else '训练检查点' }）\n")

    # ---- 采样 -------------------------------------------------------------
    prompts = load_prompts(args.n)
    gen = sample_latents(model, tok, te, prompts, bucket, args.steps,
                         on_status=lambda m: print("   " + m, flush=True),
                         budget_min=args.budget_min)
    n_gen = gen.shape[0]
    if n_gen < 4:
        raise SystemExit(f"只采到 {n_gen} 张，不足以下结论")
    LF_g, HF_g = feats_split(gen, args.grid)
    Gf = torch.cat([LF_g, HF_g], 1)

    # ---- 三个分数：合并 / 低频 / 高频（**分子分母同样本量**）-------------
    F_val, m_term, c_term = val_estimate(Gf, Rf, n_gen)
    floor_n, flo_n, fhi_n = floor_estimate(Rf, n_gen)
    score = F_val / max(floor_n, 1e-9)

    def sub_score(Gsel, Rsel):
        fv, _, _ = val_estimate(Gsel, Rsel, n_gen)
        fl, _, _ = floor_estimate(Rsel, n_gen)
        return fv / max(fl, 1e-9), fv, fl
    score_LF, F_LF, floor_LF = sub_score(LF_g, LF_r)
    score_HF, F_HF, floor_HF = sub_score(HF_g, HF_r)

    # ---- 物理读数 ---------------------------------------------------------
    var_ratio = (Gf.var(0, unbiased=True) / Rf.var(0, unbiased=True).clamp_min(1e-12))
    var_ratio = var_ratio[torch.isfinite(var_ratio)]
    contrast_sigma = float(torch.sqrt(var_ratio.median().clamp_min(0)))  # ⚠️ 报 σ 比
    hf_real, hf_gen = hf_energy(real), hf_energy(gen)
    hf_ratio = hf_gen / max(hf_real, 1e-12)
    sh_real, sh_gen = sharpness(real), sharpness(gen)
    acf_r = acf_ratio(gen, real)

    # 跨图 L1（不同提示词 + 不同种子）—— 只作塌缩探测，**不是质量分**
    l1_gen = float(torch.cdist(gen.flatten(1), gen.flatten(1)).mean())
    # 塌缩 = 不同提示词出同一张图 ⇒ 特征方差≈0 且 图间距离≈0
    collapsed = (float(var_ratio.max()) < 1e-3) or (l1_gen < 3.0)

    # 条件注入比（可选）
    cond = None
    if args.cond_k > 0:
        print(f"\n-- 条件注入比（{args.cond_k} 条提示词 × 2 种噪声，多采 "
              f"{2 * args.cond_k} 张）--")
        dn, dp, ratio = cond_probe(model, tok, te, prompts, bucket, args.steps,
                                   args.cond_k,
                                   on_status=lambda m: print("   " + m, flush=True))
        cond = (ratio, dn, dp)

    # ---- 报告 -------------------------------------------------------------
    print("\n" + "=" * 74)
    print("结果")
    print("=" * 74)
    print(f"  生成 {n_gen} 张 · 真实池 {real.shape[0]} 张 · 特征 {Gf.shape[1]} 维")
    print(f"  对角 Fréchet 距离 F(gen, real) = {F_val:.4f}"
          f"   （均值项 {m_term:.4f} + 协方差项 {c_term:.4f}）")
    print(f"  噪声地板 F(realA,realB)      = {floor_n:.4f}"
          f"   （区间 {flo_n:.4f}~{fhi_n:.4f}，同样本量 {n_gen}）")
    print(f"  · score（低频⊕高频）= {score:.3f}"
          f"   ← 分布距离（1 = 不可区分，越低越好）。**只作旁证**：")
    print(f"     实测它给「灰糊碎块拼贴」也能打到 1.06，**分辨力不足**，别单独当质量分。")
    print(f"     ⚠️ score **只在同样的 --n 下可比**：n 越小，噪声地板越大、score 越小。"
          f"本次 n={n_gen}，历史 JSON 里 n 不同就别直接比。")
    print(f"     拆开看：低频(布局/色彩) {score_LF:.2f} · 高频(纹理/边缘) {score_HF:.2f}")
    print(f"  ⭐ 对比度σ σ_gen/σ_real 中位 = {contrast_sigma:.4f}"
          f"   ← 越接近 1 越好（欠训练会偏低，这是预期的）")
    print(f"     逐维方差比区间 {float(var_ratio.min()):.4f}~{float(var_ratio.max()):.4f}"
          f"  {'❌ 真塌缩（方差≈0）' if collapsed else '✅ 未塌缩'}")
    print(f"  ⭐ 高通能量比 gen/real = {hf_ratio:.4f}"
          f"   （gen {hf_gen:.4f} / real {hf_real:.4f}）← 越接近 1 越好；")
    print(f"     ≈0 = 糊成一片 · 明显 >1 = 残留噪声没去干净")
    print(f"  ⭐ 空间自相关比 gen/real = {acf_r:.4f}"
          f"   （lag {ACF_LAGS}）← 越接近 1 越好；")
    print(f"     **这一项能把「碎块拼贴」和「有结构的图」分开**，HF 做不到")
    print(f"     锐度（诊断用，**非质量分**）gen {sh_gen:.4f} / real {sh_real:.4f}"
          f"   ← 实测早期糊模型反而更高，别拿它排名")
    print(f"     跨图 L1（仅塌缩探测）= {l1_gen:.2f}")
    if cond:
        print(f"  ⭐ 条件注入比 = {cond[0]:.3f}   （跨提示词 {cond[1]:.3f} ÷ "
              f"跨噪声 {cond[2]:.3f}）← ≥1.3 说明文本条件生效；≈1 说明被忽略")
    print()

    # ---- 自检基准：真实自己的读数 = 本 n 下"可达的最好值" ------------------
    # n 小的时候连"真实子集 vs 真实池"都到不了理想 1.0，必须把这个基准打出来，
    # 否则会拿一个根本达不到的目标去要求模型。
    sc_pool = torch.cat([LF_r[:n_gen], HF_r[:n_gen]], 1)
    self_score, _, _ = val_estimate(sc_pool, Rf, n_gen)
    self_score /= max(floor_n, 1e-9)
    self_sig = float(torch.sqrt(
        (sc_pool.var(0, unbiased=True) / Rf.var(0, unbiased=True).clamp_min(1e-12))
        .median().clamp_min(0)))
    self_hf = hf_energy(real[:n_gen]) / max(hf_real, 1e-12)
    self_acf = acf_ratio(real[:n_gen], real)
    self_dev = abs(1 - self_sig) + abs(1 - self_acf) + abs(1 - self_hf)
    dev = abs(1 - contrast_sigma) + abs(1 - acf_r) + abs(1 - hf_ratio)
    print(f"  ⭐ 物理偏离度 = {dev:.4f}   （自检基准 {self_dev:.4f}）"
          f"   ← |1−对比度σ| + |1−自相关比| + |1−高频比|，**越低越好**")
    print(f"     自检 = 用前 {n_gen} 张**真实**潜向量算同一套 ⇒ 本 n 下能达到的最好值")

    # ---- 判读：先看硬失败，再看物理偏离度 --------------------------------
    # ⚠️ 不能拿 score 的固定阈值下结论：实测 step 9,058（**灰糊碎块拼贴**）的
    # score 只有 1.06，会得到"分布已相当接近真实"的**假满分**。
    # 有分辨力的是物理偏离度（实测 9,058 → 0.975 · 117,000 → 0.659）。
    if collapsed:
        verdict = "❌ **已塌缩**：不同提示词出同一张图 —— 训练已经坏了"
    elif dev <= self_dev * 1.3:
        verdict = "物理量已达/接近**本 n 下的自检基准**"
    elif dev <= self_dev * 2.0:
        verdict = "物理量离自检基准还有距离（欠训练的正常表现）"
    else:
        verdict = "物理量明显偏离自检基准 —— 严重欠训练或特征尺度不匹配"
    verdict += f" · 分布得分 score={score:.2f}（**只作旁证**，别单独用）"
    if cond and cond[0] < 1.3:
        verdict += " · ⚠️ **条件注入比 <1.3，文本条件可能被忽略**"

    print(f"  判读：{verdict}")
    print(f"  对比度：" + (
        "接近真实图像（结构完整）" if contrast_sigma > 0.7 else
        "偏低——输出比真实图像「平」，欠训练的典型表现；**它会随训练上升**"
        if contrast_sigma > 0.1 else
        "很低——输出严重偏平均，几乎是均值图"))
    print(f"  高通：" + (
        "与真实相当（纹理/边缘都在）" if 0.7 < hf_ratio < 1.4 else
        f"只有真实的 {100 * hf_ratio:.0f}% —— **糊**，模型还没学会高频细节"
        if hf_ratio <= 0.7 else
        f"是真实的 {hf_ratio:.2f} 倍 —— **残留噪声偏多**，去噪没走干净"))
    print("=" * 74)

    # ---- 落盘 -------------------------------------------------------------
    rec = dict(时间=datetime.now().isoformat(timespec="seconds"), 模式=args.mode,
               检查点=ckpt.name, 训练步数=step, 桶=f"{H}x{W}",
               生成样本=n_gen, 真实池=int(real.shape[0]),
               特征维=int(Gf.shape[1]), 采样步数=args.steps, 特征网格=args.grid,
               F=round(F_val, 6), 均值项=round(m_term, 6), 协方差项=round(c_term, 6),
               噪声地板=round(floor_n, 6), 地板区间=[round(flo_n, 6), round(fhi_n, 6)],
               **{"score": round(score, 6)},
               score_LF=round(score_LF, 6), score_HF=round(score_HF, 6),
               F_LF=round(F_LF, 6), F_HF=round(F_HF, 6),
               地板_LF=round(floor_LF, 6), 地板_HF=round(floor_HF, 6),
               对比度σ=round(contrast_sigma, 6),
               对比度=round(contrast_sigma, 6),          # 兼容旧趋势表，含义已改为 σ 比
               方差比区间=[round(float(var_ratio.min()), 6),
                       round(float(var_ratio.max()), 6)],
               高通能量比=round(hf_ratio, 6),
               高通能量_gen=round(hf_gen, 6), 高通能量_real=round(hf_real, 6),
               空间自相关比=round(acf_r, 6),
               自相关_gen=[round(v, 6) for v in acf(gen)],
               自相关_real=[round(v, 6) for v in acf(real)],
               # ⭐ 有分辨力的那个数：物理量离"1"的总距离；自检基准 = 本 n 下的可达最好值
               物理偏离度=round(dev, 6),
               物理偏离度_自检基准=round(self_dev, 6),
               自检_score=round(self_score, 6),
               锐度_gen=round(sh_gen, 6), 锐度_real=round(sh_real, 6),
               跨图L1=round(l1_gen, 4),
               条件注入比=(round(cond[0], 4) if cond else None),
               条件跨提示词=(round(cond[1], 4) if cond else None),
               条件跨噪声=(round(cond[2], 4) if cond else None),
               塌缩=collapsed, 判读=verdict)
    out = Path(args.json_out) if args.json_out else (
        OUT_ROOT / f"质量指标_s{step}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    lat_out = (Path(args.lat_out) if args.lat_out
               else out.with_suffix("").with_name(out.stem + ".latents.pt"))
    # ⚠️ **不要把 512 张真实潜向量一起存**：那是 42 MB，而这个文件每天都要
    # 进百度云归档（上传实测只有 5 MB/s）。真实池用固定种子就能几秒重算，不必存。
    torch.save({"latents": gen, "bucket": bucket, "step": step,
                "real_pool_n": int(real.shape[0]), "real_seed": REAL_SEED,
                # 提示词**连 image_id 一起存** —— aq_clip_score.py 靠它反查
                # 对应的真实 COCO 潜向量，才能算出「真实图上界」
                "prompts": [p for _i, p in prompts],
                "prompt_ids": [int(i) for i, _p in prompts]}, lat_out)
    print(f"\n已写出 {out}")
    print(f"   以及 {lat_out.name}（生成+真实潜向量，"
          f"{lat_out.stat().st_size / 1e6:.1f} MB）—— 以后重算指标不必重新采样")
    print("\n⚠️ 两条铁律：")
    print("   ① score 只证明「分布像不像」，**证明不了「图里有没有可辨认的物体」** ——")
    print("      那一条只能靠人看图（HANDOFF §0 的合格判据）。两者缺一不可。")
    print("   ② 本指标曾**奖励过灰糊模型**（2026-10-03）。所以它下降只能当"
          "「没有变坏」的旁证，")
    print("      真正的「有没有变好」还是要配图看。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
