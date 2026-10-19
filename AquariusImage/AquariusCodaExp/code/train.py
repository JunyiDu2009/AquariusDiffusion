"""AquariusCodaExp M1 冒烟训练脚本（val2017 / 像素空间 / 一步去噪器）。

对齐计划书 v1.2：
- §3.1 加噪：x_p = √(1-p)·x0 + √p·ε，p 为噪声方差占比，逐图像采样
- §4.3 损失：MSE + 0.1×LPIPS（VGG16，不可用时自动退回纯 MSE）
- §7.1 超参：AdamW(1e-4, wd=0.01)、cosine+warmup500、grad clip 1.0、EMA 0.999
- §8.2 每 save_every 步导出 NVFP4（E2M1+per-16 E4M3+FP32 tensor scale），
  同时保留 FP16 训练态 ckpt（NVFP4 权重不可续训，双存是设计使然）
- 冒烟口径说明：val2017 同时充当训练与评估集，指标仅作管线 sanity check，
  不可当作验收值；正式训练必须切 train2017（计划书 §2.1）。

指标输出 runs/<name>/metrics.jsonl，每步一行；前端 dashboard.py 每 2s 拉取。
"""
import argparse
import json
import math
import os
import random
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from model import UNetDenoiser
from nvfp4 import export_model_nvfp4, dequantize_model

P_GRID = [0.01, 0.10, 0.20, 0.30, 0.40, 0.50]  # 覆盖全部 5 个阶梯 cap，便于观察各档能力成长
PLATEAU_EPS = 0.003  # LPIPS@cap 改善小于该值视为平台期


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/val2017")
    ap.add_argument("--out", default="runs/m1_smoke")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--stage", type=int, default=0, help="起始课程阶梯 0-4，噪声上限 cap=(stage+1)*10%%")
    ap.add_argument("--min-stage-steps", type=int, default=2000, help="每阶梯最少步数，防止过早升阶")
    ap.add_argument("--plateau-patience", type=int, default=3, help="连续 N 次评估 LPIPS@10%% 无改善则升阶")
    ap.add_argument("--eval-data", default=None, help="评估集目录（默认与 --data 相同；正式训练用 val2017 留出）")
    ap.add_argument("--ckpt-dir", default=None, help="checkpoint 存储目录（默认 <out>/ckpts；E 盘紧张时指向 D 盘临时文件夹）")
    ap.add_argument("--resume", default=None, help="last_fp16.pt 路径，从该 checkpoint 续训")
    ap.add_argument("--max-stage-steps", type=int, default=25000, help="每阶梯步数上限，防止指标震荡导致卡死不升阶")
    ap.add_argument("--early-stop", action="store_true", help="最高阶平台期早停：10/30/50%% 三档 LPIPS 均值连续 es-patience 次评估无改善即存档退出")
    ap.add_argument("--es-patience", type=int, default=20, help="早停耐心：连续多少次评估无显著改善（eval-every=500 时 20 次 = 1 万步）")
    ap.add_argument("--es-eps", type=float, default=0.002, help="早停改善判据：门控低于历史最佳超过该值才算改善")
    ap.add_argument("--es-min-steps", type=int, default=60000, help="早停生效的最小步数下限")
    ap.add_argument("--multi-res", action="store_true", help="多分辨率×多宽高比桶（256²~640×384，含 16:9）：每步按权重选一桶，batch 随像素数缩放；推理域对齐任意 /16 尺寸")
    ap.add_argument("--mem-frac", type=float, default=None, help="torch 显存上限（占总显存比例，如 0.55）；WDDM 下防 reserved 膨胀挤爆物理显存")
    ap.add_argument("--lpips-weight", type=float, default=0.1)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-images", type=int, default=256)
    ap.add_argument("--sample-every", type=int, default=200, help="每 N 步保存一张「原图→加噪→输出」样本条带到前端")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument(
        "--benchmark",
        action="store_true",
        help="启用 cudnn.benchmark（本机 autotune 极慢，16 分钟未完成；默认关闭，用启发式算法）",
    )
    return ap.parse_args()


def list_images(data_dir: str):
    files = sorted(
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not files:
        raise SystemExit(f"no images found in {data_dir}")
    return files


def load_image(path: str, size, train: bool, rng: random.Random) -> torch.Tensor:
    """size=(w,h)：按「覆盖」缩放（短边贴齐目标），再随机/中心裁剪 (w,h)。
    宽高只需为 16 的倍数（模型 4 次下采样），任意宽高比均可训练——与 SD/DiT 的
    aspect-ratio bucket 训练同一套口径：裁剪即增广（平移等变），缩放即尺度归一。"""
    w_t, h_t = size
    img = Image.open(path).convert("RGB")
    w0, h0 = img.size
    scale = max(w_t / w0, h_t / h0)
    if scale != 1.0:
        img = img.resize((max(w_t, round(w0 * scale)), max(h_t, round(h0 * scale))), Image.BICUBIC)
    w, h = img.size
    if train:
        x0 = rng.randint(0, w - w_t)
        y0 = rng.randint(0, h - h_t)
    else:
        x0 = (w - w_t) // 2
        y0 = (h - h_t) // 2
    img = img.crop((x0, y0, x0 + w_t, y0 + h_t))
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def load_batch(paths, size, train, rng, pool, fallback_files=None):
    def _safe_load(arg):
        _, p = arg
        last_err = None
        for _ in range(4):
            try:
                return load_image(p, size, train, rng)
            except Exception as e:
                # 瞬态文件锁（杀软扫描/网盘同步）不该炸掉整场训练：换一张图重试
                last_err = e
                if fallback_files:
                    p = fallback_files[rng.randrange(len(fallback_files))]
        print(f"[warn] load failed 4x, gray fallback: {p} ({type(last_err).__name__}: {last_err})", flush=True)
        return torch.zeros(3, size[1], size[0])

    tensors = list(pool.map(_safe_load, enumerate(paths)))
    return torch.stack(tensors)


def add_noise(x0: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    eps = torch.randn_like(x0)
    pf = p[:, None, None, None]
    return torch.sqrt(1 - pf) * x0 + torch.sqrt(pf) * eps


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred.float(), target.float(), reduction="mean").item()
    if mse <= 0:
        return 99.0
    return 10.0 * math.log10(4.0 / mse)


def make_sample_strip(x0: torch.Tensor, xp: torch.Tensor, pred: torch.Tensor, p_val: float) -> Image.Image:
    """三联条带：原图 | 加噪图（标 p）| 模型当前输出。张量 shape (3,H,W)，[-1,1]。"""
    panels = []
    for t in (x0, xp, pred):
        arr = ((t.detach().float().clamp(-1, 1) + 1) * 127.5).byte().cpu().numpy().transpose(1, 2, 0)
        panels.append(Image.fromarray(arr))
    w, h = panels[0].size
    sep = 4
    strip = Image.new("RGB", (w * 3 + sep * 2, h + 18), (20, 20, 22))
    labels = ["input", f"noisy p={p_val:.0f}%", "model output"]
    for i, (im, lb) in enumerate(zip(panels, labels)):
        strip.paste(im, (i * (w + sep), 0))
        d = ImageDraw.Draw(strip)
        d.text((i * (w + sep) + 4, h + 3), lb, fill=(200, 198, 190))
    return strip


@torch.no_grad()
def run_eval(model, base: torch.Tensor, device, lpips_fn, bs=8):
    """在预加载的评估张量上按 P_GRID 六档噪声评估 PSNR/LPIPS。"""
    model.eval()
    results = {}
    n = base.shape[0]
    for p in P_GRID:
        ps, ls = [], []
        for s in range(0, n, bs):
            x0 = base[s : s + bs].to(device)
            pv = torch.full((x0.shape[0],), p, device=device)
            g = torch.Generator(device="cpu").manual_seed(1000 + int(p * 1000))
            eps = torch.randn(x0.shape, generator=g).to(device)
            pf = pv[:, None, None, None]
            xp = torch.sqrt(1 - pf) * x0 + torch.sqrt(pf) * eps
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(xp, pv)
            ps.append(psnr(pred, x0))
            if lpips_fn is not None:
                ls.append(lpips_fn(pred.float(), x0).mean().item())
        results[f"psnr@{int(p*100)}%"] = round(float(np.mean(ps)), 3)
        if ls:
            results[f"lpips@{int(p*100)}%"] = round(float(np.mean(ls)), 4)
    model.train()
    return results


_EVAL_POOL = ThreadPoolExecutor(max_workers=4)


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    ckpt_dir = args.ckpt_dir if args.ckpt_dir else os.path.join(args.out, "ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)
    samples_dir = os.path.join(args.out, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    metrics_path = os.path.join(args.out, "metrics.jsonl")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = args.benchmark
    device = torch.device("cuda")
    if args.mem_frac:
        torch.cuda.set_per_process_memory_fraction(args.mem_frac, 0)
        print(f"[mem] allocator hard-capped at {args.mem_frac:.0%} of total VRAM", flush=True)

    files = list_images(args.data)
    print(f"[data] {len(files)} images from {args.data}", flush=True)
    eval_files = list_images(args.eval_data) if args.eval_data else files
    if args.eval_data:
        print(f"[data] eval on {len(eval_files)} images from {args.eval_data} (held out)", flush=True)

    # 多分辨率×多宽高比桶（--multi-res）：宽高均为 16 的倍数（模型 4 次下采样）。
    # 覆盖 1:1 / 3:2 / 4:3 / 5:3 / 16:9 及其竖版——推理要吃任意尺寸（用户拍板），
    # 纯方形训练在非方形输入上统计失配（感受野占比/注意力熵漂移），必须训练域直接覆盖。
    # batch 按像素预算缩放：预算 = args.batch×256²，batch=预算/(w×h)（1~args.batch 截断），
    # 保证每步总像素量≈恒定 → 显存恒定（实测 ~4.3GB 峰值）。
    MR_BUCKETS = [  # (w, h, weight)
        (256, 256, 2.0),
        (384, 256, 1.0), (256, 384, 1.0),
        (384, 384, 1.0),
        (512, 384, 1.0), (384, 512, 1.0),
        (512, 512, 1.5),
        (640, 384, 0.75), (384, 640, 0.75),
        (512, 288, 1.0), (288, 512, 1.0),  # ≈16:9 / 9:16（接力视频帧方向）
    ]
    if args.multi_res:
        _B = args.batch * 256 * 256
        mr_items = []
        for (mw, mh, mwt) in MR_BUCKETS:
            mb = max(1, min(args.batch, round(_B / (mw * mh))))
            mr_items.append([mw, mh, mb, mwt])
        _tot = sum(it[3] for it in mr_items)
        mr_cdf, _c = [], 0.0
        for it in mr_items:
            it[3] /= _tot
            _c += it[3]
            mr_cdf.append(_c)
        print("[data] aspect buckets: " + ", ".join(f"{w}x{h} x{b} (p={wt:.2f})" for (w, h, b, wt) in mr_items), flush=True)
    else:
        mr_items = None

    # 课程阶梯状态：cap=(stage+1)*10，每图随机整数百分比 pct∈[1,cap]
    stage = max(0, min(4, args.stage))
    cap_pct = (stage + 1) * 10
    stage_start_step = 0
    stage_best = None      # 本阶梯内最优 LPIPS@cap（升阶判据）
    stage_best_l1 = None   # 本阶梯内最优 LPIPS@1%（防遗忘哨兵）
    plateau = 0

    model = UNetDenoiser().to(device)
    model.enable_gradient_checkpointing()
    n_params = model.num_params()
    print(f"[model] params: {n_params/1e6:.1f}M, gradient checkpointing ON (8GB 显存约束)", flush=True)

    lpips_fn = None
    if args.lpips_weight > 0:
        try:
            import lpips

            lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()
            for q in lpips_fn.parameters():
                q.requires_grad_(False)
            print("[loss] LPIPS(VGG16) enabled, weight 0.1", flush=True)
        except Exception as e:
            print(f"[loss] LPIPS unavailable ({type(e).__name__}: {e}); fallback to MSE-only", flush=True)
            lpips_fn = None

    ema_state = {k: v.detach().float().clone() for k, v in model.state_dict().items()}
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)

    start_step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        ema_state = {k: v.to(device).float() for k, v in ck["ema"].items()}
        opt.load_state_dict(ck["opt"])
        start_step = int(ck.get("step", 0))
        stage = max(0, min(4, int(ck.get("stage", 0))))
        cap_pct = (stage + 1) * 10
        stage_start_step = start_step
        stage_best = None
        stage_best_l1 = None
        plateau = 0
        print(f"[resume] step {start_step}, stage {stage} (cap {cap_pct}%) from {args.resume}", flush=True)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        t = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))

    rng = random.Random(args.seed)
    pool = ThreadPoolExecutor(max_workers=args.workers)
    mfile = open(metrics_path, "a", encoding="utf-8")

    # 评估张量只加载一次（128 图 × 3×256×256 fp32 ≈ 100MB 内存），六档噪声共用
    print(f"[eval] preloading {min(args.eval_images, len(eval_files))} eval images...", flush=True)
    eval_base = torch.stack(
        list(pool.map(lambda f: load_image(f, (args.res, args.res), False, rng), eval_files[: args.eval_images]))
    )

    # 预热：cudnn.benchmark 会对每个新卷积形状做算法 autotune（本机约数分钟），
    # 必须在计时开始前用同一 batch/res 跑一次前向+反向把全部形状烤熟。
    # 多分辨率模式下三种桶形状各预热一次——也把 512² 的 OOM 风险挡在启动期而不是跑到一半炸
    print(f"[warmup] forward+backward per shape (cudnn.benchmark={'on' if args.benchmark else 'off'})...", flush=True)
    model.train()
    warm_shapes = [(w, h, b) for (w, h, b, _wt) in mr_items] if args.multi_res else [(args.res, args.res, args.batch)]
    for w_w, h_w, bs_w in warm_shapes:
        xw = torch.randn(bs_w, 3, w_w, h_w, device=device) * 0.5
        pw = torch.tensor([rng.randint(1, cap_pct) / 100.0 for _ in range(bs_w)], device=device)
        xwn = add_noise(xw, pw)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outw = model(xwn, pw)
            lw = F.mse_loss(outw.float(), xw)
        opt.zero_grad(set_to_none=True)
        lw.backward()
        opt.step()
        print(f"[warmup] {w_w}x{h_w} x{bs_w} ok, peak VRAM {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)
        del xw, xwn, outw, lw
    torch.cuda.empty_cache()

    def log_metric(obj):
        mfile.write(json.dumps(obj, ensure_ascii=False) + "\n")
        mfile.flush()

    def save_ckpts(step, ema_state):
        # 原子写入：先写 .tmp 再 os.replace，避免并发读取（demo/面板）撞上半截文件
        tmp = os.path.join(ckpt_dir, "last_fp16.pt.tmp")
        torch.save(
            {"step": step, "stage": stage, "model": model.state_dict(), "ema": ema_state, "opt": opt.state_dict(), "args": vars(args)},
            tmp,
        )
        os.replace(tmp, os.path.join(ckpt_dir, "last_fp16.pt"))
        pkg = export_model_nvfp4({k: v for k, v in ema_state.items()})
        path = os.path.join(ckpt_dir, f"step_{step:06d}_nvfp4.pt")
        tmp2 = path + ".tmp"
        torch.save(pkg, tmp2)
        os.replace(tmp2, path)
        size_mb = os.path.getsize(path) / 1e6
        sd = dequantize_model(pkg)
        probe = UNetDenoiser().to(device)
        probe.load_state_dict(sd)
        gap = run_eval(probe, eval_base, device, lpips_fn, args.batch)
        del probe
        torch.cuda.empty_cache()
        print(f"[save] step {step}: {os.path.basename(path)} {size_mb:.1f}MB, nvfp4 params {pkg['stats']['nvfp4_params']/1e6:.1f}M, gap {gap}", flush=True)
        return {"type": "save", "step": step, "file": os.path.basename(path), "size_mb": round(size_mb, 1), "nvfp4_params": pkg["stats"]["nvfp4_params"], "gap": gap}

    stop = {"flag": False}
    es_best = None  # 早停门控最佳值（Stage 4 验收三档 LPIPS 均值）
    es_bad = 0      # 早停无显著改善的评估计数

    def on_sigint(sig, frame):
        print("\n[sigint] finishing current step then saving...", flush=True)
        stop["flag"] = True

    signal.signal(signal.SIGINT, on_sigint)

    t0 = time.time()
    step_times = []
    nan_count = 0
    model.train()
    for step in range(start_step, args.steps):
        ts = time.time()
        for g in opt.param_groups:
            g["lr"] = lr_at(step)

        # 多分辨率×宽高比桶：按权重选一桶，本步 batch 全部同一 (w,h)（同批必须同尺寸）
        if args.multi_res:
            _r, bi = rng.random(), 0
            for i, c in enumerate(mr_cdf):
                if _r < c:
                    bi = i
                    break
            wb, hb, bs_b = mr_items[bi][0], mr_items[bi][1], mr_items[bi][2]
        else:
            wb, hb, bs_b = args.res, args.res, args.batch
        paths = [files[rng.randrange(len(files))] for _ in range(bs_b)]
        x0 = load_batch(paths, (wb, hb), True, rng, pool, fallback_files=files).to(device, non_blocking=True)
        # 用户规则：每张图独立随机整数百分比加噪，pct ∈ [1, cap]%（Stage 0 = 1%~10%）
        pcts = [rng.randint(1, cap_pct) for _ in range(bs_b)]
        p = torch.tensor([c / 100.0 for c in pcts], device=device)
        xp = add_noise(x0, p)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(xp, p)
            mse = F.mse_loss(pred.float(), x0)
            lp = lpips_fn(pred.float(), x0).mean() if lpips_fn is not None else torch.zeros((), device=device)
            loss = mse + args.lpips_weight * lp

        if not torch.isfinite(loss):
            nan_count += 1
            print(f"[warn] non-finite loss at step {step} ({nan_count}/3)", flush=True)
            if nan_count >= 3:
                raise SystemExit("aborted: 3 consecutive non-finite losses")
            continue

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        with torch.no_grad():
            d = args.ema
            msd = model.state_dict()
            for k, v in ema_state.items():
                v.mul_(d).add_(msd[k].float(), alpha=1 - d)

        dt = time.time() - ts
        step_times.append(dt)
        step_times = step_times[-20:]
        sps = 1.0 / (sum(step_times) / len(step_times))
        elapsed = time.time() - t0
        eta = (args.steps - step - 1) / sps

        # 样本条带：原图 → 加噪（标 p）→ 模型当前输出，供前端实况面板展示
        if args.sample_every > 0 and step % args.sample_every == 0:
            strip = make_sample_strip(x0[0], xp[0], pred[0], pcts[0])
            tmp_path = os.path.join(samples_dir, "latest.tmp.png")
            strip.save(tmp_path)
            os.replace(tmp_path, os.path.join(samples_dir, "latest.png"))
            log_metric(
                {
                    "type": "sample",
                    "step": step,
                    "stage": stage,
                    "src": os.path.basename(paths[0]),
                    "p": pcts[0],
                    "res": f"{wb}x{hb}",
                    "psnr": round(psnr(pred[0:1], x0[0:1]), 2),
                }
            )

        log_metric(
            {
                "type": "step",
                "step": step,
                "total_steps": args.steps,
                "stage": stage,
                "cap_pct": cap_pct,
                "res": f"{wb}x{hb}",
                "loss": round(loss.item(), 5),
                "mse": round(mse.item(), 6),
                "lpips": round(lp.item(), 5),
                "lr": round(lr_at(step), 8),
                "p_mean": round(float(p.mean()), 4),
                "sps": round(sps, 3),
                "gpu_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
                "elapsed": round(elapsed, 1),
                "eta": round(eta, 1),
            }
        )

        if step % 20 == 0:
            print(
                f"step {step:5d} | loss {loss.item():.4f} (mse {mse.item():.5f} lpips {lp.item():.4f}) "
                f"| lr {lr_at(step):.2e} | {sps:.2f} it/s | eta {eta/60:.1f} min",
                flush=True,
            )

        if (step + 1) % args.eval_every == 0 or step == 0:
            ev = run_eval(model, eval_base, device, lpips_fn, args.batch)
            log_metric({"type": "eval", "step": step, "stage": stage, **ev})
            # 升阶判据：LPIPS@cap（本阶梯正在教的能力上限）平台期
            gate_key = f"lpips@{cap_pct}%"
            gate_val = ev.get(gate_key)
            improved = gate_val is not None and (stage_best is None or gate_val < stage_best - PLATEAU_EPS)
            if improved:
                stage_best = gate_val
                plateau = 0
            else:
                plateau += 1
            # 防遗忘哨兵：低噪声档（1%）能力在本阶梯内退化超 0.02 则告警（不阻断）
            l1 = ev.get("lpips@1%")
            if l1 is not None:
                if stage_best_l1 is None or l1 < stage_best_l1:
                    stage_best_l1 = l1
                elif l1 > stage_best_l1 + 0.02:
                    print(f"[warn] forgetting guard: lpips@1% {l1:.4f} vs stage best {stage_best_l1:.4f}", flush=True)
            steps_in_stage = step + 1 - stage_start_step
            reason = None
            if stage < 4:
                if steps_in_stage >= args.max_stage_steps:
                    reason = f"max {args.max_stage_steps} steps"
                elif plateau >= args.plateau_patience and steps_in_stage >= args.min_stage_steps:
                    reason = f"plateau x{plateau} (eps {PLATEAU_EPS})"
            if reason:
                stage += 1
                cap_pct = (stage + 1) * 10
                stage_start_step = step + 1
                stage_best = None
                stage_best_l1 = None
                plateau = 0
                print(f"[stage] -> stage {stage} (cap {cap_pct}%) at step {step+1} [{reason}]", flush=True)
                log_metric({"type": "stage", "step": step + 1, "stage": stage, "cap_pct": cap_pct, "reason": reason})
            # 平台期早停（仅最高阶 Stage 4 且开 --early-stop）：验收三档 LPIPS 均值无实质改善 → 存档退出。
            # 与升阶 plateau 的区别：门控是三档验收均值（对应 §10 验收口径），eps 更小、耐心更长，避免误停。
            if args.early_stop and stage == 4 and step + 1 >= args.es_min_steps:
                gv = [ev.get(k) for k in ("lpips@10%", "lpips@30%", "lpips@50%")]
                if all(v is not None for v in gv):
                    es_gate = sum(gv) / 3.0
                    if es_best is None or es_gate < es_best - args.es_eps:
                        es_best = es_gate
                        es_bad = 0
                    else:
                        es_bad += 1
                        if es_bad % 5 == 0:
                            print(f"[early-stop] no-improve {es_bad}/{args.es_patience} (gate {es_gate:.4f} vs best {es_best:.4f})", flush=True)
                        if es_bad >= args.es_patience:
                            print(f"[early-stop] gate plateau x{es_bad} (eps {args.es_eps}) at step {step+1}, best {es_best:.4f} -> save & exit", flush=True)
                            log_metric({"type": "early_stop", "step": step + 1, "gate": round(es_gate, 5), "best": round(es_best, 5)})
                            rec = save_ckpts(step + 1, ema_state)
                            log_metric(rec)
                            break
            print(f"[eval] step {step}: stage {stage} (cap {cap_pct}%, plateau {plateau}): {ev}", flush=True)
            # 多形状桶训练会让缓存分配器碎片化、reserved 显存只增不减（WDDM 下溢出到共享内存
            # 会 10 倍减速）——每次评估后把未用缓存还给系统，把 reserved 压在活跃集附近
            torch.cuda.empty_cache()

        if (step + 1) % args.save_every == 0:
            rec = save_ckpts(step + 1, ema_state)
            log_metric(rec)

        if stop["flag"]:
            rec = save_ckpts(step + 1, ema_state)
            log_metric(rec)
            print("[exit] saved and stopped by user", flush=True)
            break

    mfile.close()
    print(f"[done] total {time.time()-t0:.0f}s, metrics at {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
