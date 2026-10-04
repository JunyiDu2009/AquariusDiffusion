# -*- coding: utf-8 -*-
"""定时任务用：检查训练 → 把最新**稳定**检查点导出成 ternary 2-bit → 投放百度云。

为什么要有这个脚本，而不是让定时任务直接跑 aq_pack.py：
  * 选源要小心 —— **绝不能用 latest.pt**（训练边跑边非原子重写它），
    必须取编号快照 ckpt_NNNNNN.pt，且要等它"尺寸稳定"才算写完
  * 幂等 —— 同一步已经导出过就跳过，不做无意义的重复劳动
  * 保留策略 —— 只留最新 N 份，更早的**移出同步树**（不删除），控制云上体积
  * **零 GPU 接触** —— 强制 CUDA_VISIBLE_DEVICES=""，连设备都看不见

用法：
    python code/aq_export_latest.py                # 正常跑
    python code/aq_export_latest.py --dry-run      # 只看会做什么
    python code/aq_export_latest.py --keep 2       # 只留最新 2 份
    python code/aq_export_latest.py --force        # 同一步也重新导出
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "code"))
import aq_paths                                                  # noqa: E402

CKPT = WS / "checkpoints" / "ternary"
TRAIN_LOG = WS / "Aquarius_cloud" / "aq_train.log"
WD_LOG = WS / "Aquarius_cloud" / "aq_watchdog.log"
EXPORT_LOG = WS / "Aquarius_cloud" / "aq_export.log"
PACK = WS / "code" / "aq_pack.py"

# 目录与解释器都**自动探测**（本机用户名 ajifang，目标机是 Du）
SYNC_DIR = aq_paths.sync_root()
# ⚠️ 走 aq_paths 的归档中枢，别再自己拼路径（旧写法会在同步根下重建 Aquarius_models/）
SYNC_DIR = aq_paths.models_dir()
# 被淘汰的旧导出**移到这里**（在同步树之外），不删除 —— 云上那份会随同步消失，
# 但百度云回收站还留约 30 天，需要时能捞回来。
SUPERSEDED = aq_paths.superseded_dir()
PY = aq_paths.interpreter()


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    EXPORT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(EXPORT_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# --------------------------------------------------------------------------
# 1. 检查训练状况
# --------------------------------------------------------------------------
def check_training():
    """返回 dict：current / total / loss / s_it / eta_h / alive / done。"""
    info = dict(current=None, total=None, loss=None, s_it=None, eta_h=None,
                alive=False, done=False, note="")
    try:
        lines = TRAIN_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as e:
        info["note"] = f"读不到训练日志：{e}"
        return info

    # 只认最后一条 [plan] total_steps 之后的进度行 —— 日志跨多轮追加，
    # 早前的行用的是旧的 total（例如 90580），不加这层过滤会读串
    start = 0
    for i, ln in enumerate(lines):
        if "[plan] total_steps=" in ln:
            start = i
    pat = re.compile(r"^step (\d+)/(\d+) loss ([\d.]+).*?([\d.]+)s/it .*?eta ([\d.]+)h")
    for ln in lines[start:]:
        m = pat.search(ln)
        if m:
            info.update(current=int(m.group(1)), total=int(m.group(2)),
                        loss=float(m.group(3)), s_it=float(m.group(4)),
                        eta_h=float(m.group(5)))
    info["done"] = any("[ckpt] FINAL saved" in ln for ln in lines[start:])
    # 进程还活着吗（用日志 mtime 而不是进程列表：跨机器也适用）
    try:
        age = time.time() - TRAIN_LOG.stat().st_mtime
        info["alive"] = age < 3600
        info["note"] = f"日志 {age / 60:.1f} 分钟前更新"
    except OSError:
        pass
    return info


# --------------------------------------------------------------------------
# 2. 选源：最新**稳定**的编号快照
# --------------------------------------------------------------------------
def newest_stable_ckpt(settle=8.0, wait_max=180):
    """挑步骤号最大的 ckpt_NNNNNN.pt，并确认它已写完（尺寸连续 settle 秒不变）。

    **不要用 latest.pt** —— `save_checkpoint` 是直接覆盖写、非原子，
    训练每 1000 步重写它一次，读到半个文件会得到损坏的权重。
    编号快照只写一次，写完就不再动，唯一风险是被轮转删掉（--keep-ckpts 4，
    即大约 4000 步之后才会删），所以"挑最新的"偶尔会落空 —— 那就退一位。
    """
    def numbered():
        out = []
        for p in CKPT.glob("ckpt_*.pt"):
            m = re.search(r"ckpt_(\d+)\.pt$", p.name)
            if m:
                out.append((int(m.group(1)), p))
        return sorted(out, reverse=True)

    for attempts in range(3):
        cands = numbered()
        if not cands:
            return None, None, "没有编号快照"
        step, p = cands[min(attempts, len(cands) - 1)]
        # 等尺寸稳定
        t0, last, stable_since = time.time(), -1, time.time()
        while time.time() - t0 < wait_max:
            try:
                sz = p.stat().st_size
            except OSError:
                break                     # 正在被删/被替换
            if sz != last:
                last, stable_since = sz, time.time()
            elif time.time() - stable_since >= settle:
                return step, p, f"{sz / 1e9:.3f} GB，尺寸稳定 {settle:.0f}s"
            time.sleep(1.0)
        # 没稳定下来（或文件消失），退到下一个候选
        time.sleep(2)
    return None, None, "候选择查超时"


# --------------------------------------------------------------------------
# 3. 导出（CPU-only，零 GPU 接触）
# --------------------------------------------------------------------------
def export(step, src, dry=False):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""          # ← 要求 3：不侵占 GPU 显存
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    cmd = [str(PY), "-u", str(PACK), "--mode", "ternary",
           "--ckpt", str(src), "--no-sync"]   # 同步由本脚本统一做，便于记账
    log(f"  打包命令 CUDA_VISIBLE_DEVICES=''  {' '.join(cmd[3:])}")
    if dry:
        return None, "dry-run"
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(WS), env=env,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    out_txt = (r.stdout or "") + (r.stderr or "")
    tail = [l for l in out_txt.splitlines()
            if any(k in l for k in ("磁盘体积", "bit/权重", "回读校验", "码位无损",
                                    "加载器", "预测体积", "abort", "Error", "Traceback"))]
    for l in tail:
        log("    " + l.strip())
    if r.returncode != 0:
        return None, f"打包失败 rc={r.returncode}"
    m = re.search(r"([\w\-]+\.safetensors)", out_txt)
    name = f"aquarius_ternary_step{step}.safetensors"
    path = WS / "out" / name
    if not path.is_file():
        return None, f"打包说成功但找不到产物 {path}"
    log(f"  打包完成，用时 {time.time() - t0:.0f}s -> {path.name}")
    return path, "ok"


# --------------------------------------------------------------------------
# 4. 投放 + 保留策略
# --------------------------------------------------------------------------
def publish(local, dry=False):
    if SYNC_DIR is None:
        log("  ⚠️ 未找到百度云同步目录，跳过投放"
            "（可用环境变量 AQ_SYNC_DIR 显式指定）")
        return None
    SYNC_DIR.mkdir(parents=True, exist_ok=True)
    dst = SYNC_DIR / local.name
    if dry:
        return dst
    shutil.copy2(local, dst)
    log(f"  ☁️ 已投放 {dst}（{dst.stat().st_size / 1e6:.1f} MB）")
    return dst


def prune(keep, dry=False):
    """只留最新 keep 份导出；更早的**移出同步树**（不删除）。"""
    if SYNC_DIR is None:
        return []
    files = sorted(SYNC_DIR.glob("aquarius_ternary_step*.safetensors"),
                   key=lambda p: int(re.search(r"step(\d+)", p.name).group(1)),
                   reverse=True)
    old = files[keep:]
    if not old:
        return []
    if not dry:
        SUPERSEDED.mkdir(parents=True, exist_ok=True)
    moved = []
    for p in old:
        log(f"  ↪️ 移出同步树（不删除）：{p.name} -> {SUPERSEDED}")
        if not dry:
            try:
                shutil.move(str(p), str(SUPERSEDED / p.name))
                moved.append(p.name)
            except OSError as e:
                log(f"    [warn] 移动失败 {p.name}: {e}")
        else:
            moved.append(p.name)
    if moved:
        log(f"  保留最新 {keep} 份；云上这 {len(moved)} 份会随同步移除"
            f"（百度云回收站仍留约 30 天）")
    return moved


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="检查训练 + 导出最新检查点为 ternary 2-bit")
    ap.add_argument("--keep", type=int, default=3, help="投放目录里保留最新几份（默认 3）")
    ap.add_argument("--force", action="store_true", help="同一步也重新导出")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    log("=" * 68)
    log("导出任务启动")

    # ---- 1. 训练状况 --------------------------------------------------
    t = check_training()
    if t["current"] is None:
        log(f"  ⚠️ 读不到训练进度（{t['note']}）—— 仍然尝试导出最新快照")
    else:
        pct = 100 * t["current"] / t["total"]
        log(f"  训练：step {t['current']:,}/{t['total']:,}（{pct:.1f}%）"
            f" loss {t['loss']} {t['s_it']}s/it eta {t['eta_h']}h")
        log(f"  进度来源：{t['note']}；进程{'在跑' if t['alive'] else '**可能已停**'}")
        if t["done"]:
            log("  训练已跑满目标步数（见到 FINAL saved）")
    if not t["alive"]:
        log("  ⚠️ 日志超过 1 小时没更新 —— 先看 aq_watchdog.log 是否已重启训练")

    # ---- 2. 选源 ------------------------------------------------------
    step, src, why = newest_stable_ckpt()
    if step is None:
        log(f"  ❌ 没有可用的稳定检查点：{why}")
        return 1
    # ⚠️ 大小稳定 ≠ 内容完整。2026-10-03 夜里出现**反复的瞬时写盘失败**
    # （`PytorchStreamWriter failed writing file data/NNN`），会在目录里留下
    # 半截的 ckpt_NNNNNN.pt；而"编号最大"恰恰会挑中它。
    # 所以挑完还要**结构性验证**（mmap 读顶层，约 0.2s），坏了就退到下一个。
    ok, why2 = aq_paths.ckpt_ok(src)
    if not ok:
        log(f"  ⚠️ 最新快照不可用（{why2}）—— 退到下一个好的")
        alt, why3 = aq_paths.newest_good_ckpt(CKPT)
        if alt is None:
            log(f"  ❌ 也没有可用快照：{why3}")
            return 1
        step, src, why = int(re.search(r"ckpt_(\d+)", alt.name).group(1)), alt, why3
    log(f"  源：{src.name}（step {step:,}，{why}）")
    log(f"  校验：{why2 if ok else '已跳过坏文件'}")

    if t["current"] is not None and abs(t["current"] - step) > 20_000:
        log(f"  ⚠️ 快照 step {step:,} 与日志 step {t['current']:,} 相差超过 2 万，"
            f"确认是否轮转删过头")

    # ---- 3. 幂等 ------------------------------------------------------
    target = SYNC_DIR / f"aquarius_ternary_step{step}.safetensors"
    if target.is_file() and not args.force:
        log(f"  ✅ step {step:,} 已经投放过了（{target.name}），跳过导出。"
            f"（要重导加 --force）")
        prune(args.keep, dry=args.dry_run)
        return 0

    # ---- 4. 导出 ------------------------------------------------------
    path, why = export(step, src, dry=args.dry_run)
    if path is None and why != "dry-run":
        log(f"  ❌ {why}")
        return 1

    # ---- 5. 投放 + 保留 -----------------------------------------------
    if args.dry_run:
        log(f"  [dry-run] 会投放 {target.name}；随后保留最新 {args.keep} 份")
        prune(args.keep, dry=True)
        log("dry-run 结束")
        return 0
    publish(path, dry=False)
    prune(args.keep, dry=False)
    log(f"完成。产物 {path}（{path.stat().st_size / 1e6:.1f} MB）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
