# -*- coding: utf-8 -*-
"""跨机器可用的路径解析 —— 避免把本机绝对路径写死到脚本里。

为什么需要：本机（租用 5090）是 `C:\\Users\\ajifang\\...`，
目标机（自有 5060）的用户名是 `Du`。任何写死的 `C:\\Users\\ajifang` 到了那边
要么失败、要么**在错误的盘上建出一棵新目录树**（后者更危险）。

解析优先级：
  1. 环境变量 `AQ_SYNC_DIR`（最高，显式覆盖）
  2. `%USERPROFILE%\\Desktop\\存档\\BaiduSyncdisk`
  3. 历史硬编码路径（本机就在这儿）
  4. 都找不到 → 返回 None，调用方**跳过同步并说明原因**，绝不乱建目录
"""
import os
from pathlib import Path

# 主目录下的相对位置（两台机器一致，只有用户名不同）
REL = Path("Desktop") / "存档" / "BaiduSyncdisk"
LEGACY = Path(r"C:\Users\ajifang\Desktop\存档\BaiduSyncdisk")


def sync_root():
    """返回百度云同步目录的 Path；找不到返回 None。"""
    env = os.environ.get("AQ_SYNC_DIR")
    if env:
        p = Path(env)
        return p if p.parent.is_dir() else None

    up = os.environ.get("USERPROFILE")
    if up:
        p = Path(up) / REL
        if p.parent.is_dir():
            return p

    if LEGACY.parent.is_dir():
        return LEGACY
    return None


def superseded_dir():
    """被淘汰的旧模型放这里（在同步树之外，所以不会上传）。"""
    root = sync_root()
    if root is not None:
        return root.parent / "_superseded_models"
    up = os.environ.get("USERPROFILE")
    if up:
        return Path(up) / "Desktop" / "存档" / "_superseded_models"
    return LEGACY.parent / "_superseded_models"


# ---------------------------------------------------------------------------
# 归档目录布局（2026-10-04 用户指令重组）
# ---------------------------------------------------------------------------
# **背景**：以前同步根下是三个平级目录 `Aquarius/` · `Aquarius_models/` ·
# `Aquarius_handover/`。2026-10-04 用户要求全部合并进**单一根目录**
# `Aquarius20261004/`，并按用途分子目录。
#
# ⚠️ **为什么必须有这一节**：那三个旧目录名原本被 6 个脚本硬编码着，
# 只要跑一次就会**在同步根下重新建出旧目录树**（2026-10-04 实际发生过：
# 我用 `aq_export_latest.py` 打包后 `Aquarius_models/` 又冒出来了）。
# 所有写同步目录的脚本**一律走这里的函数**，不要再自己拼路径。
ARCHIVE_NAME = "Aquarius20261004"
_SUB = {
    "code":     "01_code",
    "data":     "02_data",
    "ckpt":     "03_checkpoints",
    "models":   "04_models",
    "docs":     "05_deliverables",
    "handover": "06_handover",
    "logs":     "07_logs_docs",
}


def archive_root():
    """同步根下的归档总目录；探测不到同步根时返回 None。"""
    root = sync_root()
    return (root / ARCHIVE_NAME) if root is not None else None


def _sub(key):
    a = archive_root()
    return (a / _SUB[key]) if a is not None else None


def code_dir():
    return _sub("code")


def data_dir():
    return _sub("data")


def ckpt_dir():
    """训练检查点归档（final.pt / latest.pt 在它的 ternary/ 子目录下）。"""
    return _sub("ckpt")


def models_dir():
    """量化模型投放目录 —— aq_pack / aq_export_latest / aq_evolution 写这里。"""
    return _sub("models")


def docs_dir():
    """交付文档（报告 / 交接书 / RUNBOOK 的 md·docx·pdf 与配图）。"""
    return _sub("docs")


def handover_dir():
    return _sub("handover")


def logs_dir():
    return _sub("logs")


def interpreter():
    """训练用的解释器：优先本机托管环境，否则用当前解释器。

    目标机上托管环境不存在 → 自动用 `sys.executable`，
    也就是"谁在跑这个脚本就用谁"，不需要改代码。
    """
    import sys
    cand = Path.home() / ".workbuddy" / "binaries" / "python" / "envs" / "aquarius" / "Scripts" / "python.exe"
    if cand.is_file():
        return cand
    return Path(sys.executable)


# ---------------------------------------------------------------------------
# 检查点可用性验证
# ---------------------------------------------------------------------------
# 为什么必须有：2026-10-03 夜里出现**反复的写盘失败**
# （`PytorchStreamWriter failed writing file data/NNN: file write failed`，
#  磁盘并未满，207 GB 可用；`unexpected pos X vs X-112` 说明文件被截断），
# 每次都会在 checkpoints 目录里**留下一个半截的 ckpt_NNNNNN.pt**。
# 而 `aq_export_latest` 与归档器的 `--ckpt-mode newest` **都按"编号最大"挑源**，
# 于是会挑中这个半截文件 → 导出失败，或者更糟：**把坏检查点镜像上云**，
# 新机器拿到一个 load 不动的文件。
# 所以挑源之前必须验证。
CKPT_MIN_BYTES = 10 ** 9            # 小于 1 GB 的编号快照一定是坏的（十进制，= 953.7 MiB）


def ckpt_ok(path, want_tensors=686):
    """验证一个检查点能不能用。返回 (是否可用, 说明)。

    只做**廉价的强校验**：大小下限 + mmap 读顶层结构。
    `torch.save` 是顺序写 zip、中央目录在末尾，**被截断必然读不出来**，
    所以能 mmap 成功就等于完整（约 0.2 秒，不会把 5.2GB 读进内存）。
    """
    p = Path(path)
    try:
        sz = p.stat().st_size
    except OSError as e:
        return False, f"stat 失败：{e}"
    if sz < CKPT_MIN_BYTES:
        return False, f"体积只有 {sz / 1e6:.0f} MB（< 1 GB）—— 半截文件"
    try:
        import torch
        b = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
        step = b.get("step")
        n_ema = len(b.get("ema") or {})
        n_opt = len((b.get("opt") or {}).get("state", {}))
        del b
    except Exception as e:                                     # noqa: BLE001
        return False, f"读不出来：{type(e).__name__}: {str(e)[:70]}"
    if n_ema != want_tensors:
        return False, f"ema 张量 {n_ema} != {want_tensors}"
    if n_opt != want_tensors:
        return False, f"优化器状态 {n_opt} != {want_tensors}"
    return True, f"step={step} ema={n_ema} opt={n_opt} {sz / 1e9:.3f} GB"


def newest_good_ckpt(ckpt_dir, want_tensors=686, max_try=5):
    """按编号从新到旧找**第一个可用的**编号快照。返回 (Path, 说明) 或 (None, 原因)。

    不要用"加个 try 就算"的写法 —— 必须真的验证，否则半截文件会被静默选中。
    """
    import re as _re
    rows = []
    for p in Path(ckpt_dir).glob("ckpt_*.pt"):
        m = _re.search(r"ckpt_(\d+)\.pt$", p.name)
        if m:
            rows.append((int(m.group(1)), p))
    rows.sort(reverse=True)
    if not rows:
        return None, "目录里没有 ckpt_NNNNNN.pt"
    bad = []
    for step, p in rows[:max_try]:
        ok, why = ckpt_ok(p, want_tensors)
        if ok:
            if bad:
                return p, f"step={step}（跳过了 {len(bad)} 个坏文件：{', '.join(bad)}）"
            return p, f"step={step}（{why}）"
        bad.append(p.name)
    return None, f"最新 {len(bad)} 个都不可用：{', '.join(bad)}"


if __name__ == "__main__":
    print("USERPROFILE   =", os.environ.get("USERPROFILE"))
    print("sync_root()   =", sync_root())
    print("superseded()  =", superseded_dir())
    print("interpreter() =", interpreter())
