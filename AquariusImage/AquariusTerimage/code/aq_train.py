"""Aquarius NLT training loop — g128 mean-scale + STE, AdamW8bit, cosine schedule, x0-pred.

Data layout (files pushed to <workspace> later, paths are constants here):
  <workspace>/latents/
    latents_{H}x{W}_shard{K}.safetensors   [n,4,h,w] bf16, metadata: file_names (JSON list)
    manifest.json                          bucket distribution + failed_files
  <workspace>/embeddings/
    manifest.json                          image_ids (ascending) + file_names + length stats
    embed_{en,zh}_shard{K}.safetensors     [n,32,1024] bf16, metadata: start_index

image_id convention: COCO file names 000000XXXXXX.jpg -> int(stem) = image_id.
Latent rows are located via shard metadata file_names; embeddings are located by
searchsorted over the ascending image_ids list + per-shard start_index.

Training (per HANDOFF.md decisions):
  - cosine beta schedule (monotone increasing betas, Nichol-Dhariwal s=0.008, clip 0.999)
  - DDPM forward: x_t = sqrt(abar_t) x0 + sqrt(1-abar_t) eps; network predicts x0; MSE loss
  - NLT: fp32 master weights; every forward quantizes via g128 mean-scale + STE
    (ternary with zero state / binary without); master stays fp32 for checkpointing
  - bitsandbytes AdamW8bit; bf16 autocast; gradient checkpointing
  - epoch-driven length (total = ceil(N/bs) x epochs), checkpoint every 1k
    (fp32 master + EMA + optimizer state, resumable)

Usage:
  python aq_train.py --mode ternary --epochs 1 --bs 4 --lr 5e-4 --lang en
  python aq_train.py --mode binary --epochs 2 --resume
  python aq_train.py --self-test          # synthetic mini-data end-to-end check, no real files
"""
import argparse
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import safe_open, save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aq_unet import AquariusUNet, set_quant_mode, count_params  # noqa: E402

import bitsandbytes as bnb  # noqa: E402

# ---------------------------------------------------------------------------
# ⛔⛔ Windows 控制台是 GBK(936)。**print 里出现 GBK 编不了的字符会直接杀掉训练进程。**
#
# 2026-10-04 11:05 我就这样崩过一次：给 `--stop-at-step` 加的日志里用了 `⇒`（U+21D2），
# 于是 `UnicodeEncodeError: 'gbk' codec can't encode character '\u21d2'` 在**训练第一步之前**
# 就退出，守望器只能反复重启（每次白付一遍模型加载）。中文本身没问题（GBK 有），
# 炸的是 ⇒ ✅ ⚠️ ⭐ 这类符号 —— 而本文件里**本来就潜伏着 3 个**（546/602/603 行的
# error 分支里用了 ⚠️/✅，只是从没触发过）。
#
# ⇒ 与其逐个改字符，不如**把编码错误降级成 '?'**：这样以后任何人加任何符号都炸不了训练。
#   只影响写日志，不影响任何计算；ASCII 的 `step N/` 行不受影响（守望器就靠它判活）。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except Exception:                                         # noqa: BLE001
        pass
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# constants — data paths; workspace auto-detected so the script works from both
# <WS>\code\ and <WS>\Aquarius_cloud\code\ (the run_all.bat location)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
# <WS>\code\aq_train.py      -> parents[0] = <WS>
# <WS>\Aquarius_cloud\code\  -> parents[1] = <WS>
WORKSPACE = next((c for c in (_HERE.parents[0], _HERE.parents[1])
                  if (c / "Aquarius_cloud").exists()), _HERE.parents[0])
LATENTS_DIR = WORKSPACE / "Aquarius_cloud" / "latents"
EMBEDDINGS_DIR = WORKSPACE / "Aquarius_cloud" / "embeddings"
CHECKPOINTS_DIR = WORKSPACE / "checkpoints"

NUM_TIMESTEPS = 1000


# ---------------------------------------------------------------------------
# cosine schedule (decision #3: cosine, kept fixed for distillation)
# ---------------------------------------------------------------------------
def cosine_schedule(T: int = NUM_TIMESTEPS, s: float = 0.008):
    t = torch.arange(T + 1, dtype=torch.float64) / T
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    abar = f / f[0]
    betas = (1 - abar[1:] / abar[:-1]).clamp(max=0.999).float()
    alphas = (1.0 - betas).float()
    abar = torch.cumprod(alphas, dim=0).float()
    return betas, alphas, abar


# ---------------------------------------------------------------------------
# data pipeline
# ---------------------------------------------------------------------------
_LAT_PAT = re.compile(r"latents_(\d+)x(\d+)_shard(\d+)\.safetensors$")
_EMB_PAT = re.compile(r"embed_(\w+)_shard(\d+)\.safetensors$")


def _first_key(handle):
    keys = handle.keys()
    if not keys:
        raise ValueError(f"no tensors in shard")
    return keys[0]


def build_latent_index(latents_dir: Path):
    """-> {(H, W): [(shard_path, row, image_id), ...]} parsed from shard metadata."""
    buckets = {}
    files = sorted(latents_dir.glob("latents_*_shard*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no latent shards under {latents_dir}")
    for p in files:
        mm = _LAT_PAT.match(p.name)
        if mm is None:
            continue
        H, W = int(mm.group(1)), int(mm.group(2))
        with safe_open(p, framework="pt", device="cpu") as f:
            meta = f.metadata() or {}
            names = json.loads(meta.get("file_names", "[]"))
        for row, fn in enumerate(names):
            image_id = int(Path(fn).stem)  # 000000418829.jpg -> 418829
            buckets.setdefault((H, W), []).append((p, row, image_id))
    return buckets


def build_embedding_index(emb_dir: Path, lang: str):
    """-> {'ids': np[int64] ascending, 'shards': [(path, start, n, key), ...]}"""
    manifest_path = emb_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ids = None
    for k in ("image_ids", "image_id_list", "ids"):
        if k in manifest:
            ids = manifest[k]
            break
    if ids is None:
        raise KeyError(
            f"{manifest_path}: no image-id list. Top-level keys = {sorted(manifest.keys())}"
        )
    ids = np.asarray(ids, dtype=np.int64)
    if not np.all(np.diff(ids) > 0):
        raise ValueError("embedding manifest image_ids are not strictly ascending")

    files = sorted(emb_dir.glob(f"embed_{lang}_shard*.safetensors"),
                   key=lambda p: int(_EMB_PAT.match(p.name).group(2)))
    if not files:
        raise FileNotFoundError(f"no embed_{lang}_shard*.safetensors under {emb_dir}")
    shards = []
    covered = 0
    for p in files:
        with safe_open(p, framework="pt", device="cpu") as f:
            meta = f.metadata() or {}
            key = _first_key(f)
            n = f.get_slice(key).get_shape()[0]
        start = int(meta.get("start_index", covered))
        shards.append((p, start, n, key))
        covered = max(covered, start + n)
    if covered != len(ids):
        raise ValueError(f"embedding shards cover {covered} rows but manifest lists {len(ids)} ids")
    return {"ids": ids, "shards": shards}


class ShardCache:
    """Keeps safetensors handles open; row-level reads via get_slice (no full-shard loads)."""

    def __init__(self):
        self._handles = {}

    def handle(self, path: Path):
        h = self._handles.get(path)
        if h is None:
            h = safe_open(path, framework="pt", device="cpu")
            self._handles[path] = h
        return h

    def row(self, path: Path, key: str, row: int) -> torch.Tensor:
        return self.handle(path).get_slice(key)[row].clone()

    def __del__(self):
        self._handles.clear()


class AquariusData:
    """Bucket sampler + aligned (latent, embedding) fetcher."""

    def __init__(self, latents_dir: Path, emb_dir: Path, lang: str = "en"):
        self.buckets = build_latent_index(latents_dir)
        self.emb = build_embedding_index(emb_dir, lang)
        self.emb_ids = self.emb["ids"]
        self.emb_starts = np.asarray([s[1] for s in self.emb["shards"]], dtype=np.int64)
        self.cache = ShardCache()
        self._latent_keys = {}
        for (H, W), rows in self.buckets.items():
            for (p, _r, _i) in rows[:1]:  # key per shard resolved lazily below
                pass
        # resolve tensor key per shard file (one open per distinct file)
        self._keys = {}
        for lst in self.buckets.values():
            for (p, _r, _i) in lst:
                if p not in self._keys:
                    with safe_open(p, framework="pt", device="cpu") as f:
                        self._keys[p] = _first_key(f)

    def latent_hw(self, H: int, W: int):
        h, w = H // 8, W // 8  # VAE /8
        return h, w

    def emb_row_of(self, image_id: int) -> int:
        pos = int(np.searchsorted(self.emb_ids, image_id))
        if pos >= len(self.emb_ids) or self.emb_ids[pos] != image_id:
            raise KeyError(f"image_id {image_id} not in embedding manifest")
        return pos

    def get_embedding(self, image_id: int) -> torch.Tensor:
        pos = self.emb_row_of(image_id)
        k = int(np.searchsorted(self.emb_starts, pos, side="right") - 1)
        p, start, n, key = self.emb["shards"][k]
        if not (start <= pos < start + n):
            raise IndexError(f"image_id {image_id}: row {pos} outside shard {p.name}")
        return self.cache.row(p, key, pos - start)

    def get_latent(self, H: int, W: int, row_idx: int):
        p, row, image_id = self.buckets[(H, W)][row_idx]
        return self.cache.row(p, self._keys[p], row), image_id

    def sample_for_bucket(self, H: int, W: int, bs: int, rng: random.Random):
        """Draw one same-bucket batch: latents [B,4,h,w], ctx [B,32,1024], ids."""
        rows = self.buckets[(H, W)]
        picks = rng.sample(range(len(rows)), min(bs, len(rows)))
        lat, ids = [], []
        for r in picks:
            x0, image_id = self.get_latent(H, W, r)
            lat.append(x0)
            ids.append(image_id)
        ctx = [self.get_embedding(i) for i in ids]
        return torch.stack(lat), torch.stack(ctx), ids

    def sample_batch(self, bs: int, generator: random.Random):
        """Uniform over buckets, without replacement inside the batch.
        Returns latents [B,4,h,w] bf16, ctx [B,32,1024] bf16, (H, W)."""
        (H, W) = generator.choice(list(self.buckets.keys()))
        x0, ctx, ids = self.sample_for_bucket(H, W, bs, generator)
        return x0, ctx, ids, (H, W)


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
class EMA:
    """Exponential moving average of model weights (mirrors COCO Lite train.py).

    ``device`` / ``dtype`` exist to buy VRAM back on a small card. Measured on the
    5090 with code/aq_bench_lowvram.py (558,347,012 params):

      gpu + fp32  2.233 GB resident -- the original behaviour, unchanged by default
      gpu + bf16  1.117 GB, step time unchanged within noise (0.505 vs 0.514 s/it),
                  but the running average itself accumulates in an 8-bit mantissa
      cpu + fp32  zero VRAM, costs ~+95% step time (0.514 -> 1.001 s/it): the
                  read-modify-write spans hundreds of small tensors, so it is bound
                  by kernel-launch overhead rather than PCIe bandwidth

    Whatever the training mode, ``save_checkpoint`` always writes the shadow back
    out in fp32, so pack_weights.py sees exactly the dtype it has always seen.
    """

    def __init__(self, model, decay: float = 0.999, device: str = "gpu",
                 dtype: str = "fp32"):
        self.decay = decay
        self.device = device
        self.dtype = dtype
        self.shadow: dict = {}
        self._adopt(model.state_dict())

    def _target_dtype(self):
        return torch.bfloat16 if self.dtype == "bf16" else torch.float32

    def _adopt(self, sd, device=None):
        """(Re)build the shadow from any state_dict-like mapping.

        ``device`` is load-bearing, not cosmetic: tensors coming out of a
        checkpoint arrive on the CPU, so without an explicit target a GPU-mode
        shadow would silently stay on the host and the first ``update()`` would
        die with "Expected all tensors to be on the same device".
        """
        td = self._target_dtype()
        if self.device == "cpu":
            tgt = torch.device("cpu")
        elif device is not None:
            tgt = torch.device(device)
        else:
            tgt = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        out = {}
        for k, v in sd.items():
            if v.dtype.is_floating_point:
                t = v.detach().to(device=tgt, dtype=td)
            else:          # int buffers: keep the historical fp32 promotion
                t = v.detach().to(device=tgt, dtype=torch.float32)
            out[k] = t.clone()
        self.shadow = out

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if not v.dtype.is_floating_point:
                continue
            s = self.shadow.get(k)
            if s is None:
                continue
            src = v.detach()
            if self.device == "cpu":
                src = src.to("cpu", torch.float32)
            elif self.dtype == "bf16":
                src = src.to(torch.bfloat16)
            else:
                src = src.float()
            s.mul_(self.decay).add_(src, alpha=1 - self.decay)


def build_model(base: int, args):
    chs = [base, base * 2, base * 4, base * 4]
    cfg = dict(block_out_channels=chs, cross_attention_dim=1024, heads=8)
    if base == 352:
        cfg["mid_channels"] = 1056  # legacy pinned 1B config
    return AquariusUNet(cfg)  # base 256 -> mid auto = 1024 (trunk end width)


def _safe_unlink(p: Path) -> bool:
    """让一份检查点退役（默认真删，删不动才退回搬 `_trash/`）。

    背景（2026-10-03 实测，别再踩）：
      · 沙箱里的环境钩子拦截 `os.remove` 时**不是抛 OSError，是直接把进程杀掉**。
        所以「先 os.remove、失败再搬 _trash」在沙箱里会让 `_trash` 兜底**永远走不到**
        —— 实测 `_trash/` 从未被创建，编号检查点堆到 15 个 / 79 GB，watchdog 每 13 分钟
        重启一次，磁盘 26 GB/小时。
      · 但反过来「一律先搬」也不行：沙箱外启动的训练本该能真删，改成先搬就**再也不腾空间**，
        磁盘照样被 5.6 GB/1000 步吃满（只是从「崩溃循环」换成「几小时后写盘失败」）。

    ⇒ 由 `AQ_CKPT_RETIRE` 决定：`delete`（默认）先真删；`trash` 只搬不删。
    """
    if RETIRE != "trash":
        try:
            os.remove(p)
            return True
        except OSError as e:
            print(f"[rotate] WARN os.remove({p.name}) 失败：{e}"
                  f" —— 退回搬进 _trash/（不腾空间）。若本环境会因删除杀进程，"
                  f"请用 AQ_CKPT_RETIRE=trash 启动训练。", flush=True)
    return _to_trash(p)


# ---------------------------------------------------------------------------
# 检查点退役策略 + 里程碑保留（2026-10-03）
#
# `AQ_CKPT_RETIRE` 决定轮转出去的检查点怎么处理：
#   delete（默认）— 真的 os.remove，**能腾出空间**。用于**沙箱外**启动的训练
#                   （用户双击 run_ternary_50ep.bat 就是这种）。
#   trash        — 只改名搬进 _trash/，不删除。用于**在 agent 沙箱里**启动的训练：
#                   沙箱的 safe-delete 钩子拦截 os.remove 时**不是抛异常，是直接杀进程**
#                   （2026-10-03 实测连崩 13 次，见 EXECUTION_LOG §十八）。
#
# ⚠️ 教训：一开始我把 `_safe_unlink` 改成「一律先搬」，结果**沙箱外也不再真删**，
#   磁盘照样被 5.6GB/1000 步的速度吃满 —— 只是把「崩溃循环」换成了「几小时后写盘失败」。
#   两个都不是好结局。所以现在**默认真删**，删不动才退回搬。
RETIRE = os.environ.get("AQ_CKPT_RETIRE", "delete").strip().lower()
TRASH_PURGE_BELOW_GB = float(os.environ.get("AQ_TRASH_PURGE_BELOW_GB", "97"))
MILESTONE_DIR = WORKSPACE / "out" / "milestones"

# 永不参与轮转的名字（用户 2026-10-03：第 50 轮跑完后要保留最后的原始检查点以便续训）
PROTECT_NAMES = {"final.pt", "latest.pt"}


def _step_of_ckpt(p: Path):
    m = re.search(r"ckpt_(\d+)\.pt$", p.name)
    return int(m.group(1)) if m else None


def _milestone_path(mode: str, step: int) -> Path:
    return MILESTONE_DIR / f"aquarius_{mode}_step{step:06d}.safetensors"


def _to_trash(p: Path) -> bool:
    """改名搬进 _trash/（不删除）。"""
    trash = p.parent / "_trash"
    dst = trash / p.name
    try:
        trash.mkdir(exist_ok=True)
        if dst.exists():
            dst = trash / f"{p.stem}.{int(time.time())}{p.suffix}"
        os.rename(p, dst)
        return True
    except OSError as e:
        print(f"[rotate] WARN rename->{trash.name} failed for {p.name}: {e}")
        return False


def purge_trash(out_dir: Path, need_gb: float) -> float:
    """真删 `_trash/` 里最旧的若干份来腾空间（返回腾出的 GB）。

    ⚠️ **只在 RETIRE=delete 时才敢调**：沙箱里 `os.remove` 会杀进程，
    那种模式下一律不碰 `_trash`（留给人工/脚本间回收）。
    """
    if RETIRE == "trash" or need_gb <= 0:
        return 0.0
    trash = out_dir / "_trash"
    if not trash.is_dir():
        return 0.0
    freed = 0.0
    for f in sorted(trash.glob("ckpt_*.pt"), key=lambda q: q.stat().st_mtime):
        if freed >= need_gb:
            break
        try:
            sz = f.stat().st_size / 1e9
            os.remove(f)
            freed += sz
        except OSError as e:
            print(f"[rotate] WARN _trash 清理失败（后续不再尝试）：{e}", flush=True)
            break
    if freed:
        print(f"[rotate] 已从 _trash/ 真删 {freed:.1f}GB 腾空间", flush=True)
    return freed


def pack_milestone(ckpt: Path, step: int, mode: str = "ternary") -> bool:
    """把万步里程碑打成 153.0 MB 的位打包模型，**永久保留**。

    为什么不是留着 5.6 GB 的原始检查点：
        45 个万步里程碑 × 5.6 GB = **253 GB** —— 装不下。
        位打包后 153.0 MB ⇒ 45 个 = **6.9 GB**。
        而且打包版**更准**：它存的是精确码位，检查点的 `ema` 存 fp16 会让
        0.005% 的三元码位翻转（见 `aq_play.load_unet`）。
        ⇒ 出对比图应该**优先用打包版**。

    成功 ⇒ 那个 5.6 GB 的 .pt 就可以正常轮转掉；
    失败 ⇒ `rotate_checkpoints` 会**保护**它，等 `code/aq_milestones.py` 事后补打。
    """
    dst = _milestone_path(mode, step)
    if dst.exists() and dst.stat().st_size > 100 * 1e6:
        return True
    MILESTONE_DIR.mkdir(parents=True, exist_ok=True)
    # ⭐ **不加 --no-sync**：打包完顺便投放一份到百度云同步目录
    #   （`<同步根>/Aquarius_models/`，由 `aq_pack.sync_to_baidu` 处理；
    #    找不到同步目录时它只打印跳过，不会报错，换机器也安全）。
    #   180 MB 的同盘复制 ≈ 秒级，不影响训练。
    cmd = [sys.executable, "-u", str(WORKSPACE / "code" / "aq_pack.py"),
           "--mode", mode, "--ckpt", str(ckpt), "--out", str(dst)]
    print(f"[milestone] 打包 step {step:,} -> {dst.name} …", flush=True)
    t0 = time.time()
    try:
        # ⚠️ errors="replace"：Windows 控制台是 GBK，子进程输出含非 UTF-8 字节
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900,
                           encoding="utf-8", errors="replace",
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    except Exception as e:                                     # noqa: BLE001
        print(f"[milestone] \u26a0\ufe0f 打包异常：{e}", flush=True)
        return False
    ok = dst.exists() and dst.stat().st_size > 100 * 1e6
    print(f"[milestone] {'OK' if ok else 'FAIL'} step {step:,} "
          f"{time.time() - t0:.0f}s"
          + (f"  {dst.stat().st_size / 1e6:.1f} MB" if ok else ""), flush=True)
    if not ok:
        print("            ", (r.stderr or r.stdout or "")[-400:], flush=True)
    return ok


def rotate_checkpoints(out_dir: Path, keep: int = 2, min_free_gb: float = 65.0,
                       keep_every: int = 0, mode: str = "ternary"):
    """Trim numbered checkpoints down to the newest `keep`.

    keep > 0   rotate, as before.
    keep <= 0  keep everything -- interrupted by a disk-space safety valve. A full
               drive aborts the run and costs far more than any single checkpoint,
               so once free space drops below `min_free_gb` we fall back to
               oldest-first rotation and say so loudly.

    `keep_every`（2026-10-03 新增）> 0 时：**万步整数的里程碑不参与轮转**，
    但前提是它**已经打包**成 `out/milestones/aquarius_<mode>_stepN.safetensors`。
    没打包成功的会被保护下来（否则那个里程碑就永久丢了）。

    latest.pt / final.pt are never touched (they are not numbered `ckpt_*`).
    NOTE: the old body used `ckpts[:-keep] if keep else ckpts`, which made keep=0
    delete *every* checkpoint -- the exact opposite of what 0 reads like. Hence the
    explicit branches below.
    """
    ckpts = sorted(out_dir.glob("ckpt_*.pt"))
    if not ckpts:
        return
    if keep > 0:
        candidates = ckpts[:-keep] if len(ckpts) > keep else []
    elif min_free_gb <= 0:
        return
    else:
        free = shutil.disk_usage(out_dir).free / 1e9
        if free >= min_free_gb:
            return
        per = ckpts[-1].stat().st_size / 1e9
        need = max(1, int((min_free_gb - free) / max(per, 0.1)) + 1)
        print(f"[ckpt] \u26a0 \u5269\u4f59\u7a7a\u95f4 {free:.1f}GB < {min_free_gb:.0f}GB "
              f"\u2014 \u6062\u590d\u6700\u65e7\u4f18\u5148\u8f6e\u8f6c\uff08\u5220 {need} \u4e2a\uff09\u4ee5\u4fdd\u62a4\u8bad\u7ec3",
              flush=True)
        candidates = ckpts[:need]

    # ⭐ 硬保护：final.pt / latest.pt **绝不轮转**
    #   （`ckpt_*.pt` 的 glob 本来就看不到它们，这里再加一道，
    #    防止以后有人改 glob 或把 final 改名成 ckpt_ 前缀把成品删了）
    doomed, protected = [], []
    for c in candidates:
        if c.name in PROTECT_NAMES or not re.match(r"ckpt_\d+\.pt$", c.name):
            protected.append(c)
            continue
        s = _step_of_ckpt(c)
        if keep_every > 0 and s is not None and s % keep_every == 0:
            if not _milestone_path(mode, s).exists():
                protected.append(c)
                continue
        doomed.append(c)
    if protected:
        print(f"[rotate] [LOCK] 保护 {len(protected)} 份里程碑（尚未打包，等 "
              f"aq_milestones.py 补打）："
              f"{', '.join(c.stem for c in protected[:4])}", flush=True)

    undone = [old for old in doomed if _safe_unlink(old)]
    if undone:
        tsz = free = 0.0
        try:
            tsz = sum(f.stat().st_size for f in (out_dir / "_trash").glob("*.pt")) / 1e9
            free = shutil.disk_usage(out_dir).free / 1e9
        except OSError:
            pass
        print(f"[rotate] 退役 {len(undone)} 份（模式 {RETIRE}）· "
              f"_trash 累计 {tsz:.0f}GB · 磁盘剩余 {free:.0f}GB", flush=True)
        if free and free < TRASH_PURGE_BELOW_GB:
            print(f"[rotate] \u26a0\ufe0f 磁盘剩余 {free:.0f}GB < {TRASH_PURGE_BELOW_GB:.0f}GB，"
                  f"尝试从 _trash/ 真删腾空间…", flush=True)
            purge_trash(out_dir, min(TRASH_PURGE_BELOW_GB - free + 5, tsz))
            free2 = shutil.disk_usage(out_dir).free / 1e9
            print(f"[rotate] 清理后磁盘剩余 {free2:.0f}GB", flush=True)


def write_resume_kit(out_dir: Path, final: Path, step: int, args) -> None:
    """训练跑完后：校验 `final.pt`、写一张「怎么续训」的卡片、并把命令打到日志里。

    用户 2026-10-03 明确要求：**第 50 轮结束后保留最后的那个原始检查点，便于以后续训**。

    `final.pt` 就是它 —— model(fp32 master) + ema(fp32) + 优化器状态 + 调度器 + abar，
    约 5.2 GB。因为名字不是 `ckpt_<数字>.pt`，`rotate_checkpoints` 的 glob 永远看不到它，
    `aq_archive.prune` 也只 glob `ckpt_*.pt` ⇒ **它不会被任何轮转/清理路径删掉**。
    这里额外做三件事：结构校验、写卡片、把命令打印出来，免得几个月后忘了怎么用。
    """
    size = tensors = opt_states = None
    try:
        size = final.stat().st_size / 1e9
        blob = torch.load(final, map_location="cpu", weights_only=False, mmap=True)
        tensors = len(blob.get("ema") or blob.get("model") or {})
        opt_states = len((blob.get("opt") or {}).get("state") or {})
        del blob
        print(f"[ckpt] final.pt 校验：{size:.3f} GB · {tensors} 张量 · "
              f"{opt_states} 组优化器状态", flush=True)
    except Exception as e:                                     # noqa: BLE001
        print(f"[ckpt] ⚠️ final.pt 校验失败：{e}", flush=True)

    py = sys.executable
    total = getattr(args, "total_steps", None)
    card = f"""Aquarius 续训入口 —— 这张卡片由 aq_train.py 在最后一轮自动生成
================================================================================
成品检查点 : {final}
               {size if size is None else f'{size:.3f}'} GB · step {step:,}
               {tensors if tensors is None else tensors} 张量 · \
{opt_states if opt_states is None else opt_states} 组优化器状态
等价副本   : ckpt_{step:06d}.pt 与 latest.pt（同一步、同样内容）
内含       : model(fp32 master) + ema(fp32) + opt(paged8bit) + sched + abar

★ final.pt / latest.pt **永不参与轮转** —— rotate_checkpoints 的 glob 只认
  `ckpt_<数字>.pt`，aq_archive.prune 也只 glob 那个，所以它们不会被任何清理路径删掉。

--------------------------------------------------------------------------------
以后怎么续训
--------------------------------------------------------------------------------
    cd {WORKSPACE}
    "{py}" -u Aquarius_cloud\\code\\aq_train.py --mode ternary --resume ^
        --lr <新峰值LR> --warmup 6000 --ckpt-every 1000 --tokpx-target 64000 ^
        --keep-ckpts 4 --min-free-gb 43 --opt paged8bit --total-steps <新总步数>

（等价于双击 run_ternary_50ep.bat，但把 --total-steps 调大、必要时调 --lr）

⚠️ 三条铁律（都踩过）
  1. **优化器必须还是 paged8bit 或 adamw8bit**。bnb 与 torch AdamW 之间
     `load_state_dict` 会**静默通过**，但第一次 opt.step() 就抛 KeyError ⇒
     换优化器 = 动量作废 = 只能重跑。
  2. **`--total-steps` 就是 LR 曲线**（不是"还要跑多少步"）。加大它等于
     重新定义退火终点，语义上是 **warm restart（SGDR）**：峰值建议不超过原来的 0.5 倍，
     否则会重现 2026-10-03 那次 activation runaway（act 冲到 1e4、守卫 rc=2 中止）。
  3. **`--resume` 时命令行的 `--lr` 会覆盖检查点里的值**（aq_train.py 里已显式打印确认行）。
     不写 `--lr` 就沿用保存时的值。
     本次保存时的 LR = 见下方「本次配置」。

本次配置（记录在检查点里，`args` 字段）
  mode={getattr(args, 'mode', '?')}  total_steps={total}  lr={getattr(args, 'lr', '?')}
  warmup={getattr(args, 'warmup', '?')}  ckpt_every={getattr(args, 'ckpt_every', '?')}
  keep_ckpts={getattr(args, 'keep_ckpts', '?')}  keep_every={getattr(args, 'keep_every', '?')}
  opt={getattr(args, 'opt', '?')}  tokpx_target={getattr(args, 'tokpx_target', '?')}

--------------------------------------------------------------------------------
别混淆
--------------------------------------------------------------------------------
* `out/milestones/aquarius_ternary_step*.safetensors`（每 1 万步一份，153.0 MB）
  = **只能出图/评测**，里面没有优化器状态，**不能续训**。
* 只有 `final.pt` / `latest.pt` / `ckpt_<数字>.pt` 这三类能续训。
================================================================================
"""
    try:
        card_path = out_dir / "RESUME_HERE.txt"
        card_path.write_text(card, encoding="utf-8")
        print(f"[ckpt] 续训卡片已写好 -> {card_path}", flush=True)
    except OSError as e:
        print(f"[ckpt] ⚠️ 写续训卡片失败：{e}", flush=True)
    print("[ckpt] ✅ 原始检查点已保留：final.pt + latest.pt（永不参与轮转）", flush=True)


def save_checkpoint(path: Path, model, opt, step, args, abar, ema=None, sched=None):
    """COCO Lite-compatible: top-level 'model' (raw) + 'ema' (pack_weights reads ck['ema'])."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # drain async copies before CPU staging (WDDM segfault guard)
    blob = {
        "step": step,
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},  # fp32 master
        # Always stage the shadow as fp32 on CPU, whatever the live EMA dtype is:
        # pack_weights.py reads ck['ema'] and must keep seeing the same dtype.
        "ema": ({k: v.detach().to("cpu", torch.float32) for k, v in ema.shadow.items()}
                if ema else None),
        "opt": opt.state_dict(),
        "args": vars(args),
        "abar": abar.cpu(),
    }
    if sched is not None:
        blob["sched"] = sched.state_dict()
    if path.name == "final.pt":
        blob["mode"] = args.mode
        blob["dataset"] = "coco"
    torch.save(blob, path)


def load_checkpoint(path: Path, model, opt, ema=None, sched=None):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["model"])
    opt.load_state_dict(blob["opt"])
    if ema is not None and blob.get("ema"):
        # Re-adopt through the EMA's own device/dtype policy: a checkpoint stores
        # fp32 on CPU, but the live shadow may be bf16 and/or pinned to the host.
        # Pass the model's device explicitly -- the blob itself is on the CPU.
        ema._adopt(blob["ema"], device=next(model.parameters()).device)
    if sched is not None and blob.get("sched"):
        sched.load_state_dict(blob["sched"])
    return blob["step"], blob["args"], blob["abar"]


def train(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.backends.cudnn.benchmark = True          # conv autotune (shapes repeat per bucket)
        torch.backends.cuda.matmul.allow_tf32 = True   # fp32 matmuls on tensor cores
        torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    set_quant_mode(args.mode)

    betas, alphas, abar = cosine_schedule()
    abar = abar.to(dev)

    model = build_model(args.base, args).to(dev)  # fp32 master weights
    if getattr(args, "compile", False):
        try:
            model = torch.compile(model, dynamic=True)
            print("[compile] torch.compile enabled (dynamic shapes)")
        except Exception as e:  # noqa: BLE001
            print(f"[compile] failed, falling back to eager: {e}")
    n_params = count_params(model)
    opt_cls = bnb.optim.PagedAdamW8bit if args.opt == "paged8bit" else bnb.optim.AdamW8bit
    opt = opt_cls(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
    if args.opt != "adamw8bit":
        print(f"[opt] {args.opt} (8-bit state pages to host RAM)", flush=True)

    data = AquariusData(Path(args.latents_dir or LATENTS_DIR),
                        Path(args.emb_dir or EMBEDDINGS_DIR), args.lang)
    n_images = sum(len(v) for v in data.buckets.values())
    rng = random.Random(args.seed + 1)
    print(f"[data] buckets={len(data.buckets)} images={n_images}"
          f" lang={args.lang} mode={args.mode} base={args.base} params={n_params:,}")

    # ---- per-bucket batch size -------------------------------------------------
    # Why: bucket latent token counts differ 3.1x (256x512 -> 32x64 = 2048 px vs
    # 640x640 -> 80x80 = 6400 px). A flat bs therefore makes the big buckets the real
    # step-time driver while the small ones idle the card — measured 13.3 GB of 34.19 GB
    # used, ~51-76% util. Tokens-per-step mode gives every step ~the same token mass so
    # the *fixed* per-step cost (optimizer, EMA update over 558M params, kernel launches)
    # is amortised over ~2.5x more work per step.
    def bucket_bs(H: int, W: int, n_rows: int) -> int:
        if args.bs_mode == "uniform":
            return max(1, min(args.bs, n_rows))
        px = (H // 8) * (W // 8)                       # latent tokens for one sample
        b = int(round(args.tokpx_target / max(1, px)))
        return max(1, min(b, args.bs_max, n_rows))

    bs_map = {k: bucket_bs(k[0], k[1], len(v)) for k, v in data.buckets.items()}

    # One epoch = a full traversal: every bucket is visited ceil(n_b / bs_b) times.
    # This replaces the old flat ceil(N/bs) estimate, which cannot be right once bs
    # varies per bucket.
    steps_of = {k: math.ceil(len(v) / bs_map[k]) for k, v in data.buckets.items()}
    steps_per_epoch = sum(steps_of.values())
    # total_steps *is* the LR curve -- lr_at() computes its progress as
    # p = (s - warmup) / (total_steps - warmup). Deriving it from --epochs only
    # works while steps_per_epoch stays constant, and that moves with
    # --tokpx-target. So two machines running the SAME --epochs but different
    # batch sizes get different LR curves, and the LR jumps at the handover
    # (measured 2026-10-02: 5.00e-6 at the finish -> 6.03e-5 on the other side,
    # 12.1x). Pin --total-steps to the same number on both machines and the
    # curve is continuous instead.
    if args.total_steps > 0:
        total_steps = args.total_steps
    else:
        total_steps = steps_per_epoch * args.epochs
    eff_bs = n_images / max(1, steps_per_epoch)
    wsum = max(1, steps_per_epoch)
    tokpx_avg = sum(steps_of[k] * bs_map[k] * (k[0] // 8) * (k[1] // 8)
                    for k in data.buckets) / wsum
    _src = ("pinned by --total-steps" if args.total_steps > 0
            else f"epochs={args.epochs} x {steps_per_epoch}")
    print(f"[plan] total_steps={total_steps} ({_src}) "
          f"bs_mode={args.bs_mode} eff_bs={eff_bs:.2f} "
          f"tokpx/step~{tokpx_avg:,.0f}", flush=True)
    if args.bs_mode == "tokpx":
        bsum = ", ".join(f"{k[0]}x{k[1]}:bs{bs_map[k]}" for k in sorted(data.buckets,
                                                                        key=lambda z: -len(data.buckets[z]))[:8])
        print(f"[plan] bs_map (top-8 buckets by size): {bsum}", flush=True)

    # LR: warmup + cosine to 0.05 floor over total_steps (mirrors COCO Lite train.py)
    def lr_at(s):
        if s < args.warmup:
            return (s + 1) / args.warmup
        p = (s - args.warmup) / max(1, total_steps - args.warmup)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    ema = EMA(model, decay=0.999, device=args.ema_device, dtype=args.ema_dtype)
    if args.ema_device != "gpu" or args.ema_dtype != "fp32":
        print(f"[ema] shadow on {args.ema_device}, dtype {args.ema_dtype}", flush=True)

    out_dir = Path(args.out) / args.mode   # checkpoints/<mode>/ — rotation is per-mode
    out_dir.mkdir(parents=True, exist_ok=True)

    start_step = 0
    if args.resume:
        latest = out_dir / "latest.pt"
        if latest.exists():
            start_step, _, abar = load_checkpoint(latest, model, opt, ema, sched)
            abar = abar.to(dev)
            print(f"[resume] from step {start_step}")
            # ------------------------------------------------------------------
            # ⚠️ `--lr` 在 resume 时**必须显式再压一遍**，否则会被检查点状态静默覆盖：
            #     opt.load_state_dict(blob["opt"])  恢复 param_groups['lr']
            #     sched.load_state_dict(...)        恢复 base_lrs
            # LambdaLR 每步算 `pg['lr'] = base_lrs[i] * lr_lambda(epoch)`，
            # 所以真正起作用的是 base_lrs —— 两个都改才稳。
            # 不修的话，命令行写多少 LR 都无效，**跨机器"统一 LR"的约定会静默失效**。
            # ------------------------------------------------------------------
            prev = sched.base_lrs[0] if getattr(sched, "base_lrs", None) else None
            for pg in opt.param_groups:
                pg["lr"] = args.lr
            if getattr(sched, "base_lrs", None):
                for i in range(len(sched.base_lrs)):
                    sched.base_lrs[i] = args.lr
            if prev is not None and abs(prev - args.lr) > 1e-12:
                p = (start_step - args.warmup) / max(1, total_steps - args.warmup)
                p = min(max(p, 0.0), 1.0)
                new_lr = args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * p)))
                print(f"[lr] 检查点里的 LR 是 {prev:.3g}，按 --lr 覆盖为 {args.lr:.3g}"
                      f" → 当前步实际 LR {new_lr:.3g}"
                      f"（{'降' if args.lr < prev else '升'} "
                      f"{max(prev, args.lr) / max(min(prev, args.lr), 1e-30):.2f}×）",
                      flush=True)

    model.train()
    t0 = time.time()
    running = []
    oom_skips = 0

    # ---- health guards (added 2026-10-02, after a silent full collapse) ----
    # The old check only tested isfinite(loss). It was blind: an internal activation
    # runaway (up to ~1e25) was renormalized by the output GroupNorm, so the loss stayed
    # finite (~0.75 = the "predict the mean" baseline) while the model had already
    # degenerated to a constant output with frozen gradients. These two guards make both
    # failure modes loud and abort the run within a few hundred steps.
    import re as _re
    _canary = _re.compile(r"(conv_shortcut|downsamplers|^conv_in$|^conv_out$|\.proj_out$)")
    _act = {"max": 0.0}

    def _act_hook(_m, _i, out):
        if isinstance(out, torch.Tensor):
            v = float(out.detach().abs().max())
            if v > _act["max"]:
                _act["max"] = v

    _hooks = [mod.register_forward_hook(_act_hook)
              for nm, mod in model.named_modules() if _canary.search(nm)]
    print(f"[guard] canary hooks={len(_hooks)} max_act={args.max_act:.0e} "
          f"probe_every={args.probe_every}", flush=True)
    explode_hits = 0
    collapse_hits = 0

    step = start_step
    # Bucket draw is proportional to that bucket's own step quota, so one epoch really is
    # one traversal of the data. The old uniform-over-buckets draw gave the 2-image
    # 512x256 bucket the same 1/23 step share as the 40k-image 448x640 bucket — roughly a
    # 500x oversample of the tail buckets (and a matching undersample of the bulk).
    _bk = list(data.buckets.keys())
    _cum, _acc = [], 0.0
    for _k in _bk:
        _acc += steps_of[_k]
        _cum.append(_acc)
    _tot = _acc

    def draw_bucket():
        x = rng.random() * _tot
        lo, hi = 0, len(_cum) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if _cum[mid] < x:
                lo = mid + 1
            else:
                hi = mid
        return _bk[lo]

    if total_steps <= start_step:
        print(f"[stop] total_steps {total_steps} <= start_step {start_step} — nothing "
              f"left to run. Raise --total-steps (0 means it is derived from --epochs).",
              flush=True)
        return 0

    # --limit-steps bounds the loop only; total_steps (and therefore the LR schedule)
    # stays intact so a smoke test measures the same step the real run will take.
    stop_at = total_steps if args.limit_steps <= 0 else min(total_steps,
                                                            start_step + args.limit_steps)
    # ⭐ --stop-at-step：**绝对**目标（--limit-steps 是相对的，重启一次就漂移）。
    #   2026-10-04 用户决定「跑到第 30 轮就停」，用这个把 452,900 的计划截断到 271,740。
    #   ⚠️ 关键：**不动 total_steps ⇒ lr_at() 的 p 分母不变 ⇒ LR 曲线逐点不变**。
    #      这正是不用「重新钉 total_steps」的原因 —— 那会让 LR 从 9.6e-6 直接跌到 1.6e-6
    #      （而且早期那次 activation runaway 就是重钉 total_steps 引起的）。
    #
    #   ⚠️ 为什么还要**哨兵文件**回退：守望器是用户双击 .bat 起的、绑在用户会话上，
    #      不该由 agent 重启它；但它的 RECIPES 在进程启动时就定死了，改文件对已跑的守望器无效。
    #      ⇒ 让 aq_train 自己在**没有 CLI 开关时读 `out/STOP_AT_STEP`**，这样守望器「崩溃自动
    #      resume」那一下拉起来的新进程就能自动带上截止步数，不需要动守望器。
    _stop = args.stop_at_step
    if _stop <= 0:
        # ⚠️ 必须用上面那个带探测的 WORKSPACE，**不要写 `parents[2]`** ——
        #    `code/aq_train.py` 的 parents[2] 是 Desktop、`Aquarius_cloud/code/` 的才是 WS，
        #    而实际跑的是后者。写死下标会在源码副本上静默读不到哨兵（= 不停止，直接超时）。
        _sf = WORKSPACE / "out" / "STOP_AT_STEP"
        try:
            _stop = int(_sf.read_text(encoding="utf-8").strip().split()[0])
            print(f"[plan] 从哨兵文件读到 stop-at-step={_stop}（{_sf}）", flush=True)
        except Exception:                                     # noqa: BLE001
            _stop = 0
    if _stop > 0:
        stop_at = min(stop_at, _stop)
        print(f"[plan] stop-at-step {_stop} -> 循环在 {stop_at} 步结束"
              f"（total_steps 仍为 {total_steps}，**LR 曲线不变**）", flush=True)

    # ⭐ --stop-at-time：**墙钟截止**（本地时间 HH:MM）。
    #   为什么需要它：租期/机器归还是**硬截止**，不能拿步数去猜速度 ——
    #   2026-10-04 实测步速从 0.58 掉到 0.755 s/步，同一个步数目标就从「15:25 收」变成「17:05 收」。
    #   用法：**同时**给 --stop-at-step 和 --stop-at-time，谁先到算谁，
    #   两者都走同一套完整收尾（final.pt + RESUME_HERE.txt + 打包里程碑）。
    #   ⇒ 速度快就跑到 30 轮，速度慢就按时间保底，不用人盯。
    _deadline = (args.stop_at_time or "").strip()
    if not _deadline:
        try:
            _deadline = (WORKSPACE / "out" / "STOP_AT_TIME").read_text(
                encoding="utf-8").strip().split()[0]
            print(f"[plan] 从哨兵文件读到 stop-at-time={_deadline}", flush=True)
        except Exception:                                     # noqa: BLE001
            _deadline = ""
    if _deadline:
        print(f"[plan] stop-at-time {_deadline}（墙钟）-> 与 stop-at-step 谁先到算谁", flush=True)

    while step < stop_at:
        (H, W) = draw_bucket()
        bs_b = bs_map[(H, W)]
        opt.zero_grad(set_to_none=True)
        x0, ctx, _ids = data.sample_for_bucket(H, W, bs_b, rng)
        x0 = x0.to(dev, non_blocking=True).float()
        ctx = ctx.to(dev, non_blocking=True)  # bf16, autocast handles the rest
        bsz = x0.shape[0]
        t = torch.randint(0, NUM_TIMESTEPS, (bsz,), device=dev)
        noise = torch.randn_like(x0)
        a = abar[t].view(-1, 1, 1, 1)
        x_t = a.sqrt() * x0 + (1 - a).sqrt() * noise
        _act["max"] = 0.0
        amax = 0.0
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                pred_x0 = model(x_t, t, ctx, grad_ckpt=args.grad_ckpt)
            amax = _act["max"]
            if amax > args.max_act:
                explode_hits += 1
                print(f"[GUARD] step {step}: canary activation {amax:.2e} > "
                      f"{args.max_act:.0e} (hit {explode_hits}/3)", flush=True)
                if explode_hits >= 3:
                    print("[GUARD] \u26d4 activation runaway — model collapsing, "
                          "stopping this run. See EXECUTION_LOG.md.", flush=True)
                    return 2
            else:
                explode_hits = 0
            loss = F.mse_loss(pred_x0.float(), x0)
            if not torch.isfinite(loss):
                print(f"[WARN] bucket {H}x{W}: non-finite loss {loss.item()}, batch skipped")
                opt.zero_grad(set_to_none=True)
                continue
            loss.backward()
        except torch.OutOfMemoryError:
            oom_skips += 1
            print(f"[WARN] CUDA OOM at bucket {H}x{W} bs{bs_b} — batch skipped, cache cleared "
                  f"(total skips {oom_skips})")
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema.update(model)
        sched.step()
        step += 1

        if args.probe_every and step % args.probe_every == 0:
            # Probe on a small slice. The collapse test only needs a couple of samples,
            # and in tokpx mode a full-batch probe would spike VRAM by a whole extra
            # forward's worth of activations right after opt.step() (grads still live).
            pb = min(args.probe_bs, x_t.shape[0])
            xs_p, ts_p, cs_p = x_t[:pb], t[:pb], ctx[:pb]
            # probe must run under the same autocast as training, else bf16 ctx hits
            # fp32 weights inside the quantized linears (dtype mismatch)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                                 enabled=(dev == "cuda")):
                p1 = model(xs_p, ts_p, cs_p)
                p2 = model(torch.randn_like(xs_p), ts_p, cs_p)
            dep = float((p1 - p2).abs().max())
            if dep < 1e-4:
                collapse_hits += 1
                print(f"[GUARD] step {step}: output input-dependence={dep:.2e} "
                      f"(collapse {collapse_hits}/3)", flush=True)
                if collapse_hits >= 3:
                    print("[GUARD] \u26d4 output collapsed to a constant — stopping this "
                          "run. See EXECUTION_LOG.md.", flush=True)
                    return 2
            else:
                collapse_hits = 0

        running.append(loss.item())
        if step % args.log_every == 0:
            dt = time.time() - t0
            if dev == "cuda":
                # window peak (reset each window) — the old running global max pinned at
                # 13.3GB forever and hid how much of the card was actually idle
                mem = torch.cuda.max_memory_allocated() / 1e9
                res = torch.cuda.max_memory_reserved() / 1e9
                torch.cuda.reset_peak_memory_stats()
            else:
                mem = res = 0.0
            tokpx = (H // 8) * (W // 8) * bs_b
            print(f"step {step}/{total_steps} loss {sum(running)/len(running):.5f} "
                  f"bucket {H}x{W} bs{bs_b} lr {sched.get_last_lr()[0]:.2e} "
                  f"act {amax:.1e} {dt/args.log_every:.2f}s/it "
                  f"vram {mem:.1f}GB eta {(total_steps - step) * dt / args.log_every / 3600:.1f}h "
                  f"rx {res:.1f}GB tokpx {tokpx}", flush=True)
            running = []
            t0 = time.time()

        # ⭐ 墙钟截止（只在检查点边界判一次，避免每步都取时间）。
        #   过点就把 stop_at 收到**当前这个检查点** ⇒ 下面 `while step < stop_at` 正常退出，
        #   并因为 `step == stop_at` 而走完整收尾。
        _fired = False
        if _deadline and step % args.ckpt_every == 0 and time.strftime("%H:%M") >= _deadline:
            if stop_at > step:
                print(f"[plan] 到墙钟截止 {_deadline}（现在 {time.strftime('%H:%M')}）"
                      f" -> 收在检查点 step {step}（原目标 {stop_at}）", flush=True)
                stop_at = step
            _fired = True

        if step % args.ckpt_every == 0 or step == total_steps or step == stop_at:
            snap = out_dir / f"ckpt_{step:06d}.pt"
            save_checkpoint(snap, model, opt, step, args, abar, ema, sched)
            save_checkpoint(out_dir / "latest.pt", model, opt, step, args, abar, ema, sched)
            # ⭐ 万步里程碑：立刻打成 153MB 的位打包模型永久保留
            #   （不是留着 5.6GB 的原件 —— 45 个里程碑要 253GB，装不下）
            if args.keep_every and step % args.keep_every == 0:
                pack_milestone(snap, step, args.mode)
            rotate_checkpoints(out_dir, keep=args.keep_ckpts,
                               min_free_gb=args.min_free_gb,
                               keep_every=args.keep_every, mode=args.mode)
            # ⭐ 收尾条件：跑满 total_steps，**或**到达 --stop-at-step。
            #   后者不是「提前截断就算了」—— 用户要的是一份正式交付，所以同样走完整收尾：
            #   写 final.pt、写 RESUME_HERE.txt、并**打包这一里程碑**（否则 271,740 不是
            #   万步整数，pack_milestone 不会触发，最后那个模型就只有 5.6 GB 原件没有交付件）。
            if step == total_steps or step == stop_at:
                if _fired:
                    _why = f"stop-at-time {_deadline}（墙钟保底）"
                elif step == total_steps:
                    _why = "total_steps"
                else:
                    _why = f"stop-at-step {_stop}"
                if step != total_steps or step % (args.keep_every or 1) != 0:
                    pack_milestone(snap, step, args.mode)
                fin = out_dir / "final.pt"
                save_checkpoint(fin, model, opt, step, args, abar, ema, sched)
                print(f"[ckpt] FINAL saved step {step} -> {fin}  （{_why}）")
                # ⭐ 用户要求：最后那个**原始**检查点必须留下，以便以后续训。
                #    这里校验它的完整性，并写一张 RESUME_HERE.txt 把续训命令记下来。
                write_resume_kit(out_dir, fin, step, args)
            else:
                print(f"[ckpt] saved step {step} -> {out_dir}")
    return 0


# ---------------------------------------------------------------------------
# self-test: synthetic mini shards + alignment checks + 3 real train steps
# ---------------------------------------------------------------------------
def make_fake_data(tmp: Path):
    """Two buckets, two embedding shards (start_index logic), 8 non-contiguous image ids.
    Layout mirrors the real tree: tmp/latents/ and tmp/embeddings/."""
    lat_dir = tmp / "latents"
    emb_dir = tmp / "embeddings"
    lat_dir.mkdir(parents=True, exist_ok=True)
    emb_dir.mkdir(parents=True, exist_ok=True)
    ids = [7, 21, 34, 56, 90, 123, 187, 250]          # ascending, gappy
    names = [f"{i:012d}.jpg" for i in ids]

    # embeddings: shard0 = rows 0..4 (start 0), shard1 = rows 5..7 (start 5)
    embs = torch.randn(8, 32, 1024, dtype=torch.bfloat16)
    for r, i in enumerate(ids):
        embs[r, 0, :4] = float(i)                      # watermark for alignment check
    save_file({"emb": embs[:5].contiguous()}, emb_dir / "embed_en_shard0.safetensors",
              metadata={"start_index": "0"})
    save_file({"emb": embs[5:].contiguous()}, emb_dir / "embed_en_shard1.safetensors",
              metadata={"start_index": "5"})
    (emb_dir / "manifest.json").write_text(json.dumps(
        {"image_ids": ids, "file_names": names, "note": "fake"}), encoding="utf-8")

    # latents: bucket (64,64) -> [5,4,8,8]; bucket (128,128) -> [3,4,16,16]
    # (bucket pixel dims are multiples of 64 per project spec -> latent dims divisible by 8)
    lat0 = torch.randn(5, 4, 8, 8, dtype=torch.bfloat16)
    for r, i in enumerate(ids[:5]):
        lat0[r, 0, 0, 0] = float(i)                    # watermark
    save_file({"latent": lat0.contiguous()}, lat_dir / "latents_64x64_shard0.safetensors",
              metadata={"file_names": json.dumps(names[:5])})
    lat1 = torch.randn(3, 4, 16, 16, dtype=torch.bfloat16)
    for r, i in enumerate(ids[5:]):
        lat1[r, 0, 0, 0] = float(i)
    save_file({"latent": lat1.contiguous()}, lat_dir / "latents_128x128_shard0.safetensors",
              metadata={"file_names": json.dumps(names[5:])})
    (lat_dir / "manifest.json").write_text(json.dumps(
        {"buckets": {"64x64": 5, "128x128": 3}, "failed_files": []}), encoding="utf-8")
    return embs, lat0, lat1


def self_test(base: int = 88, steps: int = 3, bs: int = 2):
    tmp = Path(__file__).resolve().parent / "_selftest_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    print(f"[self-test] fake data dir: {tmp}")
    try:
        embs, lat0, lat1 = make_fake_data(tmp)

        # -- 1. latent bucket parsing --------------------------------------
        buckets = build_latent_index(tmp / "latents")
        assert set(buckets.keys()) == {(64, 64), (128, 128)}, buckets.keys()
        assert len(buckets[(64, 64)]) == 5 and len(buckets[(128, 128)]) == 3
        print(f"[self-test] 1. bucket parse OK: { {k: len(v) for k, v in buckets.items()} }")

        # -- 2. image_id alignment: latent row <-> embedding row -----------
        data = AquariusData(tmp / "latents", tmp / "embeddings", "en")
        for (H, W), rows in buckets.items():
            for (p, row, image_id) in rows:
                assert int(Path(str(image_id)).stem) == image_id
                pos = data.emb_row_of(image_id)
                e = data.get_embedding(image_id)
                assert torch.equal(e[0, :4].float(),
                                   torch.tensor([float(image_id)] * 4, dtype=torch.float32)), \
                    f"embedding watermark mismatch for id {image_id}"
                l, _ = data.get_latent(H, W, row)
                assert float(l[0, 0, 0]) == float(image_id), \
                    f"latent watermark mismatch for id {image_id}"
                # cross-check against the exact tensors written
                ref_e = embs[pos]
                assert torch.equal(e, ref_e), f"embedding row mismatch id {image_id}"
        print("[self-test] 2. image_id alignment OK (8/8 rows, latent<->embedding<->manifest)")

        # -- 3. batch sampling alignment ------------------------------------
        rng = random.Random(0)
        for _ in range(8):
            lat, ctx, ids, (H, W) = data.sample_batch(2, rng)
            assert lat.shape[0] == ctx.shape[0] == len(ids) == 2
            for b in range(2):
                assert float(lat[b, 0, 0, 0]) == float(ctx[b, 0, 0]) == float(ids[b]), \
                    "batch latent/ctx/id misaligned"
        print("[self-test] 3. batch sampling alignment OK (8 batches)")

        # -- 4. three real train steps (ternary) on the fake data ------------
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        set_quant_mode("ternary")
        torch.manual_seed(0)
        model = build_model(base, argparse.Namespace()).to(dev)
        opt = bnb.optim.AdamW8bit(model.parameters(), lr=1e-3)
        betas, alphas, abar = cosine_schedule()
        abar = abar.to(dev)
        model.train()
        w_before = model.down_blocks[0].resnets[0].conv1.weight.detach().clone()
        losses = []
        for step in range(steps):
            x0, ctx, _ids, _b = data.sample_batch(bs, rng)
            x0 = x0.to(dev).float()
            ctx = ctx.to(dev)
            t = torch.randint(0, NUM_TIMESTEPS, (x0.shape[0],), device=dev)
            noise = torch.randn_like(x0)
            a = abar[t].view(-1, 1, 1, 1)
            x_t = a.sqrt() * x0 + (1 - a).sqrt() * noise
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                pred = model(x_t, t, ctx, grad_ckpt=True)
            loss = F.mse_loss(pred.float(), x0)
            assert torch.isfinite(loss), f"non-finite loss at step {step}"
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        w_after = model.down_blocks[0].resnets[0].conv1.weight.detach()
        moved = (w_after - w_before).abs().max().item()
        assert moved > 0, "master weights did not update"
        print(f"[self-test] 4. {steps} train steps OK (base={base}, ternary, grad_ckpt): "
              f"losses={['%.4f' % x for x in losses]}, max|dw|={moved:.2e}, no NaN")
        print(f"[self-test] model params (base={base}): {count_params(model):,}")
        print("[self-test] ALL CHECKS PASSED")
    finally:
        cleaned = _rmtree_force(tmp)
        print(f"[self-test] cleanup: {'done' if cleaned else 'LEFTOVER (manual delete): ' + str(tmp)}")
    return 0


def _rmtree_force(path: Path) -> bool:
    """Delete our own scratch dir; fall back to per-file unlink if the trash-based
    shutil.rmtree hook fails (returns False and leaves the dir in that case)."""
    try:
        shutil.rmtree(path)
        return True
    except OSError:
        pass
    try:
        for p in sorted(path.rglob("*"), reverse=True):
            if p.is_file():
                os.remove(p)
        for p in sorted(path.rglob("*"), reverse=True):
            if p.is_dir():
                os.rmdir(p)
        os.rmdir(path)
        return True
    except OSError as e:
        print(f"[self-test] cleanup fallback failed: {e}")
        return False


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Aquarius NLT training")
    ap.add_argument("--mode", choices=["fp", "ternary", "binary"], default="ternary")
    ap.add_argument("--epochs", type=int, default=1,
                    help="training length in integer epochs; total_steps = ceil(N/bs) * epochs")
    ap.add_argument("--total-steps", type=int, default=0,
                    help="pin total_steps directly instead of deriving it from "
                         "--epochs. Use this whenever a run has to be handed to a "
                         "machine that picks a different batch size: total_steps is "
                         "what lr_at() keys off, so leaving it to --epochs makes the "
                         "LR jump at the handover (measured 12.1x). Passing the same "
                         "--total-steps on both machines keeps the LR curve "
                         "continuous. 0 = derive from --epochs.")
    ap.add_argument("--warmup", type=int, default=2000,
                    help="LR warmup steps (raised 1000->2000 after the 2026-10-02 collapse)")
    ap.add_argument("--bs", type=int, default=4,
                    help="flat batch size; only used when --bs-mode uniform")
    ap.add_argument("--bs-mode", choices=["tokpx", "uniform"], default="tokpx",
                    help="tokpx (default): per-bucket batch size so that every step "
                         "carries ~--tokpx-target latent token-pixels, which fills the "
                         "VRAM budget instead of letting small buckets idle it. "
                         "uniform: legacy flat --bs for every bucket")
    ap.add_argument("--tokpx-target", type=int, default=56000,
                    help="tokpx mode: target latent token-pixels per step "
                         "= B * (H/8) * (W/8). Raise until logged vram reaches ~75%% "
                         "of the card; the 640x640 bucket alone consumes 6400 px/sample")
    ap.add_argument("--bs-max", type=int, default=32,
                    help="tokpx mode: hard cap on any single bucket's batch size")
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="peak LR (lowered 5e-4->1e-4 after the 2026-10-02 collapse)")
    ap.add_argument("--max-act", type=float, default=1e4,
                    help="abort if a canary-layer activation exceeds this (runaway guard)")
    ap.add_argument("--probe-every", type=int, default=100,
                    help="steps between output-collapse probes (0 disables)")
    ap.add_argument("--probe-bs", type=int, default=2,
                    help="samples used by the collapse probe (small on purpose: the "
                         "probe only needs to prove output depends on the input)")
    ap.add_argument("--lang", choices=["en", "zh"], default="en")
    ap.add_argument("--base", type=int, default=256,
                    help="first block width (256 = current config, ~550M)")
    ap.add_argument("--out", type=str, default=None,
                    help=f"default: {CHECKPOINTS_DIR} (subdir per mode)")
    ap.add_argument("--ckpt-every", type=int, default=5000)
    ap.add_argument("--keep-ckpts", type=int, default=4,
                    help="numbered checkpoints retained on disk. Classic rotation: the "
                         "newest N survive, older ones are unlinked as new ones land. "
                         "Default 4 -> a snapshot lives ~37 min (--ckpt-every 1000 is "
                         "~9 min/ckpt), which is the window the hourly archive job can "
                         "realistically catch. N<=0 = keep them ALL, braked only by "
                         "--min-free-gb.")
    ap.add_argument("--keep-every", type=int, default=10000,
                    help="⭐ \u91cc\u7a0b\u7891\u4fdd\u7559\uff1a\u6bcf N \u6b65\u6253\u4e00\u4efd "
                         "153.0MB \u7684\u4f4d\u6253\u5305\u6a21\u578b\u5230 out/milestones/\uff0c"
                         "**\u6c38\u4e45\u4fdd\u7559**\uff0c\u4e0d\u53c2\u4e0e\u8f6e\u8f6c\u3002\n"
                         "\u4e3a\u4ec0\u4e48\u4e0d\u7559 5.6GB \u539f\u4ef6\uff1a45 \u4e2a\u91cc\u7a0b\u7891 "
                         "\u00d7 5.6GB = 253GB \u88c5\u4e0d\u4e0b\uff1b\u6253\u5305\u540e 45 \u00d7 "
                         "153.0MB = 6.9GB\u3002\u800c\u4e14\u6253\u5305\u7248**\u66f4\u51c6**"
                         "\uff08\u5b58\u7cbe\u786e\u7801\u4f4d\uff0c\u68c0\u67e5\u70b9 ema \u5b58 fp16 "
                         "\u4f1a\u8ba9 0.005% \u4ee3\u7801\u7ffb\u8f6c\uff09\u3002\n"
                         "\u2099 \u51fa\u5bf9\u6bd4\u56fe\u5c31\u7528 out/milestones/ \u91cc\u7684"
                         "\u4e07\u6b65\u6a21\u578b\uff0c\u6b65\u957f\u5747\u5300\u3001\u66f4\u5ba2\u89c2\u3002"
                         "0 = \u5173\u95ed\u3002")
    ap.add_argument("--min-free-gb", type=float, default=65.0,
                    help="disk safety floor, consulted only when --keep-ckpts <= 0 "
                         "(classic rotation already bounds usage): below this many GB "
                         "free the trainer falls back to oldest-first deletion and logs "
                         "loudly (default 60)")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--limit-steps", type=int, default=0,
                    help="stop the loop after this many steps (smoke test); 0 = full run. "
                         "Does not alter total_steps / the LR schedule")
    ap.add_argument("--stop-at-step", type=int, default=0,
                    help="**绝对**步数上限（0 = 关）。与 --limit-steps 的区别：那个是「再跑 N 步」，"
                         "重启一次就漂移；这个是绝对目标，跨重启稳定。到达时会像跑满一样写 "
                         "final.pt + RESUME_HERE.txt 并打包里程碑，然后 rc=0 退出。"
                         "**同样不动 total_steps -> LR 曲线不变**")
    ap.add_argument("--stop-at-time", type=str, default="",
                    help="**墙钟**截止（本地时间 HH:MM，如 15:30）。租期/机器归还这类硬截止用它，"
                         "别拿步数猜速度。与 --stop-at-step 谁先到算谁，两者都走同一套完整收尾")
    ap.add_argument("--grad-ckpt", action="store_true", default=False,
                    help="gradient checkpointing. Measured on the 5090: cuts the "
                         "activation slope from 0.221 to 0.049 GB per 1k tokpx "
                         "(-78%%) but ADDS ~0.56GB of fixed cost and ~+67%% step "
                         "time -- so it only pays off above ~3.3k tokpx, and it "
                         "pays off enormously (see code/aq_bench_lowvram.py).")
    ap.add_argument("--no-grad-ckpt", dest="grad_ckpt", action="store_false")
    ap.add_argument("--opt", choices=["adamw8bit", "paged8bit"], default="adamw8bit",
                    help="adamw8bit (default): 8-bit optimizer state resident on the "
                         "GPU, ~1.04GB. paged8bit: same math, state pages to host "
                         "RAM as needed -- frees that 1.04GB of fixed cost at "
                         "roughly unchanged step time (measured 0.990 vs 1.001 s/it "
                         "at tokpx 6400)")
    ap.add_argument("--ema-device", choices=["gpu", "cpu"], default="gpu",
                    help="where the EMA shadow lives. gpu (default) = original "
                         "behaviour, 2.08GB resident. cpu = zero VRAM, but the "
                         "558M-element read-modify-write crosses PCIe every step "
                         "(measured ~+95%% step time)")
    ap.add_argument("--ema-dtype", choices=["fp32", "bf16"], default="fp32",
                    help="EMA shadow precision. fp32 (default) = original. bf16 "
                         "halves the shadow (2.08 -> 1.04GB) at essentially no step-"
                         "time cost, but the average accumulates in an 8-bit "
                         "mantissa. Checkpoints always store fp32 regardless.")
    ap.add_argument("--compile", action="store_true", default=False,
                    help="torch.compile the model (experimental; falls back to eager on failure)")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--latents-dir", type=str, default=None, help=f"default: {LATENTS_DIR}")
    ap.add_argument("--emb-dir", type=str, default=None, help=f"default: {EMBEDDINGS_DIR}")
    ap.add_argument("--self-test", action="store_true", help="synthetic-data end-to-end check")
    args = ap.parse_args()
    if args.out is None:
        args.out = str(CHECKPOINTS_DIR)

    if args.self_test:
        raise SystemExit(self_test())
    raise SystemExit(train(args))


if __name__ == "__main__":
    main()
