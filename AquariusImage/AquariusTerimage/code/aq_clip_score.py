# -*- coding: utf-8 -*-
"""Aquarius **语义**指标 —— CLIP-score（图文对齐度）+ 真实图上界。

为什么必须有它
--------------
`aq_metrics.py` 的潜向量分布距离只能证明「**像不像真实潜向量的分布**」，
**证明不了「图里有没有提示词说的那个东西」** —— 而后者才是本项目的验收标准
（HANDOFF §0：「能否辨认提示词里的主体 + 有无合理空间层次」）。

CLIP-score 直接量「图像嵌入 ↔ 文本嵌入」的余弦相似度。它是**逐图**指标 ⇒
**几个样本就能用**，不受有限样本偏差支配（FID 那种分布指标在 n=16 时基本是噪声）。

为什么要「真实图上界」
----------------------
CLIP-score 的**绝对值没有意义**（取决于模型、提示词、图像内容）。同一个提示词，
把**真实 COCO 图**喂进去得到的分数才是「这个模型能达到的上界」。所以看归一化比值：

    clip_ratio = (生成图与本文相似度 − 错配基线) / (真实图与本文相似度 − 错配基线)
        1.0  → 达到真实图水平（图文对齐这一项满分）
        0.0  → 与随机配对无异（完全没对齐）
        < 0  → 比随机配对还差（异常，要查）

反查真实潜向量（已实测校验）
----------------------------
`Aquarius_cloud/latents/index_cache.json` 的 `items[i]` 与潜向量**全局第 i 行**一一对应：
  - 各桶条数 与 该桶分片行数**完全一致**（22 个桶全部 ✓）
  - 分片按文件名排序后，COCO id 区间**严格递增、无重叠**（512x640 / 448x640 / 640x512 抽查 ✓）
桶内第 k 行 = 该桶分片（按文件名排序）拼接后的第 k 行。

依赖
----
`openai/clip-vit-base-patch32`，落在
`~/.cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots/*/`。
**缺了不会让晨报挂掉** —— 本脚本打印补齐办法并返回 0。

用法
----
    python code/aq_clip_score.py --lat _t_ladder/step009058.pt _t_ladder/step117000.pt
    python code/aq_clip_score.py --lat 质量指标_s117000.latents.pt --n 12
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn
from PIL import Image

WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "code"))
sys.path.insert(0, str(WS / "play"))

LATENTS = WS / "Aquarius_cloud" / "latents"
CAPTIONS = WS / "Aquarius_cloud" / "captions" / "captions_train2017.json"
VAE_DIR = WS / "Aquarius_cloud" / "VAE" / "vae"
CLIP_REPO = "models--openai--clip-vit-base-patch32"
SCALE = 0.18215
_HF_HOME = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
HUB = Path(os.environ.get("HF_HUB_CACHE", str(_HF_HOME / "hub")))


# --------------------------------------------------------------------------
# CLIP / VAE
# --------------------------------------------------------------------------
def clip_dir():
    """找本地缓存的 CLIP 快照；没有返回 None（**不联网、不下载**）。"""
    base = HUB / CLIP_REPO / "snapshots"
    if not base.is_dir():
        return None
    for s in sorted(base.iterdir(), reverse=True):
        if (s / "config.json").is_file() and ((s / "pytorch_model.bin").is_file()
                                              or (s / "model.safetensors").is_file()):
            return s
    return None


def load_clip(d):
    from transformers import CLIPModel, CLIPProcessor
    return CLIPModel.from_pretrained(str(d)).eval(), CLIPProcessor.from_pretrained(str(d))


def load_vae():
    from diffusers import AutoencoderKL
    return AutoencoderKL.from_pretrained(str(VAE_DIR), torch_dtype=torch.float32).eval()


def decode(lat, vae):
    """(N,4,h,w) 潜向量 -> (N,H,W,3) uint8。CPU / fp32。"""
    with torch.no_grad():
        x = vae.decode(lat.float() / SCALE).sample
    return (((x.float() / 2 + 0.5).clamp(0, 1) * 255)
            .permute(0, 2, 3, 1).round().to(torch.uint8).numpy())


def _pil(arrs):
    from PIL import Image
    return [Image.fromarray(a) for a in arrs]


def emb_img(model, proc, imgs):
    """⚠️ **不要用 `get_image_features`**：transformers 5.x 改成了返回
    `BaseModelOutputWithPooling` 而不是张量（实测 AttributeError）。
    这里显式走 `vision_model` + `visual_projection`，跨版本稳定。"""
    inp = proc(images=_pil(imgs), return_tensors="pt")
    with torch.no_grad():
        v = model.vision_model(pixel_values=inp["pixel_values"])
        e = model.visual_projection(v.pooler_output)
    return Fn.normalize(e.float(), dim=-1)


def emb_txt(model, proc, texts):
    inp = proc(text=[t[:77] for t in texts], return_tensors="pt",
               padding=True, truncation=True, max_length=77)
    with torch.no_grad():
        t = model.text_model(input_ids=inp["input_ids"],
                             attention_mask=inp.get("attention_mask"))
        e = model.text_projection(t.pooler_output)
    return Fn.normalize(e.float(), dim=-1)


# --------------------------------------------------------------------------
# 潜向量索引（行号 ↔ COCO 文件名）
# --------------------------------------------------------------------------
class LatIndex:
    def __init__(self):
        self.items = json.loads((LATENTS / "index_cache.json")
                                .read_text(encoding="utf-8"))["items"]
        self.by_name = {it["f"]: i for i, it in enumerate(self.items)}
        self._pos = {}                      # bucket -> 全局 index 位置列表
        self._rank = {}                     # bucket -> {全局位置: 桶内行号}
        self._shards = {}                   # bucket -> (分片路径列表, 起始偏移数组)

    def _bucket_of(self, g):
        it = self.items[g]
        return int(it["H"]), int(it["W"])

    def _prepare(self, bucket):
        if bucket in self._shards:
            return
        from safetensors import safe_open
        pos = [i for i, it in enumerate(self.items)
               if (int(it["H"]), int(it["W"])) == bucket]
        shards = sorted(LATENTS.glob(f"latents_{bucket[0]}x{bucket[1]}_shard*.safetensors"))
        sizes = []
        for s in shards:
            with safe_open(str(s), framework="pt") as h:
                sizes.append(h.get_slice("latents").get_shape()[0])
        self._pos[bucket] = pos
        self._rank[bucket] = {g: r for r, g in enumerate(pos)}
        self._shards[bucket] = (shards, np.cumsum([0] + sizes[:-1]))

    def rows_of_ids(self, ids):
        """[id] -> [(id, bucket, 分片路径, 桶内行号)]，**保持输入顺序**。"""
        out = []
        for iid in ids:
            g = self.by_name.get(f"{iid:012d}.jpg")
            if g is None:
                out.append((iid, None, None, None))
                continue
            bucket = self._bucket_of(g)
            self._prepare(bucket)
            r = self._rank[bucket][g]
            shards, starts = self._shards[bucket]
            k = int(np.searchsorted(starts, r, side="right") - 1)
            out.append((iid, bucket, shards[k], r - int(starts[k])))
        return out

    def fetch_one(self, entry):
        from safetensors import safe_open
        _iid, _b, path, row = entry
        with safe_open(str(path), framework="pt") as h:
            return h.get_slice("latents")[row:row + 1].float()[0]


class StagedScorer:
    """对**图片目录**算 clip_ratio（不依赖潜向量）—— 给进化图的每一行用。

    与 `main()` 的区别：那边从 `.pt` 里的潜向量解码再打分；
    而进化图的产物是 640×640 的 **PNG**，直接打分才对应「图上看到的图」。

    两个优化：
      * **文本侧**（10 条 caption 的嵌入）与**真实图侧**（10 张真实原图）
        只算一次，之后所有阶段复用 —— 它们与阶段无关。
      * 每个阶段的结果缓存在该目录的 `clip.json` 里（带图数指纹），
        小时级自动化重复跑时**几乎零成本**。
    """

    def __init__(self, pairs, real_dir=None):
        self.pairs = [(int(i), str(c)) for i, c in pairs]
        self.real_dir = Path(real_dir) if real_dir else None
        self.clip = self.proc = None
        self._et = self._e_r = None
        self.available = None

    def _ensure(self):
        if self.available is not None:
            return self.available
        d = clip_dir()
        if d is None:
            self.available = False
            return False
        self.clip, self.proc = load_clip(d)
        self.available = True
        return True

    def _prep(self):
        """文本侧 + 真实图侧，只算一次。"""
        if self._et is not None:
            return
        self._et = emb_txt(self.clip, self.proc, [c for _i, c in self.pairs])
        if self.real_dir and self.real_dir.is_dir():
            imgs = []
            for i, (_iid, c) in enumerate(self.pairs):
                f = self.real_dir / f"{i:02d}_{_slug(c)}.png"
                if f.is_file():
                    imgs.append(np.asarray(Image.open(f).convert("RGB")))
            if len(imgs) == len(self.pairs):
                self._e_r = emb_img(self.clip, self.proc, imgs)

    def score(self, img_dir, use_cache=True):
        """返回 dict(生成图, 真实图上界, 错配基线, clip_ratio, n) 或 None。"""
        img_dir = Path(img_dir)
        cache = img_dir / "clip.json"
        n_expect = len(self.pairs)
        if use_cache and cache.is_file():
            try:
                c = json.loads(cache.read_text(encoding="utf-8"))
                if c.get("n") == n_expect and c.get("v") == 2:
                    return c
            except Exception:                                  # noqa: BLE001
                pass
        if not self._ensure():
            return None
        self._prep()
        imgs, texts = [], []
        for i, (_iid, c) in enumerate(self.pairs):
            f = img_dir / f"{i:02d}_{_slug(c)}.png"
            if not f.is_file():
                return None
            imgs.append(np.asarray(Image.open(f).convert("RGB")))
            texts.append(c)
        ei = emb_img(self.clip, self.proc, imgs)
        s_gen = (ei * self._et).sum(-1).numpy()
        sim = ei @ self._et.T
        off = float(sim[~torch.eye(len(imgs), dtype=torch.bool)].mean())
        cg = float(s_gen.mean())
        cr_each = ((self._e_r * self._et).sum(-1).numpy()
                   if self._e_r is not None else None)
        cr = float(np.mean(cr_each)) if cr_each is not None else None
        ratio = ((cg - off) / (cr - off)) if (cr is not None and cr - off > 1e-6) else None
        rec = dict(生成图=round(cg, 4),
                   真实图上界=(round(cr, 4) if cr is not None else None),
                   错配基线=round(off, 4),
                   **{"clip_ratio": (round(ratio, 4) if ratio is not None else None)},
                   n=n_expect,
                   # ---- v2：逐列原始分，供「分列诊断 + 自助法误差棒」用 --------------
                   # 为什么必须存：每列是**同 caption 同 seed** ⇒ 跨阶段是**配对**设计，
                   # 所以「某阶段退步了几列、是哪几列」本身是可解释的证据，
                   # 而这需要逐列的数，不能只有均值（均值会把个别难例摊掉）。
                   v=2,
                   逐列_生成=[round(float(x), 4) for x in s_gen],
                   逐列_真实=([round(float(x), 4) for x in cr_each]
                            if cr_each is not None else None),
                   错配矩阵=[[round(float(x), 4) for x in row] for row in sim.numpy()])
        try:
            cache.write_text(json.dumps(rec, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        except OSError:
            pass
        return rec


def _slug(s, n=52):
    return "".join(ch if ch.isalnum() else "_" for ch in str(s))[:n]


def caption_pairs(n, seed=20261003):
    """挑 n 条提示词，其原图**确实在数据集里**（这样才能算真实上界）。"""
    import random
    anns = json.loads(CAPTIONS.read_text(encoding="utf-8")).get("annotations", [])
    idx = LatIndex()
    seen, out = set(), []
    rng = random.Random(seed)
    rng.shuffle(anns)
    for a in anns:
        c = str(a.get("caption", "")).strip()
        iid = int(a.get("image_id", 0))
        if not c or c in seen or f"{iid:012d}.jpg" not in idx.by_name:
            continue
        seen.add(c)
        out.append((iid, c))
        if len(out) >= n:
            break
    return out


# --------------------------------------------------------------------------
def main():
    ap_ = argparse.ArgumentParser(description="Aquarius CLIP-score（图文对齐度）")
    ap_.add_argument("--lat", nargs="+", required=True,
                     help="一个或多个含 'latents' 的 .pt（aq_metrics.py 的产物）")
    ap_.add_argument("--n", type=int, default=16)
    ap_.add_argument("--json-out", default=None)
    a = ap_.parse_args()

    d = clip_dir()
    if d is None:
        print("⚠️ 本地没有缓存 CLIP 权重（openai/clip-vit-base-patch32），跳过语义指标。")
        print("   补齐办法（走镜像，约 605 MB）：")
        print('     HF_ENDPOINT=https://hf-mirror.com python -c "from huggingface_hub '
              "import snapshot_download as s; print(s('openai/clip-vit-base-patch32'))\"")
        return 0
    print(f"CLIP: {d}")
    clip, proc = load_clip(d)
    vae = load_vae()
    idx = LatIndex()

    results = {}
    for lp in a.lat:
        p = Path(lp)
        if not p.is_file():
            print(f"跳过（不存在）：{p}")
            continue
        blob = torch.load(p, map_location="cpu", weights_only=False)
        gen = blob["latents"][:a.n]
        n = gen.shape[0]

        # 提示词来源，优先顺序很重要 —— 用错了会**静默**算出无意义的低分：
        #   ① 文件里存的 (image_id, caption)：最可靠
        #   ② `aq_metrics.load_prompts(n)`：**确定性种子**，只要生成时用的也是它
        #      （_t_lat_ladder.py / aq_metrics.py 都是），就能精确复现同一批提示词
        #   ③ 兜底另挑一批 —— **此时图文是错配的，分数只能当"随机水平"看**
        pids, ptxt = blob.get("prompt_ids"), blob.get("prompts")
        mismatched = False
        if pids and ptxt and len(pids) >= n:
            pairs = list(zip(pids, ptxt))[:n]
        else:
            try:
                import aq_metrics as _M
                pairs = _M.load_prompts(n)
                print(f"  ⓘ 文件里没存提示词，按确定性种子复现 {n} 条"
                      f"（需要生成时用的也是 aq_metrics.load_prompts）")
            except Exception as e:                             # noqa: BLE001
                pairs = caption_pairs(n)
                mismatched = True
                print(f"  ⚠️ 复现提示词失败（{e}），改挑新的一批 ⇒ **图文错配，分数只能当随机水平**")
        n = min(n, len(pairs))
        gen, pairs = gen[:n], pairs[:n]
        texts = [c for _i, c in pairs]
        ids = [int(i) for i, _c in pairs]
        tag = blob.get("name") or p.stem
        print(f"\n{'=' * 74}\n{tag} · {n} 张 · 桶 {blob.get('bucket')}")

        t0 = time.time()
        g_imgs = decode(gen, vae)
        ei = emb_img(clip, proc, g_imgs)
        et = emb_txt(clip, proc, texts)
        s_gen = (ei * et).sum(-1).numpy()
        sim = ei @ et.T
        off = float(sim[~torch.eye(n, dtype=torch.bool)].mean())

        ri, rt = [], []
        for entry in idx.rows_of_ids(ids):
            if entry[1] is None:
                continue
            ri.append(decode(idx.fetch_one(entry)[None], vae)[0])
            rt.append(dict(pairs)[entry[0]])
        s_real = None
        if ri:
            # ⚠️ 真实图**尺寸不一**（每条提示词的原图在各自的桶里），不能 np.stack，
            # 只能逐张编码。CLIP 反正会把每张缩到 224×224，成本可以接受。
            e_r = torch.cat([emb_img(clip, proc, [im]) for im in ri], 0)
            s_real = (e_r * emb_txt(clip, proc, rt)).sum(-1).numpy()
        print(f"  （解码+推理 {(time.time() - t0) / 60:.1f} min）")

        cg = float(s_gen.mean())
        cr = float(s_real.mean()) if s_real is not None else None
        print(f"  生成图 ↔ 本文   = {cg:.4f}   逐图 {np.round(s_gen, 3).tolist()}")
        if cr is not None:
            print(f"  真实图 ↔ 本文   = {cr:.4f}   ← **上界**"
                  f"   逐图 {np.round(s_real, 3).tolist()}")
        print(f"  错配基线（生成图 × 别人的提示词）= {off:.4f}")
        ratio = None
        if cr is not None and cr - off > 1e-6:
            ratio = (cg - off) / (cr - off)
            print(f"  ⭐ clip_ratio = {ratio:.3f}"
                  f"   ← 1 = 达到真实图水平 · 0 = 与随机配对无异 · 越低越差")
        results[tag] = dict(生成图=round(cg, 4),
                            真实图上界=(round(cr, 4) if cr is not None else None),
                            错配基线=round(off, 4),
                            clip_ratio=(round(ratio, 4) if ratio is not None else None),
                            n=n, 提示词错配=mismatched, 文件=str(p))

    if len(results) >= 2:
        print(f"\n{'=' * 74}\n横向对照（clip_ratio 越大越好）\n{'=' * 74}")
        print(f"{'检查点':22s}{'生成图↔本文':>13s}{'真实上界':>11s}"
              f"{'错配':>9s}{'clip_ratio':>12s}")
        for k, v in results.items():
            rr = v["真实图上界"]
            qq = v["clip_ratio"]
            print(f"{k:22s}{v['生成图']:>13.4f}"
                  f"{(rr if rr is not None else float('nan')):>11.4f}"
                  f"{v['错配基线']:>9.4f}"
                  f"{(qq if qq is not None else float('nan')):>12.3f}")

    out = Path(a.json_out) if a.json_out else (
        Path.cwd() / f"CLIP语义指标_{datetime.now():%Y%m%d_%H%M}.json")
    out.write_text(json.dumps({"时间": datetime.now().isoformat(timespec="seconds"),
                               "CLIP": str(d), "结果": results},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写出 {out}")
    print("\n⚠️ 提醒：clip_ratio 只证明「图文是否对得上」，")
    print("   **证明不了「画面是否清晰、结构是否合理」** —— 那还是得看图。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
