# -*- coding: utf-8 -*-
"""把 Qwen3.5-0.8B 文本编码器**精简 + 低比特量化**成一个可分发的小目录。

## 为什么需要它

Aquarius 推理端要用 Qwen3.5-0.8B 当文本编码器，原版 **1746.9 MB**（旧口径 1666 MiB）太大。实测证明
（`_t_te_quant_end2end.py`：真换 TE 出图，5 条 caption × 20 步 DDIM，step 150k）：

    int8 per-channel  →  出图 L1 ≈ 1.9/255、CLIP 与 bf16 逐条一致   ⇒ **可用**
    int4 per-channel  →  L1 12~22、部分 caption CLIP 掉 0.05          ⇒ 不可用

所以本工具做两件事：
1. **精简**：丢掉**视觉塔**（`model.visual.*`，201.3 MB，不参与文本编码）与
   **mtp**（`mtp.*`，40.9 MB，实例化出的模型里根本没有这两个模块）。
   ⚠️ 但 config 里必须**同时删掉 `vision_config`**，否则 `Qwen3_5Model` 仍会实例化一个
   随机初始化的视觉塔（白占 ~402.7 MB 内存）。已实测：删掉后 AutoConfig 正常解析。
   也已实测：`model.visual = None` 后文本编码 **cos = 1.00000429（逐位相同）**。
2. **量化**：对线性/卷积权重（`ndim>=2` 且名字不含 `norm`，含 `embed_tokens`）
   做 **int8 对称 per-out-channel** RTN。

## ⚠️ 一个必须知道的限制

`transformers` 的 `from_pretrained` **读不了 int8 权重**（dtype 不匹配）。所以产物里附了一个
自带的加载器 `load_te.py`，它把 int8 + scales 还原成 bf16 再灌进模型。

⇒ **省的是磁盘（分发体积），不是内存** —— 加载后仍是全量 bf16 参数在内存里。
   真要省内存得靠 int8 矩阵乘内核（bitsandbytes/torchao），那是另一件事。
   （而显存这边本项目早有 `AQ_TE_MODE=cache` 兜底，本来就不占常驻显存。）

用法：
    python code/aq_te_slim.py --bits 8 --out deliverables/TE/Qwen3.5-0.8B-textonly-int8
    python code/aq_te_slim.py --bits 0 --out <dir>      # 只精简不量化（bf16，可直接 from_pretrained）
    python code/aq_te_slim.py --bits 8 --out <dir> --dry-run
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch

WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "code"))
sys.path.insert(0, str(WS / "play"))

# 除了权重，这些文件也要一起带过去（tokenizer / 配置）
SIDE_FILES = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
              "chat_template.jinja", "preprocessor_config.json",
              "video_preprocessor_config.json", "configuration.json",
              "LICENSE", "README.md"]
MB = 1_000_000          # 十进制 MB（2026-10-03 起全项目统一，不再用 MiB=1024²）

# 验证数字（来自 code/_t_te_slim_verify.py，2026-10-03）
TARGET_COS = "0.99474（最差 0.98226）"
L1S = "1.20 / 1.80 / 2.68（/255）"


def is_quantizable(name, t):
    """可量化的线性/卷积/嵌入权重：≥2 维、且不是 norm。"""
    return t.is_floating_point() and t.ndim >= 2 and "norm" not in name.lower()


def drop(name, keep_vision):
    if name.startswith("mtp.") or name.startswith("model.mtp."):
        return True
    if not keep_vision and (".visual." in name or name.startswith("visual.")):
        return True
    return False


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--src", default="")
    a.add_argument("--out", required=True)
    a.add_argument("--bits", type=int, default=8, choices=[0, 8, 4],
                   help="0 = 不量化（保持 bf16，可直接 from_pretrained）")
    a.add_argument("--keep-vision", action="store_true")
    a.add_argument("--dry-run", action="store_true")
    args = a.parse_args()

    if args.src:
        SRC = Path(args.src)
    else:
        import aq_play as AP
        SRC = Path(AP.TE_DIR)
    OUT = Path(args.out)

    src_f = next(SRC.glob("*.safetensors"))
    from safetensors import safe_open
    from safetensors.torch import save_file

    print(f"源    {SRC}")
    print(f"       {src_f.name}  {src_f.stat().st_size / MB:.0f} MB")
    print(f"目标  {OUT}")
    print(f"方案  丢弃视觉塔={not args.keep_vision} · 丢弃 mtp=是 · bits={args.bits}\n")

    plan, tot_in, tot_out, n_q = [], 0, 0, 0
    with safe_open(str(src_f), framework="pt") as h:
        for k in h.keys():
            if drop(k, args.keep_vision):
                continue
            sl = h.get_slice(k)
            sh = tuple(sl.get_shape())
            n = 1
            for d in sh:
                n *= d
            nbit = 2 if sl.get_dtype() in ("BF16", "F16") else 4
            tot_in += n * nbit
            q = args.bits > 0 and len(sh) >= 2 and "norm" not in k.lower()
            if q:
                tot_out += n * 1 + sh[0] * 4          # int8 + 每输出通道一个 f32 scale
                n_q += 1
            else:
                tot_out += n * 2
            plan.append((k, sh))

    print(f"保留张量 {len(plan)} 张 ·  其中量化 {n_q} 张")
    print(f"精简前（全部）        {1666 * 1.048576:.1f} MB  （含视觉塔 201.3 + mtp 40.9）")
    print(f"精简后 bf16 估算      {tot_in / MB:.0f} MB")
    print(f"精简 + int8 估算      {tot_out / MB:.0f} MB"
          f"  （省 {(1 - tot_out / tot_in) * 100:.0f}% vs 精简 bf16，"
          f"{(1 - tot_out / (1666 * 1.048576 * MB)) * 100:.0f}% vs 原版）")
    if args.dry_run:
        print("\n--dry-run：不写文件")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tensors, quantized = {}, []
    with safe_open(str(src_f), framework="pt") as h:
        for i, (k, sh) in enumerate(plan):
            t = h.get_slice(k)[:]
            if args.bits > 0 and len(sh) >= 2 and "norm" not in k.lower():
                w = t.float()
                ax = tuple(range(1, w.ndim))
                qmax = 2 ** (args.bits - 1) - 1
                s = w.abs().amax(dim=ax, keepdim=True).clamp_min(1e-12) / qmax
                q = (w / s).round().clamp(-qmax - 1, qmax).to(torch.int8)
                tensors[k] = q
                tensors[k + ".__scale__"] = s.squeeze(ax).contiguous().float()
                quantized.append(k)
            else:
                tensors[k] = t.contiguous()
            if (i + 1) % 80 == 0:
                print(f"    … {i + 1}/{len(plan)}  {time.time() - t0:.0f}s", flush=True)
    save_file(tensors, str(OUT / "model_int8.safetensors"),
              metadata={"format": "pt", "scheme": f"int{args.bits}-sym-per-out-channel"})
    nbytes = (OUT / "model_int8.safetensors").stat().st_size
    print(f"\n写出 model_int8.safetensors  {nbytes / MB:.0f} MB  ({time.time() - t0:.0f}s)")

    # ---- config ----
    # ⚠️ **刻意保留 vision_config**（2026-10-03 实测踩坑）：
    #   `Qwen3_5Model` 无论 config 里有没有 vision_config 都会建视觉塔；
    #   而**删掉** vision_config 会让 transformers 回退到一套**更大**的默认视觉配置
    #   （实测：缺失键从 153 个涨到 653 个）。保留它 = 形状与上游一致（153 张量），
    #   再由加载器 `model.visual = None` 把这块内存放掉。
    #   （已实测 `model.visual = None` 后文本编码 cos = 1.00000429 —— 逐位相同。）
    shutil.copy2(SRC / "config.json", OUT / "config.json")

    # ---- 其余附属文件 ----
    copied = []
    for f in SIDE_FILES:
        s = SRC / f
        if s.is_file():
            if f == "README.md":                     # 原始 README 是上游的，改名保留
                shutil.copy2(s, OUT / "README_upstream.md")
                continue
            shutil.copy2(s, OUT / f)
            copied.append(f)
    (OUT / "quant.json").write_text(json.dumps({
        "scheme": f"int{args.bits}-sym-per-out-channel" if args.bits else "none(bf16)",
        "bits": args.bits,
        "quantized_keys": quantized,
        "scale_suffix": ".__scale__",
        "vision_tower": "dropped" if not args.keep_vision else "kept",
        "mtp": "dropped",
        "note": "transformers 的 from_pretrained 读不了 int8；请用同目录的 load_te.py",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"附属文件 {len(copied) + 2} 个（config.json / quant.json / {', '.join(copied[:4])} …）")
    shutil.copy2(Path(__file__).with_name("_te_slim_loader.py"), OUT / "load_te.py")

    total = sum(q.stat().st_size for q in OUT.iterdir() if q.is_file())
    (OUT / "README.md").write_text(f"""# Qwen3.5-0.8B 文本编码器 · 精简{' + int8' if args.bits else ''}

从 `Qwen3.5-0.8B` 抽出**文本塔**并{'做 int8 量化' if args.bits else '保持 bf16'}，
供 Aquarius 推理端当文本编码器用。

| | 体积 |
|---|---|
| 原版（含视觉塔 + mtp） | **1746.9 MB** |
| **本产物** | **{total / MB:.0f} MB** |

## 怎么用

⚠️ **`transformers` 的 `from_pretrained` 读不了 int8 权重**（dtype 不匹配），
必须用同目录自带的 `load_te.py`：

```python
import sys; sys.path.insert(0, r"<本目录>")
from load_te import load_te

tok, model = load_te(r"<本目录>")                 # 默认 mode="int8"：权重常驻 int8，省内存
tok, model = load_te(r"<本目录>", mode="bf16")     # 加载时全部还原成 bf16，省 CPU
```

编码口径**必须**与训练时一致（裸提示词 + `max_length=32` + `padding="max_length"` +
`truncation`，取 `last_hidden_state`，不加 chat template / 不加前缀）：

```python
b = tok(["a cheese pizza cut into many slices on a table"], return_tensors="pt",
        max_length=32, truncation=True, padding="max_length")
ctx = model(**b).last_hidden_state[0]      # (32, 1024)
```

## 两种模式（2026-10-03 实测，本机 Intel Xeon 6530）

| | `mode="bf16"` | `mode="int8"`（默认） |
|---|---|---|
| 权重常驻形态 | bf16 | **int8 + 每行一个 f32 scale** |
| **进程 RSS** | 3405.8 MB | **756.0 MB** |
| 加载耗时 | 11.4 s | 11.3 s |
| 单条 caption 编码 | 0.34 s | 0.89 s |
| 适合 | 内存充裕 | **内存紧张（本机机械硬盘场景）** |

**两种模式数学结果逐位相同**（实测最大逐元素差 = **0**）—— 所以按内存挑就行，不用担心精度。

`int8` 模式的两个关键设计：
* **`torch.device("meta")` 建空壳**：`from_config()` 在 CPU 上给 853M 参数做随机初始化要
  **实测 99.6 秒**，meta 只要 1.6 秒 —— 光这一项就值 60 倍（112 s → 12 s）。
* **embedding 表只还原被 gather 到的那 32 行**：它是 248,320×1024（占整个文件 31%），
  整表还原是最大的一笔浪费，按行取就几乎零开销。

## 验证（`code/_t_te_slim_verify.py`）

* 10 条 caption 的 ctx cos：**平均 {TARGET_COS}**
* **端到端出图 L1 vs 原版 bf16：{L1S}**（同 seed、同提示词，只换 TE）

## 限制

* **省的是磁盘与内存，不是显存** —— 显存那边本项目的 `AQ_TE_MODE=cache` 早就不让 TE 常驻。
* 只验证了 5 条 caption / 1 个 seed / step 150000 这一个检查点；
  且该模型**仍欠训练**（clip_ratio ≈ 0.34）⇒ 「现在看不出损伤」≠「训练完也看不出」。
* 训练集的 caption embedding 是**原版 bf16 TE** 预计算的；换用本产物推理端条件差
  L1≈1.9（实测无关紧要），真上生产建议训练完成后复验。

## 与 GPTQ/AWQ 的关系

本产物是**朴素 RTN（round-to-nearest）per-out-channel int8**，**没有用校准数据**。
带校准的 8-bit（GPTQ/AWQ/SmoothQuant）理论上还能更好，本项目**未测**。
""", encoding="utf-8")

    total = sum(p.stat().st_size for p in OUT.iterdir() if p.is_file())
    print(f"\n✅ 产物目录总大小 {total / MB:.0f} MB  →  {OUT}")
    print(f"   原版 {src_f.stat().st_size / MB:.0f} MB ⇒ 省 {(1 - total / src_f.stat().st_size) * 100:.0f}%")
    print(f"\n加载方式（transformers 直读不了 int8）：")
    print(f"    import sys; sys.path.insert(0, r'{OUT}')")
    print(f"    from load_te import load_te")
    print(f"    tok, model = load_te(r'{OUT}')")
    return 0


if __name__ == "__main__":
    sys.exit(main())
