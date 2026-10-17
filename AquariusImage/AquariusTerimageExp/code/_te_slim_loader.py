# -*- coding: utf-8 -*-
"""随产物一起分发的加载器。**transformers 的 `from_pretrained` 读不了 int8 权重**。

本文件自包含：只需要 `torch` + `transformers` + `safetensors`，**不依赖 torchao / bitsandbytes**。

    import sys; sys.path.insert(0, r"<产物目录>")
    from load_te import load_te
    tok, model = load_te(r"<产物目录>")               # 默认 mode="int8"：权重常驻 int8，省内存
    tok, model = load_te(r"<产物目录>", mode="bf16")   # 加载时全部还原成 bf16，省 CPU

    # 与 aq_play.encode_texts 完全同口径：
    b = tok(["a cheese pizza cut into many slices on a table"], return_tensors="pt",
            max_length=32, truncation=True, padding="max_length")
    ctx = model(**b).last_hidden_state[0]      # (32, 1024)

## 两种模式

| | `mode="bf16"` | `mode="int8"`（默认） |
|---|---|---|
| 权重在内存里 | bf16 | **int8 + 每行一个 f32 scale** |
| 构建方式 | 正常建模型，**逐张量就地填充** | **`torch.device("meta")` 建空壳** + int8 模块替换 |
| 每层矩阵乘 | 直接 bf16 | 前向时**临时**还原该层再乘 |
| 适合 | 内存充裕、要最低 CPU 开销 | **内存紧张 / 机械硬盘机器** |

**embedding 表是最大收益点**：248,320×1024，占整个文件 31%。
`int8` 模式下**只还原被 gather 到的那 32 行**，不是整表 ⇒ 几乎零开销却省 ~262.1 MB。

⚠️ 两种模式**数学结果完全相同**（都是「int8 → dtype，再乘」），实测 ctx cos ≈ 1.0。

## 踩过的坑（都别再踩）

1. **`from_config + load_state_dict` 不做前缀映射**：checkpoint 的键是 `model.language_model.*`，
   而 `Qwen3_5Model.state_dict()` 是 `language_model.*`。`from_pretrained` 会自动去前缀，
   `from_config` **不会** ⇒ 全部键落空、**模型静默变成随机初始化**（cos 从 0.995 掉到 −0.012，零报错）。
2. **「没有 meta 张量剩下」的检查必须在 `model.visual = None` 之后做** ——
   视觉塔的权重我们本来就不发，它会一直保持 meta，否则会被误判成失败。
3. **不要在真实设备上先建一遍模型再换 int8** —— 峰值内存会先涨到 bf16 那一份，等于没省。
"""
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


# --------------------------------------------------------------------------- #
class _Int8Linear(nn.Module):
    """权重常驻 int8；前向时把**这一层**临时还原成激活的 dtype 再乘。"""

    def __init__(self, q, s, bias=None):
        super().__init__()
        self.register_buffer("q", q, persistent=False)     # int8 [out, in]
        self.register_buffer("s", s, persistent=False)     # fp32 [out]
        self.bias = bias
        self.out_features, self.in_features = q.shape

    def forward(self, x):
        # ⚠️ **在 fp32 里乘**，与 `_dequant` 完全一致 ⇒ 与 mode="bf16" 数值逐位相同。
        #    用 bf16 乘（scale 只有 8 位尾数）会产生可测的微小差异（实测 std 4.1746 vs 4.1772）。
        w = (self.q.to(torch.float32) * self.s.unsqueeze(-1)).to(x.dtype)
        return F.linear(x, w, self.bias)

    @property
    def weight(self):
        """兼容性：模型代码里可能有 `mod.weight` 这类内省。
        注意它**每次访问都会还原一份**，所以只应该被内省用到，别进热路径。"""
        return (self.q.to(torch.float32) * self.s.unsqueeze(-1)).to(self.s.dtype)


class _Int8Embedding(nn.Module):
    """同 `_Int8Linear`，但**只还原被 gather 到的那几行** —— 本方案最大的一笔收益。"""

    def __init__(self, q, s, dtype):
        super().__init__()
        self.register_buffer("q", q, persistent=False)
        self.register_buffer("s", s, persistent=False)
        self.num_embeddings, self.embedding_dim = q.shape
        self.out_dtype = dtype

    def forward(self, idx):
        return (self.q[idx].to(torch.float32) * self.s[idx].unsqueeze(-1)).to(self.out_dtype)

    @property
    def weight(self):
        """兼容性：见 `_Int8Linear.weight` 的说明。"""
        return (self.q.to(torch.float32) * self.s.unsqueeze(-1)).to(self.s.dtype)


# --------------------------------------------------------------------------- #
def _dequant(w, s, dtype, chunk_rows=8192):
    """int8 + 每行 scale → dtype。**分块做**，避免大张量产生一整份 fp32 临时副本。"""
    if w.dtype != torch.int8:
        return w.to(dtype)
    sr = s.reshape(-1, *([1] * (w.dim() - 1)))
    out = torch.empty(w.shape, dtype=dtype)
    for i in range(0, w.shape[0], chunk_rows):
        out[i:i + chunk_rows] = (w[i:i + chunk_rows].to(torch.float32)
                                 * sr[i:i + chunk_rows]).to(dtype)
    return out


def load_te(directory, device="cpu", dtype=torch.bfloat16, mode="int8",
            verbose=True):
    """加载精简(+量化)过的文本编码器，返回 (tokenizer, model)。

    mode="int8" —— 权重常驻 int8（省内存，默认）
    mode="bf16" —— 加载时全部还原成 bf16（省 CPU，吃内存）
    """
    t_start = time.time()
    d = Path(directory)
    qj = json.loads((d / "quant.json").read_text(encoding="utf-8"))
    suffix = qj.get("scale_suffix", ".__scale__")
    if qj.get("bits", 0) == 0:
        mode = "bf16"
    wf = next((p for p in (d / "model_int8.safetensors", d / "model.safetensors")
               if p.is_file()), None)
    if wf is None:
        raise FileNotFoundError(f"{d} 里找不到 model_int8.safetensors / model.safetensors")

    # ---- 读权重（int8 常驻，体积小；这是唯一的完整副本）----
    from safetensors import safe_open
    sd, scales = {}, {}
    with safe_open(str(wf), framework="pt") as h:
        for k in h.keys():
            if k.endswith(suffix):
                scales[k[: -len(suffix)]] = h.get_tensor(k)
            else:
                sd[k] = h.get_tensor(k)
    t_read = time.time() - t_start

    from transformers import AutoConfig, AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(d), trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(str(d), trust_remote_code=True)

    # ---- 前缀对齐（坑 1）----
    with torch.device("meta"):
        probe = AutoModel.from_config(cfg, dtype=dtype)
    expected = set(probe.state_dict().keys())
    del probe
    dropped = ""
    if not (set(sd) & expected):
        for pre in ("model.", "base_model.model.", "model.model."):
            cand = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
            if cand and (set(cand) & expected):
                sd = cand
                scales = {k[len(pre):]: v for k, v in scales.items()
                          if k.startswith(pre)}
                dropped = pre
                break
    quant = set(scales)
    if verbose:
        print(f"[load_te] {d.name}: {len(sd)} 张量 · {len(quant)} 张 int8 · "
              f"bits={qj.get('bits')} · 视觉塔={qj.get('vision_tower')} · "
              f"mtp={qj.get('mtp')}"
              + (f" · 前缀已对齐（去掉 '{dropped}'）" if dropped else ""))

    # ---- 建模型：**两条路径都在 meta 上建** ----
    # ⚠️ 为什么必须 meta：`AutoModel.from_config()` 在 CPU 上给 853M 参数做随机初始化要
    #    **实测 99.6 秒**（meta 只要 1.6 秒）—— 光这一项就值 60 倍。
    with torch.device("meta"):
        model = AutoModel.from_config(cfg, dtype=dtype)

    if mode == "bf16":
        full = {k: (_dequant(v, scales[k], dtype) if k in quant else v.to(dtype))
                for k, v in sd.items()}
        sd.clear()
        missing_names, _unexpected = model.load_state_dict(full, strict=False, assign=True)
        del full
    else:
        rest = {k: v.to(dtype) for k, v in sd.items() if k not in quant}
        missing_names, _unexpected = model.load_state_dict(rest, strict=False, assign=True)
        del rest
        mods = dict(model.named_modules())
        swapped = conv_bf16 = 0
        for name in list(quant):
            if not name.endswith(".weight"):
                continue
            path = name[: -len(".weight")]
            m = mods.get(path)
            parent_path, _, leaf = path.rpartition(".")
            parent = mods.get(parent_path) if parent_path else model
            if parent is None or leaf not in dict(parent.named_children()):
                continue
            bq = sd.get(path + ".bias")
            if isinstance(m, nn.Embedding):
                setattr(parent, leaf, _Int8Embedding(sd[name], scales[name], dtype))
                swapped += 1
            elif isinstance(m, nn.Linear):
                bias = (None if bq is None
                        else nn.Parameter(bq.to(dtype), requires_grad=False))
                setattr(parent, leaf, _Int8Linear(sd[name], scales[name], bias))
                swapped += 1
            else:
                # ⚠️ 3D conv 之类：**只换模块里的 `.weight`，不要换掉整个模块** ——
                #    模型代码是按 `self.conv1d.weight` 访问的，把模块换成裸 Parameter
                #    会炸 `'Parameter' object has no attribute 'weight'`。
                #    这类张量极少（18 层 × 24,576）且极小 ⇒ 直接还原成 bf16。
                setattr(m, "weight",
                        nn.Parameter(_dequant(sd[name], scales[name], dtype),
                                     requires_grad=False))
                conv_bf16 += 1
        if verbose:
            print(f"[load_te] int8 常驻：替换 {swapped} 个 Linear/Embedding"
                  f"（另 {conv_bf16} 个 3D 卷积已还原成 bf16）")

    # ⚠️ **先摘视觉塔再查 meta**（踩过的坑）：视觉塔的权重我们本来就不发，
    #    它会一直保持 meta，否则会被误判成失败。
    if qj.get("vision_tower") == "dropped" and hasattr(model, "visual"):
        model.visual = None

    # ---- 补上 meta 的「初始化时现算、不进 checkpoint」的 buffer ----
    # 典型就是 RoPE 的 `inv_freq` / `original_inv_freq`（非持久 buffer ⇒ state_dict 里看不到，
    # 所以先前的检查漏了它，最后 `.to()` 才炸）。它们在 `__init__` 里由 config 确定性算出 ⇒
    # 用 `type(mod)(mod.config)` 在真实设备上重建一次再抄回来即可（实测可行）。
    fixed = 0
    for mname, mod in list(model.named_modules()):
        bs = [(n, b) for n, b in mod.named_buffers(recurse=False)]
        if not any(b.is_meta for _, b in bs):
            continue
        mcfg = getattr(mod, "config", None)
        if mcfg is None:
            bad = [n for n, b in bs if b.is_meta]
            raise RuntimeError(f"[load_te] {mname} 有 meta buffer {bad} 但取不到 config，"
                               f"无法重建")
        fresh = dict(type(mod)(mcfg).named_buffers(recurse=False))
        for bn, b in bs:
            if b.is_meta:
                if bn not in fresh:
                    raise RuntimeError(f"[load_te] 重建 {mname} 后仍无 buffer '{bn}'")
                mod._buffers[bn] = fresh[bn].clone()
        fixed += 1
    if verbose and fixed:
        print(f"[load_te] 重建了 {fixed} 个「现算 buffer」模块（RoPE 的 inv_freq 之类）")

    # 最终确认：文本路径上不该再有 meta
    left = [k for k, v in list(model.named_parameters()) + list(model.named_buffers())
            if v.is_meta]
    if left:
        raise RuntimeError(f"[load_te] 仍有 {len(left)} 个 meta 张量未填充（前 3：{left[:3]}）")

    # 两类"缺失"是**正常的**，不能当失败：
    #   ① `rotary_emb.inv_freq` / `original_inv_freq` —— 初始化时现算的 buffer，**本来就不在
    #      checkpoint 里**（原始权重文件也没有）。
    #   ② `int8` 模式下被我们换成 `_Int8Linear/_Int8Embedding` 的那批 `.weight` ——
    #      它们没走 `load_state_dict`，但已经被模块接管了。
    IGNORE = ("inv_freq",)
    real_missing = [k for k in missing_names
                    if k not in quant and not any(t in k for t in IGNORE)]
    fatal = [k for k in real_missing if "language_model" in k or "embed_tokens" in k]
    if fatal:
        raise RuntimeError(f"[load_te] 文本塔缺 {len(fatal)} 个权重（前 5：{fatal[:5]}）"
                           f" —— 前缀或量化文件有问题，拒绝静默返回随机初始化的模型")
    if verbose:
        if real_missing:
            print(f"[load_te] 缺 {len(real_missing)} 个（应全部属于视觉塔，不进文本路径）："
                  f"{real_missing[:3]} …")
        if missing_names:
            print(f"[load_te] 另有 {len(missing_names) - len(real_missing)} 个属正常缺失"
                  f"（inv_freq 现算 / 已被 int8 模块接管）")
    model = model.to(device).eval()
    if verbose:
        print(f"[load_te] 读权重 {t_read:.1f}s · 总计 {time.time() - t_start:.1f}s")
    return tok, model


if __name__ == "__main__":
    import os
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent)
    md = sys.argv[2] if len(sys.argv) > 2 else "int8"
    try:
        import psutil
        mb = lambda: psutil.Process(os.getpid()).memory_info().rss / 1048576
    except Exception:                                      # noqa: BLE001
        mb = lambda: float("nan")
    print(f"  起始 RSS {mb():.0f} MB")
    tok, model = load_te(d, mode=md)
    print(f"  [{md}] 加载完成 · RSS {mb():.0f} MB")
    cap = "a cheese pizza cut into many slices on a table"
    b = tok([cap], return_tensors="pt", max_length=32, truncation=True,
            padding="max_length").to(next(model.parameters()).device)
    t0 = time.time()
    with torch.no_grad():
        ctx = model(**b).last_hidden_state[0]
    print(f"  [{md}] ctx {tuple(ctx.shape)} {ctx.dtype} std {float(ctx.float().std()):.4f}"
          f" · 编码 {time.time() - t0:.2f}s")
