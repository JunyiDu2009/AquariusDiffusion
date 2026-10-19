# -*- coding: utf-8 -*-
"""AquariusCodaExp 一步去噪器 · Gradio 推理界面（M8）

功能（计划书 §10 验收 4）：选图 → 拖噪声滑块（实时预览加噪）→ 一键去噪 → 原生 ImageSlider 图上拖动对比。
- 对比控件用 gradio 原生 gr.ImageSlider（手柄嵌在分割线上，纯前端拖动，零闪烁、零自定义 JS）。
- 权重二选一：NVFP4 导出（A3 dequant 路径，核心交付）与 FP16 EMA（量化 gap 现场对照）。
- 输入任意尺寸：最长边超 2048 自动等比缩放；reflect pad 到 16 的倍数（模型 4 次下采样），输出裁回原尺寸。
- 加噪口径与训练一致：x_p = sqrt(1-p)·x0 + sqrt(p)·eps，p∈[1%,50%] 整数百分比。
"""
import glob
import math
import os
import sys
import time
from collections import deque

import gradio as gr
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import UNetDenoiser  # noqa: E402
from nvfp4 import dequantize_model  # noqa: E402

CKPT_DIR = "D:/AI_Library_D/AquariusCodaExp_tmp/ckpts/m2_formal"
VAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "val2017")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SIDE = 640  # 最长边上限：超过则等比缩放（LANCZOS/BICUBIC），宽高比不变。
# 640 落在训练桶分布内（最大桶宽 640），demo 与训练共享 8GB 显存也最稳


def latest_nvfp4():
    fs = sorted(glob.glob(os.path.join(CKPT_DIR, "step_*_nvfp4.pt")))
    return fs[-1] if fs else None


def to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1)[None]


def to_img(t: torch.Tensor) -> Image.Image:
    arr = ((t[0].detach().float().clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr)


def prepare(img: Image.Image, max_side: int = MAX_SIDE):
    """RGB 化 + 最长边限 max_side 等比缩放 + reflect pad 到 /16；返回 (1,3,H,W) 张量与 (w,h)。"""
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        s = max_side / max(w, h)
        img = img.resize((max(16, round(w * s)), max(16, round(h * s))), Image.LANCZOS)
        w, h = img.size
    x = to_tensor(img)
    pw, ph = (16 - w % 16) % 16, (16 - h % 16) % 16
    if pw or ph:
        x = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect")
    return x, (w, h)


def make_noise(shape, seed):
    g = torch.Generator(device="cpu").manual_seed(int(seed) % (2**31))
    return torch.randn(shape, generator=g)


def noisy_of(img: Image.Image, pct: float, seed: int):
    """与训练同口径的加噪图（预览与去噪共用同一 seed → 同一张 eps）。"""
    x, (w, h) = prepare(img)
    eps = make_noise(x.shape, seed)
    p = pct / 100.0
    xp = math.sqrt(1 - p) * x + math.sqrt(p) * eps
    return to_img(xp[:, :, :h, :w])


print(f"[demo] device={DEVICE}, loading checkpoints from {CKPT_DIR} ...", flush=True)
if DEVICE == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.35, 0)  # 推理进程小预算，不与训练抢显存

nv_path = latest_nvfp4()
NV_STEP, FP_STEP = "无导出", "?"
nv_model = UNetDenoiser().to(DEVICE).eval()
if nv_path:
    pkg = torch.load(nv_path, map_location="cpu", weights_only=False)
    nv_model.load_state_dict(dequantize_model(pkg))
    NV_STEP = os.path.basename(nv_path).replace("step_", "").replace("_nvfp4.pt", "")
    print(f"[demo] NVFP4 loaded: {os.path.basename(nv_path)}", flush=True)

fp16_model = UNetDenoiser().to(DEVICE).eval()
_fp_path = os.path.join(CKPT_DIR, "last_fp16.pt")
ck = None
for _ in range(4):  # 训练进程每 1000 步重写该文件，撞上写入瞬间会读到半截 → 重试
    try:
        ck = torch.load(_fp_path, map_location="cpu", weights_only=False)
        break
    except Exception as e:
        print(f"[demo] last_fp16.pt 读取失败（{type(e).__name__}），2s 后重试...", flush=True)
        time.sleep(2)
if ck is not None:
    fp16_model.load_state_dict(ck["ema"])  # EMA 权重，与 NVFP4 导出同源
    FP_STEP = str(ck.get("step", "?"))
else:  # 兜底：用最新 NVFP4 反量化权重代替（同源 EMA，质量几乎一致）
    fp16_model.load_state_dict(dequantize_model(pkg))
    FP_STEP = NV_STEP + " (dequant)"
    print("[demo] last_fp16.pt 不可用，FP16 面板改用 NVFP4 反量化权重", flush=True)
print(f"[demo] FP16 EMA loaded: step {FP_STEP}", flush=True)

WEIGHTS = [f"NVFP4 (step {NV_STEP})", f"FP16 EMA (step {FP_STEP})"]

# 文件名含 '#' 时（如 #三叶草#xxx.jpeg），gradio 生成的预览 src 未转义 → 浏览器当锚点截断 → 输入框黑图。
# 服务端对 %23 编码路径实测返回 200，故在上传事件里把 img src 的 '#' 转义成 '%23'（gradio 原生 js 回调，可靠执行）。
FIX_SRC_JS = """
(noisy) => {
  document.querySelectorAll('img').forEach(i => {
    if (i.src && i.src.indexOf('#') !== -1 && i.src.indexOf('%23') === -1 && i.src.indexOf('/gradio_api/file=') !== -1) {
      const parts = i.src.split('#');
      i.src = parts[0] + '%23' + parts.slice(1).join('%23');
    }
  });
  return noisy;
}
"""

LOG = deque(maxlen=8)  # 最近 8 条推理记录（新在上）


@torch.no_grad()
def _denoise_impl(img, pct, seed, weight, max_side=MAX_SIDE):
    if img is None:
        return None, ("<br>".join(LOG) if LOG else "先在左侧选一张图")
    t0 = time.perf_counter()
    ow, oh = img.size
    x, (w, h) = prepare(img, max_side)
    x = x.to(DEVICE)
    eps = make_noise(x.shape, seed).to(DEVICE)
    p = torch.tensor([pct / 100.0], device=DEVICE)
    xp = (1 - p).sqrt() * x + p.sqrt() * eps
    model = nv_model if str(weight).startswith("NVFP4") else fp16_model
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
        out = model(xp, p)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t2 = time.perf_counter()
    result = to_img(out.float()[:, :, :h, :w])
    # 对比对齐关键：左边不放大原图，而是放「与推理同几何」的等比缩放图——
    # 两张图同为 (w,h)，ImageSlider 叠加才完全对齐（原始大图会各自 letterbox 错位）
    orig_fit = img.convert("RGB").resize((w, h), Image.BICUBIC)
    total_ms = (time.perf_counter() - t0) * 1000
    fwd_ms = (t2 - t1) * 1000
    scale_note = f"{ow}x{oh}" + (f" → 等比缩放 {w}x{h}" if (w, h) != (ow, oh) else "")
    LOG.insert(0, f"{time.strftime('%H:%M:%S')} | {scale_note} | p={pct:.0f}% | seed={int(seed)} | {weight} | **前向 {fwd_ms:.1f} ms** | 端到端 {total_ms:.0f} ms")
    return (orig_fit, result), "<br>".join(LOG)


@torch.no_grad()
def denoise(img, pct, seed, weight):
    """OOM 自适应：显存不足自动降分辨率重试（与训练共享 8GB 显存的环境下必要）。"""
    scales = [MAX_SIDE, 512, 384]
    try:
        for i, ms in enumerate(scales):
            try:
                return _denoise_impl(img, pct, seed, weight, max_side=ms)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                if i < len(scales) - 1:
                    LOG.insert(0, f"{time.strftime('%H:%M:%S')} | ⚠️ 显存不足，自动降分辨率至 ≤{scales[i + 1]}px 重试")
        LOG.insert(0, f"{time.strftime('%H:%M:%S')} | ❌ 显存不足：各分辨率档位均失败；建议稍后再试或关闭占显存的程序")
        return None, "<br>".join(LOG)
    finally:
        torch.cuda.empty_cache()  # 不同尺寸的瞬态缓冲及时归还，防碎片累积


def refresh_preview(img, pct, seed):
    if img is None:
        return None
    return noisy_of(img, float(pct), int(seed))


FIX_CSS = """
/* gradio 6.19 ImageSlider bug：右侧叠加图自带 margin-top:-40px（gradio 为自家头部设计），
   在本布局中导致两帧垂直错位 40px。归零并强制与底层图同盒锚定。 */
div.wrap div.content { position: relative !important; }
div.wrap img.fixed { position: absolute !important; top: 0 !important; left: 0 !important; margin-top: 0 !important; }
"""

with gr.Blocks(title="AquariusCodaExp 一步去噪", css=FIX_CSS) as demo:
    gr.Markdown(
        f"### AquariusCodaExp · 0.08B UNet 一步去噪器（M8 Demo）\n"
        f"权重快照：NVFP4 step {NV_STEP} · FP16 EMA step {FP_STEP}（训练进行中，画质随训练推进）· "
        f"加噪口径 p=噪声方差占比，与训练一致"
    )
    with gr.Row():
        with gr.Column(scale=5):
            inp = gr.Image(label="输入图（任意尺寸；最长边超 640 自动等比缩放到 640；pad 到 /16）", type="pil", height=340)
            gr.Examples(
                examples=sorted(glob.glob(os.path.join(VAL_DIR, "*.jpg")))[:4],
                inputs=[inp],
                label="示例图（val2017）",
            )
        with gr.Column(scale=4):
            pct = gr.Slider(1, 50, value=10, step=1, label="噪声强度 p（%）", info="拖动实时预览加噪")
            seed = gr.Slider(0, 9999, value=0, step=1, label="随机种子", info="同 seed 同噪声，去噪与预览一致（拖动或点数字框直接输入）")
            noisy = gr.Image(label="加噪预览", height=240, interactive=False)
    with gr.Row():
        weight = gr.Radio(WEIGHTS, value=WEIGHTS[0], label="权重")
        btn = gr.Button("一键去噪", variant="primary", scale=0)
    gr.Markdown("**原图 / 去噪 对比**（拖动图上的分割线）")
    out = gr.ImageSlider(
        type="pil", height=420, show_label=False, container=False,
    )
    gr.Markdown("**推理日志**（最近 8 条，新在上）")
    log_md = gr.Markdown("尚无推理记录")

    inp.change(refresh_preview, [inp, pct, seed], [noisy], js=FIX_SRC_JS).then(
        lambda: None, None, [out]
    )
    pct.change(refresh_preview, [inp, pct, seed], [noisy])
    seed.change(refresh_preview, [inp, pct, seed], [noisy])
    btn.click(denoise, [inp, pct, seed, weight], [out, log_md])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=8792, inbrowser=False, show_error=True)
