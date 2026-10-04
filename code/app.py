# -*- coding: utf-8 -*-
"""Aquarius Terimage -- Gradio web UI (bilingual: Chinese / English).

The whole interface is bilingual and switches with a single button in the top
right.  The run log is localised too: the engine emits precise English status
lines, and `_loc()` maps them to Chinese when the UI is in Chinese mode, so the
log reads naturally in both languages without duplicating the engine's strings.

    python app.py                  # auto-opens http://127.0.0.1:7860
    python app.py --port 8888      # different port
    python app.py --lowvram        # cap peak VRAM below 3 GB (reloads every time, slow)
    python app.py --no-browser     # do not auto-open a browser
    python app.py --share          # create a temporary public link (needs gradio.app reachable)
    python app.py --self-test      # no UI, pure CLI self-check (gradio not required)

VRAM strategy (see aq_play.Engine for details):
    UNet 2.23 GB + VAE 0.32 GB stay resident, the text encoder is unloaded right
    after use -> peak about 4.2 GB, plenty of headroom on an 8 GB card; hitting
    "Generate" a second time does not reload any model.
    With --lowvram everything is reloaded each time, peak about 2.6 GB.

NOTE: the bundled checkpoint is step 260,000 = 28.70 epochs.  Output is BLURRY
SCENE COMPOSITION (you can see sky / ground layering) with NO recognizable
objects.  That is the expected under-trained state, not a problem with your
environment or with these scripts.
"""
import argparse
import os
import queue
import re
import sys
import threading
import time
from pathlib import Path

# Gradio does a version check / telemetry ping on startup, which is slow when
# the network is blocked. Turn it off.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))          # make "import aq_play / aq_unet" CWD-independent

import aq_play as ap                                        # noqa: E402

OUT_DIR = ap.OUT_DIR

EXAMPLES = """sunset over the ocean
a cat sitting on a chair
blue sky over mountains
a busy street with cars at night
a bowl of fruit on a wooden table"""

# --------------------------------------------------------------------------
# 运行日志自动跟随到底部（tail -f 行为）。
# ⚠️ 不用 MutationObserver：Gradio 是通过 JS 改 textarea.value，不产生 DOM 变动，观察不到；
#    轮询 scrollHeight 才可靠。且**只在用户本来就贴着底部时**跟随 —— 手动往上翻看历史时
#    不会被拽回去。
# ⚠️ Gradio 6 里 `head` 是 launch() 的参数，不再是 Blocks() 的了（实测会告警）。
# --------------------------------------------------------------------------
AUTOSCROLL_JS = """
<script>
(function () {
  function follow() {
    var host = document.getElementById('aq-log');
    if (!host) return;
    var ta = host.querySelector('textarea');
    if (!ta) return;
    var atBottom = (ta.scrollHeight - ta.scrollTop - ta.clientHeight) < 48;
    if (ta._aqLast === undefined) { ta._aqLast = ta.scrollHeight; atBottom = true; }
    if (atBottom && ta.scrollHeight !== ta._aqLast) {
      ta._aqLast = ta.scrollHeight;
      ta.scrollTop = ta.scrollHeight;
    }
  }
  setInterval(follow, 250);
  document.addEventListener('DOMContentLoaded', follow);
})();
</script>
"""


# --------------------------------------------------------------------------
# Bilingual strings.  Every key must exist in BOTH languages.
# --------------------------------------------------------------------------
T = {
    "zh": {
        "btn_lang": "🌐 English",
        "header": (
            "# Aquarius Terimage · 原生三值（ternary）文生图\n"
            "**558M 参数，权重从第 0 步起就活在量化空间（NLT）。**"
            "本包内置 **step 260,000 = 28.70 轮**的最终检查点。出图是**模糊的场景构图**"
            "（能看出天空／地面分层、色调被提示词驱动），但**认不出具体物体** —— "
            "这是欠训练的预期表现，不是环境问题。"
        ),
        "prompt": "提示词（一行一条；多条会依次生成）",
        "steps": "采样步数（20 够用；越多越慢）",
        "size_acc": "尺寸（宽 × 高）",
        "preset": "训练用过的 22 个桶（选一个，或直接拖下面的滑块）",
        "width": "宽", "height": "高",
        "seed": "随机种子", "seed_info": "-1 = 每次随机",
        "per_prompt": "每个提示词生成几张",
        "lowvram": "低显存模式（每次重新加载模型，峰值 <3GB，慢）",
        "generate": "生成", "free": "释放显存",
        "tips": (
            "**小技巧**：它对**色调 + 空间布局**有响应，所以写 `sunset over the ocean` "
            "比写 `a cat` 更容易看出东西。**同一个种子 + 同一个提示词 = 同一张图**（确定性 DDIM）。\n\n"
            "**尺寸**：训练是在 22 个不同长宽比的桶上做的（`640` 封顶），横图／竖图都支持 —— "
            "挑一个训练用过的桶通常比正方形更好看。\n\n"
            "> 该模型训练时**没有启用 CFG**，所以没有「负面提示词」和「引导强度」这两个参数。"
        ),
        "log": "运行日志", "results": "生成结果", "outdir": "输出目录",
        "info_acc": "模型信息", "inspect": "查看", "open_dir": "打开输出目录",
        "custom": "自定义（用下面的滑块）",
        "ori_sq": "方", "ori_land": "横", "ori_port": "竖",
        "not_bucket": "⚠️ 非训练桶",
        "ok_bucket": "✅ 这是训练用过的 22 个桶之一",
        "warn_busy": "⚠️ 上一个任务还在跑，等它结束再点。",
        "warn_noprompt": "⚠️ 请至少输入一条提示词。",
        "err_prefix": "❌ 失败：",
        "done_prefix": "✅ 完成",
        "done_suffix": "张 →",
        "images": "张",
        "reload_note": "（下次生成会重新加载模型，第一次会慢一些）",
    },
    "en": {
        "btn_lang": "🌐 中文",
        "header": (
            "# Aquarius Terimage - native ternary text-to-image\n"
            "**558M parameters, weights live in the quantized space from step 0 (NLT).** "
            "This bundle ships the final checkpoint at **step 260,000 = 28.70 epochs**. "
            "Output is **blurry scene composition** (sky / ground layering and colour tone "
            "do follow the prompt) with **no recognizable objects** -- the expected "
            "under-trained behaviour, not an environment problem."
        ),
        "prompt": "Prompts (one per line; several lines generate in sequence)",
        "steps": "Sampling steps (20 is enough; more is slower)",
        "size_acc": "Size (width x height)",
        "preset": "Trained buckets (pick one, or drag the sliders)",
        "width": "Width", "height": "Height",
        "seed": "Seed", "seed_info": "-1 = random each time",
        "per_prompt": "Images per prompt",
        "lowvram": "Low VRAM mode (reloads models each time, peak <3 GB, slow)",
        "generate": "Generate", "free": "Free VRAM",
        "tips": (
            "**Tip**: it responds to *colour tone + spatial layout*, so "
            "`sunset over the ocean` shows more structure than `a cat`. "
            "**Same seed + same prompt = same image** (deterministic DDIM).\n\n"
            "**Size**: training used 22 different aspect-ratio buckets (capped at `640`), "
            "so landscape and portrait both work -- picking a trained bucket usually looks "
            "better than a square.\n\n"
            "> This model was trained **without CFG**, so there are no negative-prompt or "
            "guidance-scale controls."
        ),
        "log": "Run log", "results": "Results", "outdir": "Output directory",
        "info_acc": "Model info", "inspect": "Inspect", "open_dir": "Open output directory",
        "custom": "Custom (use the sliders below)",
        "ori_sq": "square", "ori_land": "landscape", "ori_port": "portrait",
        "not_bucket": "⚠️ not a trained bucket",
        "ok_bucket": "✅ one of the 22 trained buckets",
        "warn_busy": "⚠️ The previous job is still running; wait for it to finish.",
        "warn_noprompt": "⚠️ Enter at least one prompt.",
        "err_prefix": "❌ failed: ",
        "done_prefix": "✅ done,",
        "done_suffix": "image(s) ->",
        "images": "image(s)",
        "reload_note": "(the next generation reloads the models; the first one is slower)",
    },
}


# --------------------------------------------------------------------------
# Run-log localisation.  The engine emits precise English status lines; rather
# than duplicating every string inside aq_play.py we translate them here.  Each
# rule is (compiled regex, Chinese template) -- group numbering is preserved.
# --------------------------------------------------------------------------
_LOG_RULES = [
    (re.compile(r"^no seed given, picked random (\d+)$"), "未指定种子，随机取 \\1"),
    (re.compile(r"^device (.+?) - cache=(\S+)$"), "设备 \\1 · 显存模式 cache=\\2"),
    (re.compile(r"^checkpoint (.+?) - step=(.+)$"), "检查点 \\1 · step \\2"),
    (re.compile(r"^size (\d+)x(\d+) \(width x height\) -> latent (\d+)x(\d+)(.*)$"),
     "尺寸 \\1×\\2（宽×高）→ latent \\3×\\4\\5"),
    (re.compile(r"^(\[\d+/\d+\]) encoding text \(TE (.+?)\) \.\.\.$"),
     "\\1 编码文本（TE \\2）…"),
    (re.compile(r"^(\[\d+/\d+\]) sampling (\d+/\d+) - (\d+) steps - seed (\d+) - (.+)$"),
     "\\1 采样 \\2 · \\3 步 · seed \\4 · \\5"),
    (re.compile(r"^(\[\d+/\d+\]) decoding image \.\.\.$"), "\\1 解码图像 …"),
    (re.compile(r"^(\[\d+/\d+\]) saved (.+)$"), "\\1 已保存 \\2"),
    (re.compile(r"^done, (\d+) image\(s\) in ([\d.]+) s(.*)$"),
     "完成 \\1 张，用时 \\2 秒\\3"),
    (re.compile(r"^\((\d+) OOM downgrade\(s\) along the way\)$"),
     "（途中 OOM 降级 \\1 次）"),
    (re.compile(r"^\(a trained bucket\)$"), "（训练过的桶）"),
]


# Trailing fragments that appear *inside* other lines, so a whole-line regex
# cannot catch them.  Applied after the rules above.
_FRAG = [
    ("(a trained bucket)", "（训练过的桶）"),
    (" OOM downgrade(s) along the way", " 途中 OOM 降级"),
    ("s/image", "秒/张"),
]


def _loc(line, lang):
    """Chinese-ify an engine status line when the UI is in Chinese mode."""
    if lang != "zh":
        return line
    out = line
    for rx, zh in _LOG_RULES:
        m = rx.match(out)
        if m:
            tmp = zh
            for i in range(1, (m.re.groups or 0) + 1):
                tmp = tmp.replace(f"\\{i}", m.group(i) or "")
            out = tmp
            break
    for a, b in _FRAG:
        out = out.replace(a, b)
    return out


# --------------------------------------------------------------------------
# Size presets: reuse the 22 buckets the training actually used.
# TRAINED_BUCKETS stores (H, W); the UI always shows "width x height".
# --------------------------------------------------------------------------
def _ori(w, h, lang="en"):
    t = T[lang]
    return t["ori_sq"] if w == h else (t["ori_land"] if w > h else t["ori_port"])


def _label(w, h, lang="en"):
    return f"{w} x {h} ({_ori(w, h, lang)})"


def _preset_list(lang="en"):
    """Squares first, then landscape, then portrait; each by ascending area."""
    hs = sorted(ap.TRAINED_BUCKETS, key=lambda b: (b[1] != b[0], b[1] * b[0]))
    sq = [b for b in hs if b[1] == b[0]]
    land = [b for b in hs if b[1] > b[0]]
    port = [b for b in hs if b[1] < b[0]]
    return [T[lang]["custom"]] + [_label(w, h, lang) for (h, w) in sq + land + port]


PRESET = {lg: _preset_list(lg) for lg in ("zh", "en")}
# label -> (h, w), per language
L2WH = {lg: {_label(w, h, lg): (h, w) for (h, w) in ap.TRAINED_BUCKETS}
        for lg in ("zh", "en")}

_LOCK = threading.Lock()          # only one generation at a time (VRAM would stack)
_LAST = {"gallery": []}           # keep the previous gallery so streaming does not flicker


def size_note(w, h, lang="en"):
    """Show whether this size is legal / was ever trained, as soon as a slider moves."""
    w, h = int(w), int(h)
    t = T[lang]
    ok, note = ap.check_res(h, w)
    head = f"**{w} × {h}**"
    if not ok:
        return head + f"\n\n❌ {note.strip()}"
    if (h, w) in ap.TRAINED_BUCKETS:
        return head + "\n\n" + t["ok_bucket"]
    return head + "\n\n" + t["not_bucket"]


def _on_preset(label, cur_w, cur_h, lang="en"):
    """Pick a preset -> push the values into the sliders. 'Custom' leaves them alone."""
    if label in L2WH[lang]:
        h, w = L2WH[lang][label]
        return w, h
    return cur_w, cur_h


def _resize_note(w, h, lang="en"):
    """Slider handler: same as size_note but takes the language state so the note
    follows the active UI language."""
    return size_note(w, h, lang)


def toggle_state(cur_lang, cur_preset, cur_w, cur_h):
    """Language toggle.  Returns (new_lang, *gr.update(...)) -- one entry per
    component listed in the button's `outputs`, in the same order."""
    import gradio as gr

    new = "en" if cur_lang == "zh" else "zh"
    t = T[new]
    wh = L2WH[cur_lang].get(cur_preset)
    new_preset = _label(wh[1], wh[0], new) if wh else t["custom"]
    return (
        new,                                                   # lang_state
        gr.update(value=t["header"]),                          # header
        gr.update(value=t["btn_lang"]),                        # btn_lang
        gr.update(label=t["prompt"]),                          # prompt
        gr.update(label=t["steps"]),                           # steps
        gr.update(choices=PRESET[new], value=new_preset,       # preset
                  label=t["preset"]),
        gr.update(label=t["width"]),                           # width
        gr.update(label=t["height"]),                          # height
        gr.update(value=size_note(int(cur_w), int(cur_h), new)),   # size_info
        gr.update(label=t["seed"], info=t["seed_info"]),       # seed
        gr.update(label=t["per_prompt"]),                      # per_prompt
        gr.update(label=t["lowvram"]),                         # lowvram
        gr.update(value=t["generate"]),                        # btn
        gr.update(value=t["free"]),                            # btn_rel
        gr.update(value=t["tips"]),                            # tips
        gr.update(label=t["log"]),                             # log
        gr.update(label=t["results"]),                         # gallery
        gr.update(label=t["outdir"]),                          # outdir
        gr.update(value=t["inspect"]),                         # btn_probe
        gr.update(value=t["open_dir"]),                        # btn_open
    )



# --------------------------------------------------------------------------
# Generation callback (a generator: streams log lines to the UI while running)
# --------------------------------------------------------------------------
def make_handlers(engine):
    """The callbacks take the hidden language Textbox as an explicit input (`lang`),
    so the log lines and the validation messages come out in the active language.

    ⚠️ Do NOT read `lang_state.value` off the component object instead: Gradio keeps
    a component's live value in the frontend/session and never writes it back to the
    Python object, so `.value` stays at its *initial* value forever.  Measured
    2026-10-04 with gradio 6.19: after a toggle set the box to "en", the object still
    reported "zh".
    """

    def on_generate(prompt_text, steps, width, height, seed, per_prompt, lowvram, lang):
        lg = lang if lang in T else "en"
        if not _LOCK.acquire(blocking=False):
            yield T[lg]["warn_busy"], _LAST["gallery"], ""
            return

        box = []

        def say(msg):
            box.append(str(msg))
            print("[ui]", msg, flush=True)

        prompts = [l.strip() for l in (prompt_text or "").splitlines() if l.strip()]
        if not prompts:
            _LOCK.release()
            yield T[lg]["warn_noprompt"], _LAST["gallery"], ""
            return

        q = queue.Queue()
        holder = {}

        def work():
            try:
                holder["res"] = engine.generate(
                    prompts, steps=int(steps), h=int(height), w=int(width),
                    seed=int(seed),
                    per_prompt=int(per_prompt), low_vram=bool(lowvram),
                    on_status=lambda m: q.put(("log", m)))
            except Exception as e:                          # noqa: BLE001
                holder["err"] = f"{type(e).__name__}: {e}"
            finally:
                q.put(("done", None))

        th = threading.Thread(target=work, daemon=True)
        th.start()
        try:
            while True:
                try:
                    kind, msg = q.get(timeout=0.4)
                except queue.Empty:
                    yield "\n".join(box[-60:]), _LAST["gallery"], ""
                    continue
                if kind == "done":
                    break
                say(_loc(str(msg), lg))
                yield "\n".join(box[-60:]), _LAST["gallery"], ""
            th.join()

            if "err" in holder:
                say(T[lg]["err_prefix"] + holder["err"])
                yield "\n".join(box[-60:]), _LAST["gallery"], ""
                return

            res_list = holder.get("res") or []
            gallery = [(p, f"seed {sd} | {pt[:48]}")
                       for (pt, sd, p) in res_list]
            if gallery:
                _LAST["gallery"] = gallery
            say(f"{T[lg]['done_prefix']} {len(gallery)} {T[lg]['images']} -> {OUT_DIR}")
            yield ("\n".join(box[-60:]), _LAST["gallery"],
                   f"{OUT_DIR}\n{len(gallery)} {T[lg]['images']}")
        finally:
            _LOCK.release()

    def on_release(lang):
        lg = lang if lang in T else "en"
        msg = engine.release()
        _LAST["gallery"] = []
        return _loc(msg, lg) + "\n" + T[lg]["reload_note"], [], ""

    def on_probe():
        if engine.step is None:
            step, src = engine.peek_ckpt()
        else:
            step, src = engine.step, "(already loaded)"
        vram = engine.vram()
        lines = [
            f"Device: {engine.device_info()}",
            f"Model file: {engine.ckpt.name}",
            f"Training step: {step if step is not None else '(read failed)'}",
            f"VRAM mode: {'resident' if engine.cache else 'low VRAM'}",
            f"Text encoder: {engine.te_state()} (te_mode={engine.te_mode})",
        ]
        if vram is not None:
            lines.append(f"VRAM allocated: {vram:.2f} GB")
        if engine.oom_retries:
            lines.append(f"OOM downgrades: {engine.oom_retries}")
        lines.append("Text encoder contract: Qwen3.5-0.8B (32 tokens x 1024)")
        lines.append(f"Output dir: {OUT_DIR}")
        return "\n".join(lines)

    def on_open_dir():
        import subprocess
        try:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            if sys.platform == "win32":
                os.startfile(str(OUT_DIR))                  # noqa: S606
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(OUT_DIR)])
            else:
                subprocess.Popen(["xdg-open", str(OUT_DIR)])
            return f"Opened: {OUT_DIR}"
        except Exception as e:                              # noqa: BLE001
            return f"Could not open it; go there manually: {OUT_DIR}\n({e})"

    return on_generate, on_release, on_probe, on_open_dir


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
def build_ui(engine):
    import gradio as gr

    with gr.Blocks(title="Aquarius Terimage") as demo:
        # ⚠️ 这个隐藏 Textbox 必须创建在 Blocks 上下文「之内」。
        #    在 Blocks 之外创建的组件不会进入布局组件表（config["components"]），
        #    但事件仍然引用它的 component id ⇒ 前端找不到该组件，**一次点击就会让
        #    整片输出变成错误态**（2026-10-04 实测：点击语言切换后所有组件报错，
        #    `toggle_state` 的 inputs/outputs 都含一个布局里不存在的 id）。
        #    另外：用「隐藏 Textbox」而不是 gr.State 当语言载体，是因为 Gradio 不会把
        #    「含 gr.State 输入」的事件导出成 API 端点 ⇒ ① gradio_client 无法调用、
        #    无法自动化测试；② 出问题时也没有可复现的调用路径。隐藏 Textbox 没有这个限制。
        lang_state = gr.Textbox(value="zh", visible=False, label="lang")
        on_generate, on_release, on_probe, on_open_dir = make_handlers(engine)

        with gr.Row():
            with gr.Column(scale=6):
                header = gr.Markdown(T["zh"]["header"])
            with gr.Column(scale=1, min_width=150):
                btn_lang = gr.Button(T["zh"]["btn_lang"], variant="secondary")

        with gr.Row():
            with gr.Column(scale=4):
                prompt = gr.Textbox(label=T["zh"]["prompt"], value="sunset over the ocean",
                                    lines=5, max_lines=12, placeholder=EXAMPLES)
                steps = gr.Slider(4, 50, value=20, step=1, label=T["zh"]["steps"])
                with gr.Accordion("尺寸 Size", open=True) as acc_size:
                    preset = gr.Dropdown(PRESET["zh"], value=_label(512, 512, "zh"),
                                         label=T["zh"]["preset"])
                    with gr.Row():
                        width = gr.Slider(ap.RES_MIN, ap.RES_MAX, value=512,
                                          step=ap.RES_STEP, label=T["zh"]["width"])
                        height = gr.Slider(ap.RES_MIN, ap.RES_MAX, value=512,
                                           step=ap.RES_STEP, label=T["zh"]["height"])
                    size_info = gr.Markdown(size_note(512, 512, "zh"))
                with gr.Row():
                    seed = gr.Number(value=-1, precision=0, label=T["zh"]["seed"],
                                     info=T["zh"]["seed_info"])
                    per_prompt = gr.Slider(1, 4, value=1, step=1,
                                           label=T["zh"]["per_prompt"])
                lowvram = gr.Checkbox(value=False, label=T["zh"]["lowvram"])
                with gr.Row():
                    btn = gr.Button(T["zh"]["generate"], variant="primary", scale=2)
                    btn_rel = gr.Button(T["zh"]["free"], scale=1)

                tips = gr.Markdown(T["zh"]["tips"])

            with gr.Column(scale=3):
                log = gr.Textbox(label=T["zh"]["log"], lines=12, interactive=False,
                                 max_lines=12, elem_id="aq-log")
                # ⚠️ Gallery 不要设固定 height：设了之后格子高度被压死，图片会被裁掉，
                #    用户得在组件里上下滑动才能看全（2026-10-04 实测：height=420 时
                #    512/640 的上下边都被切，height="auto" 同样被裁）。留 None 让格子按
                #    图片比例自适应 ⇒ 整图一眼可见。
                # 单列：一张图一行，占满结果栏宽度 ⇒ 整图最大且完整可见
                # （columns=2 时每图只有约半栏宽，看质量偏小）
                gallery = gr.Gallery(label=T["zh"]["results"], columns=1,
                                     object_fit="contain", elem_id="aq-gallery")
                outdir = gr.Textbox(label=T["zh"]["outdir"], lines=2,
                                    interactive=False)
                with gr.Accordion("模型信息 Model info", open=False) as acc_info:
                    info = gr.Textbox(label="", lines=8, interactive=False)
                    with gr.Row():
                        btn_probe = gr.Button(T["zh"]["inspect"])
                        btn_open = gr.Button(T["zh"]["open_dir"])
                    probe_msg = gr.Textbox(label="", lines=1, interactive=False)

        # ---------------- language toggle ----------------
        btn_lang.click(
            toggle_state,
            inputs=[lang_state, preset, width, height],
            # ⚠️ 不要把 gr.Accordion 放进 outputs：布局组件在 Gradio 6 里
            #   不接受 `gr.update(label=...)`，点击时会抛错（2026-10-04 实际踩到）。
            outputs=[lang_state, header, btn_lang, prompt, steps, preset,
                     width, height, size_info, seed, per_prompt, lowvram, btn,
                     btn_rel, tips, log, gallery, outdir, btn_probe, btn_open],
        )

        gen_inputs = [prompt, steps, width, height, seed, per_prompt, lowvram, lang_state]
        btn.click(on_generate, inputs=gen_inputs, outputs=[log, gallery, outdir])
        prompt.submit(on_generate, inputs=gen_inputs,
                      outputs=[log, gallery, outdir])
        btn_rel.click(on_release, inputs=[lang_state], outputs=[log, gallery, outdir])
        btn_probe.click(on_probe, outputs=[info])
        btn_open.click(on_open_dir, outputs=[probe_msg])

        # Size wiring: preset -> sliders; slider drag -> refresh the note.
        # NOTE: the note handlers must take the language state as an input, or
        # dragging a slider after switching to English would flip the note back
        # to Chinese.
        preset.change(_on_preset, inputs=[preset, width, height, lang_state],
                      outputs=[width, height])
        width.change(_resize_note, inputs=[width, height, lang_state],
                     outputs=[size_info])
        height.change(_resize_note, inputs=[width, height, lang_state],
                      outputs=[size_info])

        demo.load(on_probe, outputs=[info])

    return demo


# --------------------------------------------------------------------------
# Self-check: runs without gradio, generates one small image on CPU
# --------------------------------------------------------------------------
def self_test(engine, probe_only=False):
    print("=" * 68)
    print("Aquarius Terimage engine self-check")
    print("=" * 68)
    t0 = time.time()
    step, ema = engine.peek_ckpt()
    print(f"[peek] step={step} ema={ema}  (mmap read, took {time.time() - t0:.2f}s)")
    print(f"[env ] {engine.device_info()}")
    print(f"[ckpt] {engine.ckpt}")
    if probe_only:
        return 0

    # Deliberately non-square (192 high x 256 wide) so the "any aspect ratio"
    # path is covered too.
    res = engine.generate(["blue sky over mountains"], steps=2, h=192, w=256,
                          seed=3407, per_prompt=1, low_vram=True,
                          on_status=lambda m: print("  ", m))
    print(f"[out ] {len(res)} image(s) -> {res[0][2] if res else '(none)'}")
    print(f"[vram] peak allocated {torch_peak():.2f} GB")
    print(f"[done] UNet load count = {engine.loads} (should be 1 per image in low-VRAM mode)")
    return 0


def torch_peak():
    import torch
    if not ap._cuda_ok():
        return 0.0
    return torch.cuda.max_memory_allocated() / 2 ** 30


# --------------------------------------------------------------------------
def main():
    ap_arg = argparse.ArgumentParser(description="Aquarius Terimage (Gradio UI)")
    ap_arg.add_argument("--host", default="127.0.0.1", help="bind address")
    ap_arg.add_argument("--port", type=int, default=7860)
    ap_arg.add_argument("--share", action="store_true",
                        help="create a temporary public link (needs gradio.app reachable)")
    ap_arg.add_argument("--no-browser", action="store_true", help="do not auto-open a browser")
    ap_arg.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap_arg.add_argument("--lowvram", action="store_true",
                        help="start directly in low-VRAM mode (peak <3 GB, slow)")
    ap_arg.add_argument("--cvram", action="store_true",
                        help="[alias] same as --lowvram")
    ap_arg.add_argument("--te-mode", choices=["resident", "cache"], default=None,
                        help="text encoder policy: resident=keep in VRAM (default), cache=keep weights in RAM")
    ap_arg.add_argument("--ckpt", default=None, help="model file path")
    ap_arg.add_argument("--self-test", action="store_true",
                        help="no UI; generate one small image on CPU to verify the whole path")
    ap_arg.add_argument("--probe-only", action="store_true",
                        help="with --self-test: read metadata only")
    args = ap_arg.parse_args()

    if args.self_test:
        eng = ap.Engine(device=args.device, cache=False, ckpt=args.ckpt,
                        te_mode=args.te_mode)
        return self_test(eng, probe_only=args.probe_only)

    eng = ap.Engine(device=args.device, cache=not args.lowvram, ckpt=args.ckpt,
                    te_mode=args.te_mode)
    print(f"[env ] {eng.device_info()}")
    print(f"[ckpt] {eng.ckpt}")

    demo = build_ui(eng)
    try:
        demo = demo.queue()
    except Exception:                                       # noqa: BLE001
        pass
    demo.launch(server_name=args.host, server_port=args.port,
                share=args.share, inbrowser=not args.no_browser,
                allowed_paths=[str(OUT_DIR)], head=AUTOSCROLL_JS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
