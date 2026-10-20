# AquariusCodaExp · 0.08B 一步去噪器

> 像素空间单步去噪：输入一张"带噪但轮廓大致清晰"的图，**单次前向**输出干净原图。无 VAE、无扩散迭代、无采样链——就是一步回归。

**48.1M 参数**（~0.08B）UNet · RGB 3 通道 · RTX 5060 8GB 可训 · 全流程含 NVFP4 量化与 Gradio Demo

## 这是什么

| | |
|---|---|
| 任务 | 单步去噪：`x_p = √(1−p)·x₀ + √p·ε` → 模型一次前向还原 `x₀` |
| 噪声范围 | p = 噪声方差占比 ∈ **1% ~ 50%**（主任务 1–30%，30–50% 极限档） |
| 架构 | UNet，GroupNorm + SiLU，中段多头自注意力，梯度检查点 |
| 损失 | MSE + LPIPS(VGG) 联合 |
| 数据 | COCO 2017（115K 训练 / 5K 验证） |

与扩散模型相比，它没有采样链——推理就是一次 `model(x_p, p)`，毫秒级。

## 训练配方亮点

- **多分辨率 × 多宽高比桶**：11 个桶（256² ~ 640×384，含 16:9/9:16），每步按权重抽桶，batch 按像素预算自动缩放——推理域对齐任意 /16 尺寸输入。
- **四阶段噪声课程**：1–10% → +20% → +30% → 1–50%，逐阶解锁。
- **平台期早停**：三档 LPIPS 均值连续 1 万步改善 ≤ 0.002 → 自动存档退出。实测 **76,000 步触发**。
- **NVFP4 量化**：E2M1 + per-16 E4M3 块 scale + FP32 per-tensor scale，手写编解码（`code/nvfp4.py`），训练中每 1000 步自动导出。注意：本实验为 **FP16 训练 + 训练后 NVFP4 导出**（非原生低比特路线）。

## 结果（val2017 5K，NVFP4 反量化，step 76000）

| 噪声档 | PSNR (dB) | LPIPS |
|---|---|---|
| p=1%  | 33.50 | 0.091 |
| p=10% | 28.54 | 0.191 |
| p=20% | 26.86 | 0.236 |
| p=30% | 25.78 | 0.270 |
| p=40% | 24.90 | 0.297 |
| p=50% | 24.07 | 0.326 |

量化代价：48M 参数 96MB(FP16) → **27.2MB(NVFP4)**，3.5× 压缩。完整训练曲线见 [`results/`](results/)。

## 快速开始

```bash
# 环境：Python 3.13+，CUDA 版 PyTorch，lpips（8GB 级 GPU 即可）
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install lpips

# 数据：COCO 2017（训练图 / 验证图，路径见下）

cd code
python train.py --data D:/COCO2017/train2017 --eval-data ../data/val2017 \
  --out runs/m2_formal --ckpt-dir ckpts/m2_formal \
  --steps 250000 --batch 6 --res 256 --save-every 1000 --eval-every 500 \
  --workers 8 --early-stop --multi-res --mem-frac 0.65

python demo.py          # Gradio 推理界面 http://127.0.0.1:8792
python dashboard.py --port 8791 --run m2_formal   # 训练监控面板
```

关键参数：`--multi-res` 多桶训练 · `--early-stop` 门控早停 · `--mem-frac` torch 显存硬帽 · `--benchmark` cudnn autotune（本机实测病态慢，默认关闭）

## Demo 特性

- 选图 → 噪声滑块实时预览（与训练口径一致）→ 一键去噪 → 图上分割线拖动对比
- NVFP4 vs FP16 权重现场对照；任意尺寸输入（超 640 等比缩放，reflect pad /16）
- 显存不足自动降档重试；底部日志行记录精确耗时

## 8GB 卡注意事项（Windows WDDM）

显存接近满时页面被挤到共享内存：**GPU 100% 利用率但功耗 <50W** 是特征，速度掉 5–10 倍。治理组合：`--mem-frac` 硬帽 + `max_split_size_mb:256,garbage_collection_threshold:0.9` + 周期性 `empty_cache()`。硬帽必须高于**训练真实峰值**而非预热峰值。

## 目录

```
AquariusCodaExp/
├── README.md / README_zh.md   本页
├── code/                      train / model / nvfp4 / demo / dashboard / test_pipeline
├── docs/                      训练计划书 v1.1 / v1.2（权威文档 v1.2）
└── results/                   训练曲线（评估记录 + 全量压缩日志）
```

## 技术报告

- 中文版（CVPR 双栏格式）：[docs/report/AquariusCodaExp-技术报告-中文版.pdf](docs/report/AquariusCodaExp-技术报告-中文版.pdf)
- English: [docs/report/AquariusCodaExp-Technical-Report-EN.pdf](docs/report/AquariusCodaExp-Technical-Report-EN.pdf)

## 已知问题与路线

- [ ] LPIPS 验收线校准（设计线与实测差距见上表）
- [ ] M3：FP8 QAT（torchao）
- [x] 权重发布：[Release v0.1-m2](https://github.com/JunyiDu2009/AquariusDiffusion/releases/tag/v0.1-m2)
