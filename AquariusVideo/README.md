# AquariusVideo

**Planned branch — video diffusion.** / **规划中的分支 —— 视频扩散。**

This branch of [AquariusDiffusion](../..) is reserved for native low-bit (NLT)
**video** diffusion models. Nothing is released yet.

本目录是 [AquariusDiffusion](../..) 的**视频扩散**分支，为原生低比特（NLT）
视频扩散模型预留。**目前没有任何已发布内容。**

## What will live here / 这里将来会放什么

- Binary (`{−s, +s}`) text-to-video / image-to-video models trained natively at low
  bit-width — the family convention from `AquariusImage/AquariusTerimageExp` onward is
  **binary weights, named by domain, not by bit-width**.
- 原生低比特训练的**二值**（`{−s, +s}`）文生视频 / 图生视频模型 —— 自
  `AquariusImage/AquariusTerimageExp` 起，家族约定为**二值权重、按域命名，不按位宽命名**。

## Reference point / 可参考的先行实验

The first experiment of the family (ternary, image domain) is
[`AquariusImage/AquariusTerimageExp`](../AquariusImage/AquariusTerimageExp/) — its
technical report documents the end-to-end recipe (NLT training → bit-packing →
fused-kernel inference) and the honest negative result at this data scale.

家族的第一个实验（三值、图像域）是
[`AquariusImage/AquariusTerimageExp`](../AquariusImage/AquariusTerimageExp/) ——
其技术报告记录了完整链路（NLT 训练 → 位打包 → 融合内核推理）以及当前数据规模下
如实记录的负面结论。

## Licence / 许可

Apache License 2.0 (see the repository root [`LICENSE`](../../LICENSE) /
[`NOTICE`](../../NOTICE)).

Apache License 2.0（见仓库根目录 [`LICENSE`](../../LICENSE) / [`NOTICE`](../../NOTICE)）。
