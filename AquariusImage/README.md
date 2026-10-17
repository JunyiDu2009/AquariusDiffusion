# AquariusImage

Image-diffusion sub-series of **AquariusDiffusion** — text-to-image diffusion
models trained natively at low bit-width.

| Model | Quantizer | Status |
|---|---|---|
| [AquariusTerimageExp](AquariusTerimageExp/) | ternary `{−s, 0, +s}` (g128 mean-scale + STE) | **released** — step 260,000; the family's first experiment |

From here on this branch trains **binary** models (`{−s, +s}`). They are named by
domain, not by bit-width, so there is no separate "Binimage" line: the planned binary
counterpart of the first experiment was never trained (the compute budget ran out,
report §7.1), and every model after it is simply named for what it is.

Licence: Apache License 2.0 for code, docs and the weights (see the leaf
repository's `LICENSE` / `NOTICE`).
