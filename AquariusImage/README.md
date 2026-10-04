# AquariusImage

Image-diffusion sub-series of **AquariusDiffusion** — text-to-image diffusion
models trained natively at low bit-width.

| Model | Quantizer | Status |
|---|---|---|
| [AquariusTerimage](AquariusTerimage/) | ternary `{−s, 0, +s}` (g128 mean-scale + STE) | **trained** — step 260,000, released |
| AquariusBinimage | binary `{−s, +s}` | **not trained** — the compute budget was exhausted (report §7.1) |

`AquariusBinimage` has no trained checkpoint: do not expect weights, samples or
download links for it.

Licence: Apache License 2.0 for code, docs and the weights (see the leaf
repository's `LICENSE` / `NOTICE`).
