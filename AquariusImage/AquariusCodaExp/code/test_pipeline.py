"""启动前管线自检：模型前向 → NVFP4 往返误差 → 导出/加载/反量化前向。"""
import os
import sys
import tempfile
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import UNetDenoiser
from nvfp4 import export_model_nvfp4, dequantize_model, quantize_weight, dequantize_weight

dev = torch.device("cuda")
print("[1] build model")
m = UNetDenoiser().to(dev)
print(f"    params: {m.num_params()/1e6:.2f}M")

print("[2] forward 256 batch2 / 512 batch1")
x = torch.randn(2, 3, 256, 256, device=dev) * 0.5
p = torch.tensor([0.1, 0.3], device=dev)
t0 = time.time()
with torch.autocast("cuda", dtype=torch.bfloat16):
    y = m(x, p)
torch.cuda.synchronize()
print(f"    out256 {tuple(y.shape)} {y.dtype}, {(time.time()-t0)*1000:.0f}ms")
x5 = torch.randn(1, 3, 512, 512, device=dev) * 0.5
p5 = torch.tensor([0.2], device=dev)
with torch.autocast("cuda", dtype=torch.bfloat16):
    y5 = m(x5, p5)
print(f"    out512 {tuple(y5.shape)}")

print("[3] nvfp4 weight round-trip")
w = (torch.randn(512, 512, device=dev) * 0.05)
q = quantize_weight(w)
wd = dequantize_weight(q)
rel = (wd.float().cpu() - w.detach().cpu()).abs().mean().item() / w.detach().cpu().abs().mean().item()
print(f"    codes {tuple(q['codes'].shape)} scales {q['block_scales'].dtype} rel_err {rel:.4f}")

print("[4] full-model nvfp4 export -> save -> load -> dequant -> forward")
sd = {k: v.detach().cpu() for k, v in m.state_dict().items()}
pkg = export_model_nvfp4(sd)
path = os.path.join(tempfile.gettempdir(), "test_nvfp4.pt")
torch.save(pkg, path)
print(f"    file {os.path.getsize(path)/1e6:.1f}MB, nvfp4 params {pkg['stats']['nvfp4_params']/1e6:.1f}M, fp16 {pkg['stats']['fp16_params']/1e6:.2f}M")
pkg2 = torch.load(path, weights_only=False)
sd2 = dequantize_model(pkg2)
m2 = UNetDenoiser().to(dev)
m2.load_state_dict(sd2)
with torch.autocast("cuda", dtype=torch.bfloat16):
    y2 = m2(x, p)
diff = (y2.float() - y.float()).abs().mean().item()
print(f"    dequant forward mean|diff| {diff:.5f}")
print("ALL OK")
