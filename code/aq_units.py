# -*- coding: utf-8 -*-
"""Aquarius 统一体积单位 —— **十进制 MB / GB**，全项目禁用 MiB / GiB。

用户指令（2026-10-03）：
    「以后统一使用 MB 或 GB 做单位，不用 MiB 或 GiB 了」

⚠️ 这不是改标签，是**换算**。同一字节数换个单位，数值会变：

    1 MiB = 1,048,576 B      1 MB = 1,000,000 B      ⇒  MiB × 1.048576     = MB
    1 GiB = 1,073,741,824 B  1 GB = 1,000,000,000 B  ⇒  GiB × 1.073741824  = GB

也就是说，历史记录里凡是写 MiB/GiB 的地方，**把单位名换掉会低估 4.86%（MB）/ 7.37%（GB）**。
正确做法是把数值也重算一遍 —— 本模块就是那唯一的换算入口，
`legacy_mib()` / `legacy_gib()` 专门用来把旧文档里的数字搬过来。

换算速查（本项目实测量，旧口径 → 新口径）
--------------------------------------------------------------
| 项 | 旧（MiB/GiB） | 新（MB/GB） |
|---|---|---|
| 打包交付 旧格式 b2/fp32     | 179.71 MiB | 188.44 MB |
| **打包交付 新格式 b3/fp16** | **145.90 MiB** | **152.99 MB** |
| int4 常驻权重              | 274.0 MiB  | 287.3 MB  |
| 权重稳态（含非量化层）      | 315.0 MiB  | 330.3 MB  |
| bf16 搬卡                  | 1070.0 MiB | 1122.0 MB |
| 转换峰值                    | 1105.0 MiB | 1158.7 MB |
| UNet 前向峰值               | 406.0 MiB  | 425.7 MB  |
| VAE 解码峰值                | 886.0 MiB  | 929.0 MB  |
| 训练检查点                  | 5.2189 GiB | 5.6038 GB |
| 单卡总显存（RTX 5090）      | 31.85 GiB  | 34.19 GB  |
| 训练时显存占用              | 21.2 GB*   | 22.76 GB  |

\\* 旧代码用 `2**30` 除，所以「21.2GB」其实是 21.2 GiB。

用法
----
    from aq_units import human, mb, gb
    human(152_990_556)      # '153.0 MB'
    mb(286_953_472)         # 286.953472
    legacy_mib(145.9)       # 152.99...  ← 把旧文档里的 MiB 数字搬过来
"""
from __future__ import annotations

# --- 十进制常量（SI 前缀）--------------------------------------------------
KB = 10 ** 3
MB = 10 ** 6
GB = 10 ** 9
TB = 10 ** 12

# --- 旧口径换算系数（只在迁移历史数字时用）---------------------------------
MIB_TO_MB = 1.048576
GIB_TO_GB = 1.073741824


def mb(nbytes) -> float:
    """字节 → MB（10^6）。"""
    return float(nbytes) / MB


def gb(nbytes) -> float:
    """字节 → GB（10^9）。"""
    return float(nbytes) / GB


def human(nbytes, digits: int = 1) -> str:
    """字节 → 带单位的字符串，**十进制**：

        999        -> '999 B'
        152990556  -> '153.0 MB'
        5603704966 -> '5.6 GB'
    """
    n = float(nbytes)
    if n < KB:
        return f"{n:.0f} B"
    for unit, div in (("TB", TB), ("GB", GB), ("MB", MB), ("KB", KB)):
        if n >= div:
            return f"{n / div:.{digits}f} {unit}"
    return f"{n:.0f} B"


def human_bytes(nbytes, digits: int = 2) -> str:
    """同 `human`，浏览器/报告里更常用的两位小数。"""
    return human(nbytes, digits)


# --- 迁移助手：把旧文档里的 MiB/GiB 数字搬成 MB/GB -------------------------
def legacy_mib(x: float, digits: int = 2) -> float:
    """旧文档的 MiB 数值 → MB 数值（× 1.048576）。"""
    return round(x * MIB_TO_MB, digits)


def legacy_gib(x: float, digits: int = 2) -> float:
    """旧文档的 GiB 数值 → GB 数值（× 1.073741824）。"""
    return round(x * GIB_TO_GB, digits)


if __name__ == "__main__":                       # 自检：python code/aq_units.py
    assert human(999) == "999 B"
    assert human(152_990_556) == "153.0 MB"
    assert human(5_603_704_966) == "5.6 GB"
    assert abs(mb(286_953_472) - 286.953472) < 1e-6
    assert abs(legacy_mib(145.90) - 153.0) < 0.01
    assert abs(legacy_gib(5.2189) - 5.60) < 0.01
    print("aq_units 自检通过：十进制 MB/GB")
    for b, lab in ((188_437_928, "旧格式 b2/fp32"), (152_990_556, "新格式 b3/fp16"),
                   (286_953_472, "int4 常驻权重"), (5_603_704_966, "训练检查点"),
                   (34_190_917_632, "RTX 5090 总显存")):
        print(f"  {lab:16s} {b:>15,} B = {human(b, 2):>10s}")
