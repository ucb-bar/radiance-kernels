#!/usr/bin/env python3
"""rpbench data: ROPE phase inputs (llama_e2e prefill shapes) for cyclotron.

One region1 blob per case (rp_data_<case>.h gives the offsets), M = 128 tokens at positions P0 = 128 .. 255 of an
SK = 256 cache (the kernel's layouts do not depend on SK beyond the row pitch; 256 keeps the outputs small):
  gemma2     hd 256, 8 / 4 heads              (rope_qk_fast<8> + rope_v)
  tinyllama  hd 64, 32 / 4 heads              (rope_qk_fast<2> + rope_v)
  deepseek   hd 128, 12 / 2 heads, q/k/v bias, key center   (reference: the fused rope_*_unit<true>)
  llama7b    hd 128, 32 / 32 heads            (reference: rope_fa + rope_v_fa)
qkv: bf16 ~ N(0, 2); cos / sin: the real tables (theta 10000); bias bf16 ~ N(0, 0.5); key center fp32 ~ N(0, 0.5).
"""
import os
import numpy as np

CASES = {
    "gemma2": dict(hd=256, Hq=8, Hkv=4, TM=32),
    "tinyllama": dict(hd=64, Hq=32, Hkv=4, TM=64),
    "deepseek": dict(hd=128, Hq=12, Hkv=2, TM=64, bias=True),
    "llama7b": dict(hd=128, Hq=32, Hkv=32, TM=64),
}
M, P0, SK = 128, 128, 256
REG1 = 0x20000000
rng = np.random.default_rng(2)


def bf16_bits(x):
    u = np.asarray(x, np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


os.makedirs("data", exist_ok=True)
for name, c in CASES.items():
    hd, Hq, Hkv = c["hd"], c["Hq"], c["Hkv"]
    NQKV = (Hq + 2 * Hkv) * hd
    blob, off = bytearray(), {}

    def put(key, arr):
        while len(blob) % 256:
            blob.append(0)
        off[key] = REG1 + len(blob)
        blob.extend(np.ascontiguousarray(arr).tobytes())

    put("qkv", bf16_bits(rng.normal(0, 2, (M, NQKV))))
    inv = 1.0 / 10000.0 ** (np.arange(0, hd, 2, dtype=np.float64) / hd)
    ang = np.outer(np.arange(P0, P0 + M), inv)
    put("cos", np.cos(ang).astype(np.float32))
    put("sin", np.sin(ang).astype(np.float32))
    put("bias", bf16_bits(rng.normal(0, 0.5, NQKV)) if c.get("bias") else np.zeros(64, np.uint8))
    put("kcen", rng.normal(0, 0.5, (Hkv, hd)).astype(np.float32) if c.get("bias") else np.zeros(64, np.uint8))
    open(f"data/rp_{name}.region1.bin", "wb").write(blob)
    with open(f"rp_data_{name}.h", "w") as f:
        f.write("#pragma once\n")
        for k, v in off.items():
            f.write(f"#define RP_{k.upper()} 0x{v:08x}u\n")
        f.write(f"#define RP_HD {hd}u\n#define RP_HQ {Hq}u\n#define RP_HKV {Hkv}u\n#define RP_TM {c['TM']}u\n")
        f.write(f"#define RP_HAS_BIAS {int(bool(c.get('bias')))}\n#define RP_M {M}u\n#define RP_P0 {P0}u\n#define RP_SK {SK}u\n")
    print(f"{name}: region1 {len(blob)} bytes")
