#!/usr/bin/env python3
"""smbench data: SmolVLA expert SOFTMAX inputs (kernel_vla.cpp softmax_phase) for cyclotron.

Three cases.  SmolVLA, as gen_vla.py emits them (2 batches of 192 rows = 3 heads x 64 action rows, M = 64, TM = 64, TK = 256):
  self   SK = 512, mask table stab (prefix keys 0 .. 204, suffix keys 256 .. 256 + min(t, 49)), b2 = 256
  cross  SK = 256, mask table ctab (prefix keys 0 .. 204), no second range
  dec    kernel.cpp decode (gen_decode.py, TinyLlama shapes): SK 1024, 16 batches of TM 16 rows (8 valid: Gq = 8, T = 1),
         query position 1000 (keys 1001 .. 1023 masked)
S: bf16 scores ~ N(0, 16) (logits S / 8 ~ N(0, 2)).  region1 holds the S arrays and the tables; sm_data.h the offsets.
"""
import os
import numpy as np

B, R, M, TM, TK = 2, 192, 64, 64, 256
LV, NA = 205, 50
REG1 = 0x20000000
NOROW = 0xFFFFFFFF
rng = np.random.default_rng(1)


def bf16_bits(x):
    u = np.asarray(x, np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def tab(lims):
    return np.array(lims, np.uint32).reshape(-1)


blob, off = bytearray(), {}


def put(name, arr):
    while len(blob) % 256:
        blob.append(0)
    off[name] = REG1 + len(blob)
    blob.extend(np.ascontiguousarray(arr).tobytes())


for name, SK in (("s_self", 512), ("s_cross", 256)):
    put(name, bf16_bits(rng.normal(0, 4, (B, R, SK)).astype(np.float32)))
put("s_dec", bf16_bits(rng.normal(0, 4, (16, 16, 1024)).astype(np.float32)))
put("stab", tab([[LV - 1, 256 + min(t, NA - 1)] for t in range(M)]))
put("ctab", tab([[LV - 1, 0] for t in range(M)]))

os.makedirs("data", exist_ok=True)
open("data/sm.region1.bin", "wb").write(blob)
with open("sm_data.h", "w") as f:
    f.write("#pragma once\n")
    for k, v in off.items():
        f.write(f"#define SM_{k.upper()} 0x{v:08x}u\n")
    f.write(f"#define SM_B {B}u\n#define SM_R {R}u\n#define SM_M {M}u\n#define SM_TM {TM}u\n#define SM_TK {TK}u\n")
    f.write(f"#define SM_NOROW 0x{NOROW:08x}u\n")
print(f"region1 {len(blob)} bytes")
