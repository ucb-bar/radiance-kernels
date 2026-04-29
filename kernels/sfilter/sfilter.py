#!/usr/bin/env python3

import os
import random
import struct
from pathlib import Path
from string import Template


SIZE = int(os.environ.get("SFILTER_SIZE", "48"))
SEED = 0
MASK = [1.0] * 9


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def bits_to_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def fmt_u32(values: list[int]) -> str:
    return ",".join(f"0x{value:08x}" for value in values) + ","


def fmt_f32(values: list[float]) -> str:
    return ",".join(f"{value.hex()}f" for value in values) + ","


def main() -> None:
    rng = random.Random(SEED)

    src_vals = [f32(rng.uniform(0.0, 100.0)) for _ in range(SIZE * SIZE)]
    dst_init_bits = [0 for _ in range(SIZE * SIZE)]
    mask_vals = [f32(value) for value in MASK]

    expected_vals = [f32(0.0) for _ in range(SIZE * SIZE)]
    for y in range(1, SIZE - 1):
      for x in range(1, SIZE - 1):
        addr = x + y * SIZE
        terms = [
            f32(src_vals[addr - 1 - SIZE] * mask_vals[0]),
            f32(src_vals[addr - SIZE] * mask_vals[1]),
            f32(src_vals[addr + 1 - SIZE] * mask_vals[2]),
            f32(src_vals[addr - 1] * mask_vals[3]),
            f32(src_vals[addr] * mask_vals[4]),
            f32(src_vals[addr + 1] * mask_vals[5]),
            f32(src_vals[addr - 1 + SIZE] * mask_vals[6]),
            f32(src_vals[addr + SIZE] * mask_vals[7]),
            f32(src_vals[addr + 1 + SIZE] * mask_vals[8]),
        ]
        acc = f32(0.0)
        for term in terms:
            acc = f32(acc + term)
        expected_vals[addr] = acc

    src_bits = [f32_bits(value) for value in src_vals]
    expected_bits = [f32_bits(value) for value in expected_vals]
    mask_bits = [f32_bits(value) for value in mask_vals]

    data_template = Template(
        """
__global uint32_t src_raw[] = {
    $src_bits
};

__global uint32_t dst_raw[] = {
    $dst_init_bits
};

const uint32_t ldc = $size;
const uint32_t m0_raw = $m0;
const uint32_t m1_raw = $m1;
const uint32_t m2_raw = $m2;
const uint32_t m3_raw = $m3;
const uint32_t m4_raw = $m4;
const uint32_t m5_raw = $m5;
const uint32_t m6_raw = $m6;
const uint32_t m7_raw = $m7;
const uint32_t m8_raw = $m8;
"""
    )

    expected_template = Template(
        """
size: $size
seed: $seed

const uint32_t expected_sfilter_raw[] = {
    $expected_bits
};

const float expected_sfilter_f32[] = {
    $expected_vals
};
"""
    )

    data_output = data_template.substitute(
        src_bits=fmt_u32(src_bits),
        dst_init_bits=fmt_u32(dst_init_bits),
        size=SIZE,
        m0=f"0x{mask_bits[0]:08x}",
        m1=f"0x{mask_bits[1]:08x}",
        m2=f"0x{mask_bits[2]:08x}",
        m3=f"0x{mask_bits[3]:08x}",
        m4=f"0x{mask_bits[4]:08x}",
        m5=f"0x{mask_bits[5]:08x}",
        m6=f"0x{mask_bits[6]:08x}",
        m7=f"0x{mask_bits[7]:08x}",
        m8=f"0x{mask_bits[8]:08x}",
    ).lstrip()

    expected_output = expected_template.substitute(
        expected_bits=fmt_u32(expected_bits),
        expected_vals=fmt_f32([bits_to_f32(bits) for bits in expected_bits]),
        size=SIZE,
        seed=SEED,
    ).lstrip()

    Path("data").write_text(data_output)
    Path("expected").write_text(expected_output)


if __name__ == "__main__":
    main()
