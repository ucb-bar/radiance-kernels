#!/usr/bin/env python3

import os
import random
import struct
from pathlib import Path
from string import Template


N = int(os.environ.get("SAXPY_N", "4096"))
SEED = 0


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

    factor = f32(rng.uniform(0.0, 100.0))
    src_vals = [f32(rng.uniform(0.0, 100.0)) for _ in range(N)]
    dst_init_vals = [f32(0.0) for _ in range(N)]
    expected_vals = [f32(dst + src * factor) for src, dst in zip(src_vals, dst_init_vals)]

    factor_bits = f32_bits(factor)
    src_bits = [f32_bits(value) for value in src_vals]
    dst_init_bits = [f32_bits(value) for value in dst_init_vals]
    expected_bits = [f32_bits(value) for value in expected_vals]

    data_template = Template(
        """
__global uint32_t src_raw[] = {
    $src_bits
};

__global uint32_t dst_raw[] = {
    $dst_init_bits
};

const uint32_t factor_raw = $factor_bits;
const uint32_t n = $n;
"""
    )

    expected_template = Template(
        """
n: $n
seed: $seed

const float factor_f32 = $factor;

const float src_f32[] = {
    $src_vals
};

const uint32_t expected_saxpy_raw[] = {
    $expected_bits
};

const float expected_saxpy_f32[] = {
    $expected_vals
};
"""
    )

    data_output = data_template.substitute(
        src_bits=fmt_u32(src_bits),
        dst_init_bits=fmt_u32(dst_init_bits),
        factor_bits=f"0x{factor_bits:08x}",
        n=N,
    ).lstrip()

    expected_output = expected_template.substitute(
        factor=bits_to_f32(factor_bits).hex() + "f",
        src_vals=fmt_f32([bits_to_f32(bits) for bits in src_bits]),
        expected_bits=fmt_u32(expected_bits),
        expected_vals=fmt_f32([bits_to_f32(bits) for bits in expected_bits]),
        n=N,
        seed=SEED,
    ).lstrip()

    Path("data").write_text(data_output)
    Path("expected").write_text(expected_output)


if __name__ == "__main__":
    main()
