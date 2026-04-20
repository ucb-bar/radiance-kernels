#!/usr/bin/env python3

import random
import struct
from pathlib import Path
from string import Template


N = 64
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

    a_vals = [f32(rng.uniform(-4.0, 4.0)) for _ in range(N)]
    b_vals = [f32(rng.uniform(-4.0, 4.0)) for _ in range(N)]
    c_init_bits = [0 for _ in range(N)]
    expected_vals = [f32(a + b) for a, b in zip(a_vals, b_vals)]

    a_bits = [f32_bits(value) for value in a_vals]
    b_bits = [f32_bits(value) for value in b_vals]
    expected_bits = [f32_bits(value) for value in expected_vals]

    data_template = Template(
        """
__global uint32_t A_raw[] = {
    $a_bits
};

__global uint32_t B_raw[] = {
    $b_bits
};

__global uint32_t C_raw[] = {
    $c_init_bits
};

const uint32_t n = $n;
"""
    )

    expected_template = Template(
        """
n: $n
seed: $seed

const float A_f32[] = {
    $a_vals
};

const float B_f32[] = {
    $b_vals
};

const uint32_t expected_vecadd_raw[] = {
    $expected_bits
};

const float expected_vecadd_f32[] = {
    $expected_vals
};
"""
    )

    data_output = data_template.substitute(
        a_bits=fmt_u32(a_bits),
        b_bits=fmt_u32(b_bits),
        c_init_bits=fmt_u32(c_init_bits),
        n=N,
    ).lstrip()

    expected_output = expected_template.substitute(
        a_vals=fmt_f32([bits_to_f32(bits) for bits in a_bits]),
        b_vals=fmt_f32([bits_to_f32(bits) for bits in b_bits]),
        expected_bits=fmt_u32(expected_bits),
        expected_vals=fmt_f32([bits_to_f32(bits) for bits in expected_bits]),
        n=N,
        seed=SEED,
    ).lstrip()

    Path("data").write_text(data_output)
    Path("expected").write_text(expected_output)


if __name__ == "__main__":
    main()
