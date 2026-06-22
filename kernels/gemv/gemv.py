#!/usr/bin/env python3

from pathlib import Path
import random
import struct
from string import Template


ROWS = 16
COLS = 4096
SEED = 0


def f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def bits_to_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def f32_to_bf16_bits(value: float) -> int:
    bits = f32_bits(value)
    return ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) & 0xffff


def bf16_bits_to_f32(bits: int) -> float:
    return bits_to_f32(bits << 16)


def fmt_u16(values: list[int]) -> str:
    return ",".join(f"0x{value:04x}" for value in values) + ","


def fmt_bf16_bits(values: list[int]) -> str:
    return ",".join(f"{bf16_bits_to_f32(value):.9g}" for value in values) + ","


def main() -> None:
    rng = random.Random(SEED)

    a_bits = [
        f32_to_bf16_bits(rng.gauss(0.0, 1.0))
        for _ in range(ROWS * COLS)
    ]
    x_bits = [
        f32_to_bf16_bits(rng.gauss(0.0, 1.0))
        for _ in range(COLS)
    ]
    y_init_bits = [0 for _ in range(ROWS)]

    y_bits = []
    for row in range(ROWS):
        acc = 0.0
        row_offset = row * COLS
        for col in range(COLS):
            acc += bf16_bits_to_f32(a_bits[row_offset + col]) * bf16_bits_to_f32(x_bits[col])
        y_bits.append(f32_to_bf16_bits(acc))

    data_template = Template(
        """
__global uint16_t A_raw[] = {
    $a_bits
};

__global uint16_t x_raw[] = {
    $x_bits
};

__global uint16_t y_raw[] = {
    $y_init_bits
};

const uint32_t m = $rows;
const uint32_t n = $cols;
"""
    )

    expected_template = Template(
        """
rows: $rows
cols: $cols
seed: $seed

const _Float16 A_bf16[] = {
    $a_bf16
};

const _Float16 x_bf16[] = {
    $x_bf16
};

const uint16_t expected_gemv_raw[] = {
    $y_bits
};

const _Float16 expected_gemv_bf16[] = {
    $y_bf16
};
"""
    )

    data_output = data_template.substitute(
        a_bits=fmt_u16(a_bits),
        x_bits=fmt_u16(x_bits),
        y_init_bits=fmt_u16(y_init_bits),
        rows=ROWS,
        cols=COLS,
    ).lstrip()

    expected_output = expected_template.substitute(
        a_bf16=fmt_bf16_bits(a_bits),
        x_bf16=fmt_bf16_bits(x_bits),
        y_bits=fmt_u16(y_bits),
        y_bf16=fmt_bf16_bits(y_bits),
        rows=ROWS,
        cols=COLS,
        seed=SEED,
    ).lstrip()

    Path("data").write_text(data_output)
    Path("expected").write_text(expected_output)


if __name__ == "__main__":
    main()
