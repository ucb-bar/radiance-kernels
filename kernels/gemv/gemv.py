#!/usr/bin/env python3

from pathlib import Path
from string import Template

import torch


ROWS = 2
COLS = 4096
SEED = 0


def fmt_u16(values: torch.Tensor) -> str:
    flat = values.reshape(-1).tolist()
    return ",".join(f"0x{int(v):04x}" for v in flat) + ","


def fmt_bf16(values: torch.Tensor) -> str:
    flat = values.reshape(-1).tolist()
    return ",".join(f"{float(v):.9g}" for v in flat) + ","


def main() -> None:
    torch.manual_seed(SEED)

    a_bf16 = torch.randn((ROWS, COLS), dtype=torch.bfloat16)
    x_bf16 = torch.randn((COLS,), dtype=torch.bfloat16)
    y_init_bits = torch.zeros((ROWS,), dtype=torch.uint16)

    y_bf16 = torch.matmul(a_bf16, x_bf16).to(torch.bfloat16)

    a_bits = a_bf16.view(torch.uint16)
    x_bits = x_bf16.view(torch.uint16)
    y_bits = y_bf16.view(torch.uint16)

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
        a_bf16=fmt_bf16(a_bf16),
        x_bf16=fmt_bf16(x_bf16),
        y_bits=fmt_u16(y_bits),
        y_bf16=fmt_bf16(y_bf16),
        rows=ROWS,
        cols=COLS,
        seed=SEED,
    ).lstrip()

    Path("data").write_text(data_output)
    Path("expected").write_text(expected_output)


if __name__ == "__main__":
    main()
