#!/usr/bin/env python3

from pathlib import Path
from string import Template

import torch


ROWS = 4
COLS = 1024
SEED = 0


def fmt_u16(values: torch.Tensor) -> str:
    flat = values.reshape(-1).tolist()
    return ",".join(f"0x{int(v):04x}" for v in flat) + ","


def fmt_bf16(values: torch.Tensor) -> str:
    flat = values.reshape(-1).tolist()
    return ",".join(f"{float(v):.9g}" for v in flat) + ","


def main() -> None:
    torch.manual_seed(SEED)

    x_bf16 = torch.randn((ROWS, COLS), dtype=torch.bfloat16)
    softmax_bf16 = torch.softmax(x_bf16, dim=-1)

    row_max_bf16 = torch.max(x_bf16, dim=-1).values

    x_bits = x_bf16.view(torch.uint16)
    row_max_bits = row_max_bf16.view(torch.uint16)
    softmax_bits = softmax_bf16.view(torch.uint16)

    data_template = Template(
        """
__global uint16_t x_raw[] = {
    $x_bits
};

const uint32_t rows = $rows;
const uint32_t cols = $cols;
"""
    )

    expected_template = Template(
        """
rows: $rows
cols: $cols
seed: $seed

const _Float16_t x_bf16[] = {
    $x_bf16
}

const uint16_t expected_row_max_raw[] = {
    $row_max_bits
};

const uint16_t expected_softmax_raw[] = {
    $softmax_bits
};

const _Float16 expected_softmax_bf16[] = {
    $softmax_bf16
};
"""
    )

    data_output = data_template.substitute(
        x_bits=fmt_u16(x_bits),
        rows=ROWS,
        cols=COLS,
    ).lstrip()

    expected_output = expected_template.substitute(
        x_bf16=fmt_bf16(x_bf16),
        row_max_bits=fmt_u16(row_max_bits),
        softmax_bits=fmt_u16(softmax_bits),
        softmax_bf16=fmt_bf16(softmax_bf16),
        rows=ROWS,
        cols=COLS,
        seed=SEED,
    ).lstrip()

    Path("data").write_text(data_output)
    Path("expected").write_text(expected_output)


if __name__ == "__main__":
    main()
