#!/usr/bin/env python3

from pathlib import Path
from string import Template

import torch


ROWS = 4
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
    b_bf16 = torch.randn((ROWS, COLS), dtype=torch.bfloat16)

    # SwiGLU: C = swish(A) * B = (A * sigmoid(A)) * B
    c_bf16 = (torch.nn.functional.silu(a_bf16.float()) * b_bf16.float()).to(
        torch.bfloat16
    )

    a_bits = a_bf16.view(torch.uint16)
    b_bits = b_bf16.view(torch.uint16)
    # C_raw is output buffer, initialized to zero
    c_init_bits = torch.zeros((ROWS, COLS), dtype=torch.uint16)
    c_bits = c_bf16.view(torch.uint16)

    data_template = Template(
        """
__global uint16_t A_raw[] = {
    $a_bits
};

__global uint16_t B_raw[] = {
    $b_bits
};

__global uint16_t C_raw[] = {
    $c_init_bits
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

const _Float16 B_bf16[] = {
    $b_bf16
};

const uint16_t expected_swiglu_raw[] = {
    $c_bits
};

const _Float16 expected_swiglu_bf16[] = {
    $c_bf16
};
"""
    )

    data_output = data_template.substitute(
        a_bits=fmt_u16(a_bits),
        b_bits=fmt_u16(b_bits),
        c_init_bits=fmt_u16(c_init_bits),
        rows=ROWS,
        cols=COLS,
    ).lstrip()

    expected_output = expected_template.substitute(
        a_bf16=fmt_bf16(a_bf16),
        b_bf16=fmt_bf16(b_bf16),
        c_bits=fmt_u16(c_bits),
        c_bf16=fmt_bf16(c_bf16),
        rows=ROWS,
        cols=COLS,
        seed=SEED,
    ).lstrip()

    Path("data").write_text(data_output)
    Path("expected").write_text(expected_output)


if __name__ == "__main__":
    main()
