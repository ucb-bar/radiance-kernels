#!/usr/bin/env python3

from pathlib import Path
from string import Template

import torch


M = 64
K = 64
N = 64
SEED = 0


def fmt_u16(values: torch.Tensor) -> str:
    flat = values.reshape(-1).tolist()
    return ",".join(f"0x{int(v):04x}" for v in flat) + ","


def fmt_bf16(values: torch.Tensor) -> str:
    flat = values.reshape(-1).tolist()
    return ",".join(f"{float(v):.9g}" for v in flat) + ","


def main() -> None:
    torch.manual_seed(SEED)

    a_bf16 = torch.randn((M, K), dtype=torch.bfloat16)
    b_bf16 = torch.randn((K, N), dtype=torch.bfloat16)
    c_init_bits = torch.zeros((M, N), dtype=torch.uint16)

    c_bf16 = torch.matmul(a_bf16.float(), b_bf16.float()).to(torch.bfloat16)

    a_bits = a_bf16.view(torch.uint16)
    b_bits = b_bf16.view(torch.uint16)
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

const uint32_t M_val = $M;
const uint32_t K_val = $K;
const uint32_t N_val = $N;
"""
    )

    expected_template = Template(
        """
M: $M
K: $K
N: $N
seed: $seed

const _Float16 A_bf16[] = {
    $a_bf16
};

const _Float16 B_bf16[] = {
    $b_bf16
};

const uint16_t expected_C_raw[] = {
    $c_bits
};

const _Float16 expected_C_bf16[] = {
    $c_bf16
};
"""
    )

    data_output = data_template.substitute(
        a_bits=fmt_u16(a_bits),
        b_bits=fmt_u16(b_bits),
        c_init_bits=fmt_u16(c_init_bits),
        M=M,
        K=K,
        N=N,
    ).lstrip()

    expected_output = expected_template.substitute(
        a_bf16=fmt_bf16(a_bf16),
        b_bf16=fmt_bf16(b_bf16),
        c_bits=fmt_u16(c_bits),
        c_bf16=fmt_bf16(c_bf16),
        M=M,
        K=K,
        N=N,
        seed=SEED,
    ).lstrip()

    Path("data").write_text(data_output)
    Path("expected").write_text(expected_output)


if __name__ == "__main__":
    main()
