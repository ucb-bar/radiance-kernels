#!/usr/bin/env python3
"""Generate test vectors A, B and expected C = A * B (component-wise bf16)."""

from pathlib import Path
from string import Template

import torch

N = 4096  # bf16 elements per vector  (8 KiB each)
SEED = 42


def fmt_u16(values: torch.Tensor) -> str:
    flat = values.reshape(-1).tolist()
    return ",".join(f"0x{int(v):04x}" for v in flat) + ","


def fmt_bf16(values: torch.Tensor) -> str:
    flat = values.reshape(-1).tolist()
    return ",".join(f"{float(v):.9g}" for v in flat) + ","


def main() -> None:
    torch.manual_seed(SEED)

    a_bf16 = torch.randn(N, dtype=torch.bfloat16)
    b_bf16 = torch.randn(N, dtype=torch.bfloat16)
    c_bf16 = (a_bf16 * b_bf16).to(torch.bfloat16)

    a_bits = a_bf16.view(torch.uint16)
    b_bits = b_bf16.view(torch.uint16)
    c_bits = c_bf16.view(torch.uint16)

    a_template = Template(
        """\
__attribute__((section(".a_data")))
__global uint16_t a_raw[] = {
    $a_bits
};

const uint32_t n_elems = $n;
"""
    )

    b_template = Template(
        """\
__attribute__((section(".b_data")))
__global uint16_t b_raw[] = {
    $b_bits
};
"""
    )

    expected_template = Template(
        """\
n_elems: $n
seed: $seed

const uint16_t expected_c_raw[] = {
    $c_bits
};

const _Float16 expected_c_bf16[] = {
    $c_bf16
};
"""
    )

    Path("a_data").write_text(
        a_template.substitute(a_bits=fmt_u16(a_bits), n=N)
    )
    Path("b_data").write_text(
        b_template.substitute(b_bits=fmt_u16(b_bits))
    )
    Path("expected").write_text(
        expected_template.substitute(
            c_bits=fmt_u16(c_bits),
            c_bf16=fmt_bf16(c_bf16),
            n=N,
            seed=SEED,
        )
    )


if __name__ == "__main__":
    main()
