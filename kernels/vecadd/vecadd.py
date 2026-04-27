#!/usr/bin/env python3

import os
import random
import struct
from pathlib import Path


N = int(os.environ.get("VECADD_N", "8192"))
SEED = 0


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def fmt_u32(values: list[int]) -> str:
    return ",".join(f"0x{value:08x}" for value in values) + ","


def main() -> None:
    rng = random.Random(SEED)

    a_vals = [f32(rng.uniform(-4.0, 4.0)) for _ in range(N)]
    b_vals = [f32(rng.uniform(-4.0, 4.0)) for _ in range(N)]
    c_init_bits = [0 for _ in range(N)]
    expected_vals = [f32(a + b) for a, b in zip(a_vals, b_vals)]

    a_bits = [f32_bits(value) for value in a_vals]
    b_bits = [f32_bits(value) for value in b_vals]
    expected_bits = [f32_bits(value) for value in expected_vals]

    data_output = f"""
__global uint32_t A_raw[] = {{
    {fmt_u32(a_bits)}
}};

__global uint32_t B_raw[] = {{
    {fmt_u32(b_bits)}
}};

__global uint32_t C_raw[] = {{
    {fmt_u32(c_init_bits)}
}};

const uint32_t n = {N};
""".lstrip()

    expected_output = f"""
n: {N}
seed: {SEED}

const uint32_t expected_vecadd_raw[] = {{
    {fmt_u32(expected_bits)}
}};
""".lstrip()

    Path("data").write_text(data_output)
    Path("expected").write_text(expected_output)


if __name__ == "__main__":
    main()
