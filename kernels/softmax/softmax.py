import numpy as np
from string import Template

ROWS = 4
COLS = 256

x = np.random.randn(ROWS, COLS).astype(np.float32)


def f32_to_bf16_hex_literal(v: np.float32) -> str:
    u32 = np.array([v], dtype=np.float32).view(np.uint32)[0]
    lsb = (u32 >> 16) & np.uint32(1)
    rounding_bias = np.uint32(0x7FFF) + lsb
    bf16_u32 = np.uint32((u32 + rounding_bias) & np.uint32(0xFFFF0000))
    bf16_f32 = np.array([bf16_u32], dtype=np.uint32).view(np.float32)[0]
    return f"(_Float16){float(bf16_f32).hex()}"

template = Template(
"""
__global _Float16 x[] = {
    $x_literals
};

const uint32_t rows = $rows;
const uint32_t cols = $cols;
"""
)

x_literals = ""
for i in range(ROWS):
    for j in range(COLS):
        x_literals += f"{f32_to_bf16_hex_literal(x[i, j])},"

with open("data", "w") as f:
    f.write(
        template.substitute(
            x_literals=x_literals,
            rows=ROWS,
            cols=COLS,
        )
    )
