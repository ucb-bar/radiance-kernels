import numpy as np
from string import Template

ROWS = 4
COLS = 256

x = np.random.randn(ROWS, COLS).astype(np.float32)


def f32_to_bf16_u16(v: np.float32) -> int:
    u32 = np.array([v], dtype=np.float32).view(np.uint32)[0]
    lsb = (u32 >> 16) & np.uint32(1)
    rounding_bias = np.uint32(0x7FFF) + lsb
    bf16_u32 = np.uint32((u32 + rounding_bias) & np.uint32(0xFFFF0000))
    return int((bf16_u32 >> np.uint32(16)) & np.uint32(0xFFFF))

def bf16_u16_to_f32(u: int) -> np.float32:
    return np.array([np.uint32(u) << np.uint32(16)], dtype=np.uint32).view(np.float32)[0]

def row_to_bf16_f32(row: np.ndarray) -> np.ndarray:
    out = np.zeros_like(row, dtype=np.float32)
    for i, v in enumerate(row):
        out[i] = bf16_u16_to_f32(f32_to_bf16_u16(np.float32(v)))
    return out

template = Template(
"""
__global uint16_t x_raw[] = {
    $x_bits
};

const uint16_t expected_row_min_raw[] = {
    $row_min_bits
};

const uint16_t expected_row_max_raw[] = {
    $row_max_bits
};

const uint16_t expected_softmax_raw[] = {
    $softmax_bits
};

const uint32_t rows = $rows;
const uint32_t cols = $cols;
"""
)

x_bits = ""
row_min_bits = ""
row_max_bits = ""
softmax_bits = ""

x_bf16 = np.zeros((ROWS, COLS), dtype=np.float32)
for i in range(ROWS):
    x_bf16[i] = row_to_bf16_f32(x[i])

softmax_ref = np.zeros((ROWS, COLS), dtype=np.float32)
for i in range(ROWS):
    row = x_bf16[i]
    row_max = np.max(row)
    exps = np.exp(row - row_max)
    softmax_ref[i] = exps / np.sum(exps)
    row_min_bits += f"0x{f32_to_bf16_u16(np.min(row)):04x},"
    row_max_bits += f"0x{f32_to_bf16_u16(np.max(row)):04x},"

for i in range(ROWS):
    for j in range(COLS):
        x_bits += f"0x{f32_to_bf16_u16(x_bf16[i, j]):04x},"
        softmax_bits += f"0x{f32_to_bf16_u16(softmax_ref[i, j]):04x},"

with open("data", "w") as f:
    f.write(
        template.substitute(
            x_bits=x_bits,
            row_min_bits=row_min_bits,
            row_max_bits=row_max_bits,
            softmax_bits=softmax_bits,
            rows=ROWS,
            cols=COLS,
        )
    )
