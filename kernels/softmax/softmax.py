import numpy as np
from string import Template

ROWS = 4
COLS = 256

x = np.random.randn(ROWS, COLS).astype(np.float32)

template = Template(
"""
__global float x[] = {
    $x_literals
};

const uint32_t rows = $rows;
const uint32_t cols = $cols;
"""
)

x_literals = ""
for i in range(ROWS):
    for j in range(COLS):
        x_literals += f"{x[i, j]}f,"

with open("data", "w") as f:
    f.write(
        template.substitute(
            x_literals=x_literals,
            rows=ROWS,
            cols=COLS,
        )
    )
