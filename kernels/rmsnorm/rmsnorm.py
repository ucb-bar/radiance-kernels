import numpy as np
from string import Template

np.random.seed(42)

N = 128
D = 192
input_tensor = np.random.randn(N, D).astype(np.float32)
gamma = np.random.randn(D).astype(np.float32)

EPS = np.float32(1.0e-6)
inv_rms = np.float32(1.0) / np.sqrt(EPS + np.sum(input_tensor * input_tensor, axis=1, keepdims=True) / np.float32(D))
expected_tensor = (input_tensor * inv_rms * gamma).astype(np.float32)

template = Template(
"""
__global float data[] = {
    $data_literals
};

__global float gamma[] = {
    $gamma_literals
};
"""
)

data_literals = ""
gamma_literals = ""

for i in range(N):
    for j in range(D):
        data_literals += f"{input_tensor[i, j]}f,"

for i in range(D):
    gamma_literals += f"{gamma[i]}f,"

with open('data', 'w') as f:
    f.write(template.substitute(data_literals=data_literals, gamma_literals=gamma_literals))

with open('expected.bin', 'wb') as f:
    expected_tensor.tofile(f)
