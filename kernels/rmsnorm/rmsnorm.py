import numpy as np
from string import Template

N = 128
D = 192
input_tensor = np.random.randn(N, D)
gamma = np.random.randn(D)


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
