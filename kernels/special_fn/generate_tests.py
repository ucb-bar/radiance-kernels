import math

sin_test_vector = [
    0.0,
    math.pi / 6,
    math.pi / 4,
    math.pi / 3,
    math.pi / 2,
    math.pi,
    3 * math.pi / 2,
    2 * math.pi,

    1.0e-6,
    1.0e-10,
    1.0e-20,
    1.0e-38,
    1.4e-45,
    5.9e-39,
    -1.0e-6,
    -1.4e-45,

    100.0,
    1000.0,
    1.0e6,
    1.0e10,
    -1.0e6,

    -0.0
]

with open('sin', 'w') as f:
    for arg in sin_test_vector:
        expected = math.sin(arg)
        f.write(f'{float.hex(arg)},{float.hex(expected)},')

with open('cos', 'w') as f:
    for arg in sin_test_vector:
        expected = math.cos(arg)
        f.write(f'{float.hex(arg)},{float.hex(expected)},')

exp_test_vector = [
    0.0,
    1.0,
    10.0,
    -1.0,
    -10.0
]

with open('exp', 'w') as f:
    for arg in exp_test_vector:
        expected = math.exp(arg)
        f.write(f'{float.hex(arg)},{float.hex(expected)},')