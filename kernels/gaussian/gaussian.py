#!/usr/bin/env python3

import ctypes
import random
import struct
from pathlib import Path


SIZE = 5
SEED = 0


LIBM = ctypes.CDLL("libm.so.6")
LIBM.fmaf.restype = ctypes.c_float
LIBM.fmaf.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float)


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def add_f32(a: float, b: float) -> float:
    return f32(a + b)


def sub_f32(a: float, b: float) -> float:
    return f32(a - b)


def mul_f32(a: float, b: float) -> float:
    return f32(a * b)


def div_f32(a: float, b: float) -> float:
    return f32(a / b)


def fma_f32(a: float, b: float, c: float) -> float:
    return float(LIBM.fmaf(ctypes.c_float(a), ctypes.c_float(b), ctypes.c_float(c)))


def copy_matrix(matrix: list[list[float]]) -> list[list[float]]:
    return [row[:] for row in matrix]


def flatten_matrix_bits(matrix: list[list[float]]) -> list[int]:
    return [f32_bits(value) for row in matrix for value in row]


def flatten_vector_bits(vector: list[float]) -> list[int]:
    return [f32_bits(value) for value in vector]


def fmt_u32(values: list[int]) -> str:
    return ",".join(f"0x{value:08x}" for value in values) + ","


def emit_data(path: Path, a: list[list[float]], b: list[float], m: list[list[float]], t: int) -> None:
    text = f"""__global uint32_t a_raw[] = {{
    {fmt_u32(flatten_matrix_bits(a))}
}};

__global uint32_t b_raw[] = {{
    {fmt_u32(flatten_vector_bits(b))}
}};

__global uint32_t m_raw[] = {{
    {fmt_u32(flatten_matrix_bits(m))}
}};

const uint32_t size_val = {SIZE};
const uint32_t t_val = {t};
"""
    path.write_text(text)


def emit_expected_fan1(path: Path, m: list[list[float]], t: int) -> None:
    text = f"""phase: fan1
t: {t}

const uint32_t expected_m_raw[] = {{
    {fmt_u32(flatten_matrix_bits(m))}
}};
"""
    path.write_text(text)


def emit_expected_fan2(path: Path, a: list[list[float]], b: list[float], t: int) -> None:
    text = f"""phase: fan2
t: {t}

const uint32_t expected_a_raw[] = {{
    {fmt_u32(flatten_matrix_bits(a))}
}};

const uint32_t expected_b_raw[] = {{
    {fmt_u32(flatten_vector_bits(b))}
}};
"""
    path.write_text(text)


def emit_wrapper(path: Path, include_name: str, phase_macro: str) -> None:
    path.write_text(
        f'#define GAUSSIAN_DATA_HEADER "{include_name}"\n'
        f"#define {phase_macro}\n"
        f'#include "kernel_impl.hpp"\n'
    )


def fan1_step(a: list[list[float]], m: list[list[float]], t: int) -> None:
    pivot = a[t][t]
    for row in range(t + 1, SIZE):
        m[row][t] = div_f32(a[row][t], pivot)


def fan2_step(a: list[list[float]], b: list[float], m: list[list[float]], t: int) -> None:
    b_t = b[t]
    for row in range(t + 1, SIZE):
        factor = m[row][t]
        for col in range(t, SIZE):
            a[row][col] = fma_f32(-factor, a[t][col], a[row][col])
        b[row] = fma_f32(-factor, b_t, b[row])


def remove_old_generated() -> None:
    patterns = (
        "fan1_t*.cpp",
        "fan2_t*.cpp",
        "fan1_t*_data",
        "fan2_t*_data",
        "fan1_t*_expected",
        "fan2_t*_expected",
    )
    for pattern in patterns:
        for path in Path(".").glob(pattern):
            path.unlink()


def main() -> None:
    remove_old_generated()

    rng = random.Random(SEED)
    a = [
        [f32(rng.uniform(-2.0, 2.0)) for _ in range(SIZE)]
        for _ in range(SIZE)
    ]
    for i in range(SIZE):
        a[i][i] = add_f32(a[i][i], f32(SIZE + 1.0))
    b = [f32(rng.uniform(-2.0, 2.0)) for _ in range(SIZE)]
    m = [[f32(0.0) for _ in range(SIZE)] for _ in range(SIZE)]

    for t in range(SIZE - 1):
        fan1_stem = f"fan1_t{t}"
        emit_data(Path(f"{fan1_stem}_data"), a, b, m, t)
        emit_wrapper(Path(f"{fan1_stem}.cpp"), f"{fan1_stem}_data", "GAUSSIAN_FAN1")

        m_after_fan1 = copy_matrix(m)
        fan1_step(a, m_after_fan1, t)
        emit_expected_fan1(Path(f"{fan1_stem}_expected"), m_after_fan1, t)
        m = m_after_fan1

        fan2_stem = f"fan2_t{t}"
        emit_data(Path(f"{fan2_stem}_data"), a, b, m, t)
        emit_wrapper(Path(f"{fan2_stem}.cpp"), f"{fan2_stem}_data", "GAUSSIAN_FAN2")

        a_after_fan2 = copy_matrix(a)
        b_after_fan2 = b[:]
        fan2_step(a_after_fan2, b_after_fan2, m, t)
        emit_expected_fan2(Path(f"{fan2_stem}_expected"), a_after_fan2, b_after_fan2, t)
        a = a_after_fan2
        b = b_after_fan2


if __name__ == "__main__":
    main()
