#!/usr/bin/env python3

import ctypes
import random
import struct
from pathlib import Path
from string import Template


NUM_RECORDS = 256
SEED = 0

LIBM = ctypes.CDLL("libm.so.6")
LIBM.fmaf.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]
LIBM.fmaf.restype = ctypes.c_float
LIBM.sqrtf.argtypes = [ctypes.c_float]
LIBM.sqrtf.restype = ctypes.c_float


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def bits_to_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def fmaf32(a: float, b: float, c: float) -> float:
    return float(LIBM.fmaf(ctypes.c_float(a), ctypes.c_float(b), ctypes.c_float(c)))


def sqrtf32(value: float) -> float:
    return float(LIBM.sqrtf(ctypes.c_float(value)))


def fmt_u32(values: list[int]) -> str:
    return ",".join(f"0x{value:08x}" for value in values) + ","


def fmt_f32(values: list[float]) -> str:
    return ",".join(f"{value.hex()}f" for value in values) + ","


def fmt_locations(locations: list[tuple[float, float]]) -> str:
    return ",\n    ".join(
        f"{{{lat.hex()}f,{lng.hex()}f}}"
        for lat, lng in locations
    ) + ","


def main() -> None:
    rng = random.Random(SEED)

    query_lat = f32(rng.uniform(0.0, 100.0))
    query_lng = f32(rng.uniform(0.0, 100.0))
    locations = [
        (f32(rng.uniform(0.0, 100.0)), f32(rng.uniform(0.0, 100.0)))
        for _ in range(NUM_RECORDS)
    ]

    expected_vals = []
    for lat, lng in locations:
        dx = f32(query_lat - lat)
        dy = f32(query_lng - lng)
        dy_sq = f32(dy * dy)
        dist_sq = fmaf32(dx, dx, dy_sq)
        expected_vals.append(sqrtf32(dist_sq))

    expected_bits = [f32_bits(value) for value in expected_vals]
    dst_init_bits = [0 for _ in range(NUM_RECORDS)]

    data_template = Template(
        """
__global LatLong locations_raw[] = {
    $locations
};

__global uint32_t distances_raw[] = {
    $dst_init_bits
};

const uint32_t num_records = $num_records;
const uint32_t query_lat_raw = $query_lat_raw;
const uint32_t query_lng_raw = $query_lng_raw;
"""
    )

    expected_template = Template(
        """
num_records: $num_records
seed: $seed

const float expected_nearn_f32[] = {
    $expected_vals
};

const uint32_t expected_nearn_raw[] = {
    $expected_bits
};
"""
    )

    data_output = data_template.substitute(
        locations=fmt_locations(locations),
        dst_init_bits=fmt_u32(dst_init_bits),
        num_records=NUM_RECORDS,
        query_lat_raw=f"0x{f32_bits(query_lat):08x}",
        query_lng_raw=f"0x{f32_bits(query_lng):08x}",
    ).lstrip()

    expected_output = expected_template.substitute(
        expected_vals=fmt_f32([bits_to_f32(bits) for bits in expected_bits]),
        expected_bits=fmt_u32(expected_bits),
        num_records=NUM_RECORDS,
        seed=SEED,
    ).lstrip()

    Path("data").write_text(data_output)
    Path("expected").write_text(expected_output)


if __name__ == "__main__":
    main()
