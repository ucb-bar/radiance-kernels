#!/usr/bin/env python3

import argparse
import os
import sys

import numpy as np

N = 128
D = 192
NUMEL = N * D


def load_tensor(path: str) -> tuple[np.ndarray, np.dtype]:
    size = os.path.getsize(path)
    if size == NUMEL * np.dtype(np.float32).itemsize:
        dtype = np.float32
    elif size == NUMEL * np.dtype(np.float64).itemsize:
        dtype = np.float64
    else:
        raise ValueError(
            f"{path}: unexpected size {size} bytes "
            f"(expected {NUMEL * 4} for float32 or {NUMEL * 8} for float64)."
        )

    arr = np.fromfile(path, dtype=dtype)
    return arr.reshape(N, D), np.dtype(dtype)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check RMSNorm output .bin against expected.bin using np.allclose()."
    )
    parser.add_argument("actual_bin", help="Path to output .bin to validate.")
    parser.add_argument(
        "--expected-bin",
        default="expected.bin",
        help="Path to expected .bin (default: expected.bin).",
    )
    parser.add_argument("--rtol", type=float, default=1e-5, help="Relative tolerance.")
    parser.add_argument("--atol", type=float, default=1e-8, help="Absolute tolerance.")
    args = parser.parse_args()

    actual, actual_dtype = load_tensor(args.actual_bin)
    expected, expected_dtype = load_tensor(args.expected_bin)

    is_close = np.isclose(actual, expected, rtol=args.rtol, atol=args.atol)
    num_different = int(is_close.size - np.count_nonzero(is_close))
    ok = num_different == 0
    if ok:
        print(
            "PASS",
            f"(actual={actual_dtype}, expected={expected_dtype}, rtol={args.rtol}, atol={args.atol})",
        )
        print(f"different_elements={num_different}/{is_close.size}")
        return 0

    abs_diff = np.abs(actual - expected)
    rel_denom = np.maximum(np.abs(expected), 1e-30)
    rel_diff = abs_diff / rel_denom
    max_abs_idx = np.unravel_index(int(np.argmax(abs_diff)), abs_diff.shape)
    max_rel_idx = np.unravel_index(int(np.argmax(rel_diff)), rel_diff.shape)

    print(
        "FAIL",
        f"(actual={actual_dtype}, expected={expected_dtype}, rtol={args.rtol}, atol={args.atol})",
    )
    print(f"different_elements={num_different}/{is_close.size}")
    print(
        f"max_abs_diff={abs_diff[max_abs_idx]:.6e} at index={max_abs_idx}, "
        f"actual={actual[max_abs_idx]:.6e}, expected={expected[max_abs_idx]:.6e}"
    )
    print(
        f"max_rel_diff={rel_diff[max_rel_idx]:.6e} at index={max_rel_idx}, "
        f"actual={actual[max_rel_idx]:.6e}, expected={expected[max_rel_idx]:.6e}"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
