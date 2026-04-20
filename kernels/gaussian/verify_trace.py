#!/usr/bin/env python3

import argparse
import re
import sqlite3
import struct
import subprocess
import sys
from pathlib import Path


ARRAY_RE = re.compile(
    r"(?:__global|const)\s+uint32_t\s+([A-Za-z0-9_]+)\[\]\s*=\s*\{([^}]*)\};",
    re.S,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify a gaussian phase kernel from a sqlite trace."
    )
    parser.add_argument("trace_db", type=Path, help="Path to sqlite trace database")
    parser.add_argument("elf", type=Path, help="Path to phase kernel ELF")
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to generated data file (defaults to <stem>_data)",
    )
    parser.add_argument(
        "--expected",
        type=Path,
        default=None,
        help="Path to generated expected file (defaults to <stem>_expected)",
    )
    return parser.parse_args()


def parse_u32_arrays(path: Path) -> dict[str, list[int]]:
    text = path.read_text()
    arrays: dict[str, list[int]] = {}
    for name, body in ARRAY_RE.findall(text):
        values = []
        for token in body.replace("\n", " ").split(","):
            token = token.strip()
            if token:
                values.append(int(token, 0))
        arrays[name] = values
    return arrays


def resolve_symbol_address(elf: Path, symbol: str) -> int:
    result = subprocess.run(
        ["llvm-nm", "-n", str(elf)],
        check=True,
        capture_output=True,
        text=True,
    )
    pattern = re.compile(rf"^([0-9a-fA-F]+)\s+\S\s+{re.escape(symbol)}$")
    for line in result.stdout.splitlines():
        match = pattern.match(line.strip())
        if match:
            return int(match.group(1), 16)
    raise ValueError(f"symbol not found in ELF: {symbol}")


def reconstruct_actual_words(
    trace_db: Path,
    start: int,
    initial_words: list[int],
) -> list[int]:
    image = bytearray().join(struct.pack("<I", word) for word in initial_words)
    end = start + len(image)

    conn = sqlite3.connect(str(trace_db))
    try:
        rows = conn.execute(
            """
            SELECT id, address, size, data
            FROM dmem
            WHERE store = 1 AND address < ? AND (address + size) > ?
            ORDER BY id
            """,
            (end, start),
        ).fetchall()
    finally:
        conn.close()

    for _, address, size, data in rows:
        address = int(address)
        size = max(1, int(size))
        payload = int(data).to_bytes(size, byteorder="little", signed=False)
        lo = max(start, address)
        hi = min(end, address + size)
        if lo >= hi:
            continue
        src_begin = lo - address
        src_end = hi - address
        dst_begin = lo - start
        image[dst_begin:dst_begin + (src_end - src_begin)] = payload[src_begin:src_end]

    return [
        struct.unpack_from("<I", image, 4 * index)[0]
        for index in range(len(initial_words))
    ]


def bits_to_float(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def main() -> int:
    args = parse_args()
    stem = args.elf.name.replace(".radiance.elf", "")
    data_path = args.data or Path(f"{stem}_data")
    expected_path = args.expected or Path(f"{stem}_expected")

    if not args.trace_db.exists():
        raise SystemExit(f"trace db not found: {args.trace_db}")
    if not args.elf.exists():
        raise SystemExit(f"ELF not found: {args.elf}")
    if not data_path.exists():
        raise SystemExit(f"data file not found: {data_path}")
    if not expected_path.exists():
        raise SystemExit(f"expected file not found: {expected_path}")

    data_arrays = parse_u32_arrays(data_path)
    expected_arrays = parse_u32_arrays(expected_path)

    checks = []
    for expected_name, expected_words in expected_arrays.items():
        if not expected_name.startswith("expected_"):
            continue
        symbol = expected_name[len("expected_"):]
        if symbol not in data_arrays:
            raise SystemExit(f"initial data for symbol not found: {symbol}")
        actual_words = reconstruct_actual_words(
            args.trace_db,
            resolve_symbol_address(args.elf, symbol),
            data_arrays[symbol],
        )
        checks.append((symbol, expected_words, actual_words))

    if not checks:
        raise SystemExit("no expected_* arrays found to verify")

    failed = False
    for symbol, expected_words, actual_words in checks:
        mismatches = [
            (index, expected, actual)
            for index, (expected, actual) in enumerate(zip(expected_words, actual_words))
            if expected != actual
        ]
        address = resolve_symbol_address(args.elf, symbol)
        print(f"symbol={symbol} address=0x{address:08x} elements={len(expected_words)}")
        if not mismatches:
            print("PASS")
            continue

        failed = True
        print(f"FAIL mismatches={len(mismatches)}")
        for index, expected, actual in mismatches[:16]:
            print(
                f"[{index}] expected=0x{expected:08x} ({bits_to_float(expected):.8g}) "
                f"actual=0x{actual:08x} ({bits_to_float(actual):.8g})"
            )

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
