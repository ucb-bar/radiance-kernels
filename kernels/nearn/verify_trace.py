#!/usr/bin/env python3

import argparse
import re
import sqlite3
import struct
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify nearn output by reconstructing distances_raw from a Cyclotron sqlite trace."
    )
    parser.add_argument("trace_db", type=Path, help="Path to Cyclotron sqlite trace database")
    parser.add_argument("elf", type=Path, help="Path to nearn kernel.radiance.elf")
    parser.add_argument(
        "--expected",
        type=Path,
        default=Path("expected"),
        help="Path to expected data file (default: expected)",
    )
    parser.add_argument(
        "--symbol",
        default="distances_raw",
        help="Output symbol to resolve with llvm-nm (default: distances_raw)",
    )
    return parser.parse_args()


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


def parse_expected_words(expected_path: Path) -> list[int]:
    text = expected_path.read_text()
    match = re.search(
        r"const\s+uint32_t\s+expected_nearn_raw\[\]\s*=\s*\{([^}]*)\};",
        text,
        re.S,
    )
    if match is None:
        raise ValueError("expected_nearn_raw[] not found in expected file")

    values = []
    for token in match.group(1).replace("\n", " ").split(","):
        token = token.strip()
        if token:
            values.append(int(token, 0))
    if not values:
        raise ValueError("expected_nearn_raw[] is empty")
    return values


def clipped_payload(address: int, size: int, data: int, start: int, end: int) -> tuple[int, bytes] | None:
    size = max(1, size)
    payload = int(data).to_bytes(size, byteorder="little", signed=False)
    payload_end = address + size
    clip_lo = max(start, address)
    clip_hi = min(end, payload_end)
    if clip_lo >= clip_hi:
        return None
    begin = clip_lo - address
    stop = clip_hi - address
    return (clip_lo, payload[begin:stop])


def reconstruct_actual_words(trace_db: Path, start: int, num_words: int) -> list[int]:
    end = start + 4 * num_words
    image = bytearray(4 * num_words)

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
        clipped = clipped_payload(int(address), int(size), int(data), start, end)
        if clipped is None:
            continue
        clipped_addr, payload = clipped
        offset = clipped_addr - start
        image[offset:offset + len(payload)] = payload

    return [struct.unpack_from("<I", image, 4 * index)[0] for index in range(num_words)]


def bits_to_float(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def main() -> int:
    args = parse_args()
    if not args.trace_db.exists():
        raise SystemExit(f"trace db not found: {args.trace_db}")
    if not args.elf.exists():
        raise SystemExit(f"ELF not found: {args.elf}")
    if not args.expected.exists():
        raise SystemExit(f"expected file not found: {args.expected}")

    expected_words = parse_expected_words(args.expected)
    output_addr = resolve_symbol_address(args.elf, args.symbol)
    actual_words = reconstruct_actual_words(args.trace_db, output_addr, len(expected_words))

    mismatches = []
    for index, (expected, actual) in enumerate(zip(expected_words, actual_words)):
        if expected != actual:
            mismatches.append((index, expected, actual))

    print(f"symbol={args.symbol} address=0x{output_addr:08x} elements={len(expected_words)}")
    if not mismatches:
        print("PASS")
        return 0

    print(f"FAIL mismatches={len(mismatches)}")
    for index, expected, actual in mismatches[:16]:
        print(
            f"[{index}] expected=0x{expected:08x} ({bits_to_float(expected):.8g}) "
            f"actual=0x{actual:08x} ({bits_to_float(actual):.8g})"
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
