#!/usr/bin/env python3
"""Rewrite an ELF's GPU-memory load segments into the hashed physical layout of a hashed Radiance
memory system, so that a loader which writes DRAM directly (SimDRAM +loadmem, FireSim LoadMem) puts
every byte where the hashed hardware looks for it.

usage: scramble_gpu_elf.py IN OUT [--base 0x0] [--size 0x80000000] [--unit 32] [--slices 4]

The hash is the one in radiance's AddressHashParams (radiance/memory/AddressHash.scala) and must use
the same parameters as the RTL. Inside [base, base + size) an offset splits into a unit index v and a
byte within the unit; with s = log2(slices), o = v >> s and slice c = (v mod slices) XOR fold(o),
where fold XORs the s-bit groups of o. The physical offset is c * size/slices + o * unit + byte.

Run it on the rv32 kernel ELF before fusing (GPU-local addresses, base 0) or on any ELF whose
GPU-memory addresses lie in [base, base + size). Only the program headers change: each PT_LOAD inside
the range is replaced by the maximal contiguous physical runs of its file bytes, sorted by address.
Section headers and their data are left untouched (they still describe the unscrambled layout).
Bytes past p_filesz (.bss) are not loaded, as the fuse step already ignores them.

A scrambled ELF must not be loaded through TSI: the front-bus hash would apply the hash a second time.
"""

import argparse
import random
import struct
import sys
from dataclasses import dataclass

PT_LOAD = 1


@dataclass(frozen=True)
class AddressHash:
    """Line-granular XOR-fold hash; mirrors AddressHashParams in radiance."""

    base: int
    size: int
    unit: int
    slices: int

    def __post_init__(self) -> None:
        for name in ("size", "unit", "slices"):
            value = getattr(self, name)
            if value <= 0 or value & (value - 1):
                raise ValueError(f"{name} must be a power of two, got {value:#x}")
        if self.base & (self.size - 1):
            raise ValueError(f"base {self.base:#x} must be aligned to size {self.size:#x}")
        if self.slices < 2 or self.unit * self.slices > self.size:
            raise ValueError("need slices >= 2 and at least one unit per slice")

    @property
    def slice_bits(self) -> int:
        return self.slices.bit_length() - 1

    @property
    def unit_bits(self) -> int:
        return self.unit.bit_length() - 1

    @property
    def range_bits(self) -> int:
        return self.size.bit_length() - 1

    def contains(self, addr: int) -> bool:
        return self.base <= addr < self.base + self.size

    def _fold(self, o: int) -> int:
        acc = 0
        while o:
            acc ^= o & (self.slices - 1)
            o >>= self.slice_bits
        return acc

    def forward(self, addr: int) -> int:
        """Client (unscrambled) address to physical address."""
        if not self.contains(addr):
            return addr
        off = addr - self.base
        byte = off & (self.unit - 1)
        v = off >> self.unit_bits
        o = v >> self.slice_bits
        c = (v & (self.slices - 1)) ^ self._fold(o)
        return self.base + (c << (self.range_bits - self.slice_bits)) + (o << self.unit_bits) + byte

    def inverse(self, addr: int) -> int:
        """Physical address back to the client address."""
        if not self.contains(addr):
            return addr
        off = addr - self.base
        byte = off & (self.unit - 1)
        c = off >> (self.range_bits - self.slice_bits)
        o = (off >> self.unit_bits) & ((1 << (self.range_bits - self.unit_bits - self.slice_bits)) - 1)
        v = (o << self.slice_bits) | (c ^ self._fold(o))
        return self.base + (v << self.unit_bits) + byte

    def self_check(self, samples: int = 4096) -> None:
        rng = random.Random(1)
        for _ in range(samples):
            addr = self.base + rng.randrange(self.size)
            phys = self.forward(addr)
            if not self.contains(phys) or self.inverse(phys) != addr:
                raise AssertionError(f"hash is not a bijection at {addr:#x}")


@dataclass
class ElfLayout:
    """Byte layout of the ELF header fields this tool reads and writes."""

    phdr_fmt: str
    phoff_fmt: str
    phoff_at: int
    phentsize_at: int
    phnum_at: int

    @staticmethod
    def for_ident(ident: bytes) -> "ElfLayout":
        if ident[:4] != b"\x7fELF":
            raise ValueError("not an ELF file")
        if ident[5] != 1:
            raise ValueError("only little-endian ELF is supported")
        if ident[4] == 1:  # ELF32: type offset vaddr paddr filesz memsz flags align
            return ElfLayout("<IIIIIIII", "<I", 0x1C, 0x2A, 0x2C)
        if ident[4] == 2:  # ELF64: type flags offset vaddr paddr filesz memsz align
            return ElfLayout("<IIQQQQQQ", "<Q", 0x20, 0x36, 0x38)
        raise ValueError("unknown ELF class")

    @property
    def is64(self) -> bool:
        return self.phdr_fmt.endswith("QQQQQQ")


@dataclass
class Phdr:
    ptype: int
    flags: int
    offset: int
    vaddr: int
    paddr: int
    filesz: int
    memsz: int
    align: int

    @staticmethod
    def unpack(layout: ElfLayout, raw: bytes) -> "Phdr":
        f = struct.unpack(layout.phdr_fmt, raw)
        if layout.is64:
            return Phdr(f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7])
        return Phdr(f[0], f[6], f[1], f[2], f[3], f[4], f[5], f[7])

    def pack(self, layout: ElfLayout) -> bytes:
        if layout.is64:
            return struct.pack(layout.phdr_fmt, self.ptype, self.flags, self.offset, self.vaddr,
                               self.paddr, self.filesz, self.memsz, self.align)
        return struct.pack(layout.phdr_fmt, self.ptype, self.offset, self.vaddr, self.paddr,
                           self.filesz, self.memsz, self.flags, self.align)


def scramble_segment(image: bytes, seg: Phdr, hash_: AddressHash) -> list[tuple[int, int, bytes]]:
    """Map a segment's file bytes unit by unit; return (physical address, flags, bytes) chunks."""
    end = seg.vaddr + seg.filesz
    if not (hash_.contains(seg.vaddr) and (seg.filesz == 0 or hash_.contains(end - 1))):
        raise ValueError(f"segment {seg.vaddr:#x}+{seg.filesz:#x} is not inside the hash range")
    data = image[seg.offset:seg.offset + seg.filesz]
    chunks = []
    unit_base = seg.vaddr & ~(hash_.unit - 1)
    while unit_base < end:
        lo, hi = max(seg.vaddr, unit_base), min(end, unit_base + hash_.unit)
        phys = hash_.forward(unit_base) + (lo - unit_base)
        chunks.append((phys, seg.flags, data[lo - seg.vaddr:hi - seg.vaddr]))
        unit_base += hash_.unit
    return chunks


def merge_runs(chunks: list[tuple[int, int, bytes]]) -> list[tuple[int, int, bytes]]:
    """Merge physically adjacent chunks with equal flags into runs, sorted by address."""
    runs: list[tuple[int, int, bytearray]] = []
    for phys, flags, data in sorted(chunks, key=lambda c: c[0]):
        if runs and runs[-1][1] == flags and runs[-1][0] + len(runs[-1][2]) == phys:
            runs[-1][2].extend(data)
        else:
            if runs and runs[-1][0] + len(runs[-1][2]) > phys:
                raise ValueError(f"scrambled segments overlap at {phys:#x}")
            runs.append((phys, flags, bytearray(data)))
    return [(phys, flags, bytes(data)) for phys, flags, data in runs]


def scramble_elf(image: bytes, hash_: AddressHash) -> tuple[bytes, int, int]:
    """Return the rewritten ELF, the number of segments scrambled, and the number of runs made."""
    layout = ElfLayout.for_ident(image[:16])
    (phoff,) = struct.unpack_from(layout.phoff_fmt, image, layout.phoff_at)
    phentsize, phnum = struct.unpack_from("<HH", image, layout.phentsize_at)
    phdrs = [Phdr.unpack(layout, image[phoff + i * phentsize:phoff + (i + 1) * phentsize])
             for i in range(phnum)]

    kept, chunks, scrambled = [], [], 0
    for ph in phdrs:
        if ph.ptype == PT_LOAD and ph.memsz and hash_.contains(ph.vaddr):
            chunks += scramble_segment(image, ph, hash_)
            scrambled += 1
        else:
            kept.append(ph)
    runs = merge_runs(chunks)

    out = bytearray(image)
    new_phdrs = list(kept)
    for phys, flags, data in runs:
        new_phdrs.append(Phdr(PT_LOAD, flags, len(out), phys, phys, len(data), len(data), 1))
        out.extend(data)
    while len(out) % 8:
        out.append(0)
    table_off = len(out)
    for ph in new_phdrs:
        out.extend(ph.pack(layout))
    struct.pack_into(layout.phoff_fmt, out, layout.phoff_at, table_off)
    struct.pack_into("<H", out, layout.phnum_at, len(new_phdrs))
    return bytes(out), scrambled, len(runs)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--base", type=lambda s: int(s, 0), default=0,
                        help="start of the hashed range in this ELF's address space (default 0, GPU-local)")
    parser.add_argument("--size", type=lambda s: int(s, 0), default=0x80000000)
    parser.add_argument("--unit", type=lambda s: int(s, 0), default=32)
    parser.add_argument("--slices", type=lambda s: int(s, 0), default=4)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    hash_ = AddressHash(args.base, args.size, args.unit, args.slices)
    hash_.self_check()
    with open(args.input, "rb") as f:
        image = f.read()
    out, scrambled, runs = scramble_elf(image, hash_)
    with open(args.output, "wb") as f:
        f.write(out)
    print(f"scrambled {scrambled} load segments into {runs} runs "
          f"(base={args.base:#x} size={args.size:#x} unit={args.unit} slices={args.slices}): {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
