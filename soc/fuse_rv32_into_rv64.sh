#!/usr/bin/env bash
set -euo pipefail

# toolchain
CROSS64="${CROSS64:-riscv64-unknown-elf}"
CC="${CC:-${CROSS64}-gcc}"
LD="${LD:-${CROSS64}-ld}"
OBJCOPY="${OBJCOPY:-${CROSS64}-objcopy}"
READELF="${READELF:-readelf}"

# inputs/outputs
RV32_ELF="${RV32_ELF:-muon.elf}"     # your rv32 payload
OUT="${OUT:-fused.elf}"              # final fused rv64 elf
ENTRY_SYM="${ENTRY_SYM:-_start}"
OFFSET_HEX=0x100000000               # +1_0000_0000

[[ -f "${RV32_ELF}" ]] || { echo "error: ${RV32_ELF} not found"; exit 1; }

# 1) build tiny rv64 carrier (compile as rv64; no libs; medany to avoid reloc truncation)
${CC} -march=rv64gc -mabi=lp64 -ffreestanding -nostdlib -mcmodel=medany -c start.S -o start.o
${CC} -march=rv64gc -mabi=lp64 -ffreestanding -nostdlib -mcmodel=medany -c main.c  -o main.o

# 2) read RV32 PT_LOADs
mapfile -t LOADS < <(
  ${READELF} -lW "${RV32_ELF}" | awk '
    $1=="LOAD" {
      off=$2; vaddr=$3; filesz=$5; memsz=$6; flags=$7; align=$8;
      gsub("0x","",off); gsub("0x","",vaddr);
      gsub("0x","",filesz); gsub("0x","",memsz); gsub("0x","",align);
      printf "%s %s %s %s %s %s\n", off, vaddr, filesz, memsz, flags, align;
    }')
[[ ${#LOADS[@]} -gt 0 ]] || { echo "error: no PT_LOAD segments in ${RV32_ELF}"; exit 1; }

WRK=$(mktemp -d)
trap 'rm -rf "$WRK"' EXIT

SEG_PHDRS=()
SEG_SECTIONS=()
SEG_OBJS=()

idx=0
for line in "${LOADS[@]}"; do
  read -r off_hex vaddr_hex filesz_hex memsz_hex flags align_hex <<<"${line}"
  off=$((16#${off_hex}))
  vaddr=$((16#${vaddr_hex}))
  filesz=$((16#${filesz_hex}))
  memsz=$((16#${memsz_hex}))
  newv=$(( vaddr + OFFSET_HEX ))

  # numeric PT_LOAD flags (PF_R=0x4, PF_W=0x2, PF_X=0x1)
  fl=0x4
  [[ "${flags}" == *"W"* ]] && fl=$((fl|0x2))
  [[ "${flags}" == *"E"* ]] && fl=$((fl|0x1))

  # extract bytes for this segment
  bin="${WRK}/seg_${idx}.bin"
  dd if="${RV32_ELF}" of="${bin}" bs=1 skip=${off} count=${filesz} status=none

  # 3) turn raw bytes into a *binary object* with no ABI: ld -r -b binary
  #    then rename its default .data section to a unique name so we can place it
  obj="${WRK}/seg_${idx}.o"
  ${LD} -r -b binary "${bin}" -o "${obj}"
  ${OBJCOPY} --rename-section .data=.rv32.seg${idx} "${obj}"
  SEG_OBJS+=("${obj}")

  # phdrs + sections
  SEG_PHDRS+=("  seg${idx} PT_LOAD FLAGS(0x$(printf %X ${fl}));")
  SEG_SECTIONS+=("  . = 0x$(printf %X ${newv});
  .rv32.seg${idx} 0x$(printf %X ${newv}) : { KEEP(*(.rv32.seg${idx})) } :seg${idx}
  . = 0x$(printf %X $((newv+memsz)));")
  idx=$((idx+1))
done

# 4) synthesize full GNU ld script (single PHDRS block)
LD_SCRIPT="${WRK}/fused.ld"
cat > "${LD_SCRIPT}" <<EOF
/* fused ld script */
ENTRY(${ENTRY_SYM})

PHDRS {
  text PT_LOAD FLAGS(0x5);  /* R|X */
  data PT_LOAD FLAGS(0x6);  /* R|W */
$(printf "%s\n" "${SEG_PHDRS[@]}")
}

__rom_base  = 0x80000000;
__stack_top = 0x80080000;

SECTIONS
{
  . = __rom_base;

  .text : {
    KEEP(*(.text.start))
    *(.text .text.*)
    *(.rodata .rodata.*)
  } :text

  . = ALIGN(4096);
  .data : {
    *(.data .data.*)
    *(.sdata .sdata.*)
  } :data

  . = ALIGN(8);
  .bss (NOLOAD) : {
    *(.bss .bss.*)
    *(.sbss .sbss.*)
    *(COMMON)
  } :data

  /* mirrored rv32 segments as raw bytes */
$(printf "%s\n" "${SEG_SECTIONS[@]}")

  __rv32_min  = 0x110000000;
  __rv32_max  = .;
  __rv32_size = __rv32_max - __rv32_min;
}
EOF

# 5) final link: rv64 objs + binary segment objs
${LD} -o "${OUT}" -T "${LD_SCRIPT}" start.o main.o "${SEG_OBJS[@]}"

echo "wrote ${OUT}"
echo "verify:"
echo "  readelf -lW ${OUT}"

