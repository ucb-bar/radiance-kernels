/* mu_link_nightly.ld: Muon link script for the nightly kernels.
 *
 * Differences from mu_link.ld:
 *   1. .bss and .sbss (including the .bss.* names -fdata-sections produces) are placed inside
 *      .data, so every zero-initialised static is part of the loaded image and really is zero.
 *      mu_link.ld's `*(.bss)` does not match `.bss.*`, and mu_start.S never clears .bss, so
 *      there such statics hold stale DRAM.
 *   2. Four fixed-address data regions for large tensors (512 MiB each of device
 *      0x0000_0000..0x7FFF_FFFF):
 *        .region1  0x2000_0000
 *        .region2  0x4000_0000
 *        .region3  0x6000_0000   (the warp stacks sit below 0x7F00_0000)
 *      Kernels put a data blob there (lib/nightly or the kernel's gen_data.py) and address
 *      tensors by fixed offsets that the host shares through a generated header.
 *      Code, small data and region 0 tensors stay at 0x1000_0000.
 *      Without the address hash, each region is one GPU L2 slice / DRAM channel.  RadianceHBMConfig
 *      hashes every 32 B over all 4 slices, so there a region does not select a slice.
 */
OUTPUT_ARCH( "riscv" )
ENTRY(_start)

/* one PT_LOAD per region: without PHDRS lld merges the regions into one segment that spans the
 * 1.5 GiB between them, and the fused ELF becomes that large */
PHDRS
{
  init PT_LOAD FLAGS(5);
  host PT_LOAD FLAGS(6);
  text PT_LOAD FLAGS(5);
  data PT_LOAD FLAGS(6);
  r1   PT_LOAD FLAGS(6);
  r2   PT_LOAD FLAGS(6);
  r3   PT_LOAD FLAGS(6);
}

SECTIONS
{
  .shared_mem 0 (NOLOAD) : { KEEP(*(.shared)) } :NONE

  . = 0x10000000;
  .init : { KEEP(*(.init)) } :init
  .text.init : { KEEP(*(.text.init)) }
  . = ALIGN(0x1000);
  .tohost : { *(.tohost) } :host
  . = ALIGN(0x1000);
  .text : { *(.text .text.*) } :text
  . = ALIGN(0x1000);
  .data : {
    *(.data .data.*)
    *(.sdata .sdata.*)
    . = ALIGN(64);
    PROVIDE(__nightly_bss_start = .);
    *(.bss .bss.*)
    *(.sbss .sbss.*)
    *(COMMON)
    PROVIDE(__nightly_bss_end = .);
    /* a non-empty tail so the zeroed statics above are inside the segment's FileSiz */
    LONG(0x6E1A0C11)
  } :data
  . = ALIGN(0x1000);
  .rodata : { *(.rodata .rodata.*) *(.srodata .srodata.*) } :data
  . = ALIGN(0x1000);
  .region0 : { KEEP(*(.region0 .region0.*)) }

  .eh_frame_hdr   : { *(.eh_frame_hdr) *(.eh_frame_entry .eh_frame_entry.*) }
  .eh_frame       : ONLY_IF_RO { KEEP (*(.eh_frame)) *(.eh_frame.*) }

  .tdata :
    {
      PROVIDE_HIDDEN (__tdata_start = .);
      *(.tdata .tdata.* .gnu.linkonce.td.*)
      PROVIDE_HIDDEN (__tdata_end = .);
    }
  PROVIDE (_edata = .);
  PROVIDE (__tdata_size = SIZEOF (.tdata));
  PROVIDE (__tcb_aligned_size = 0);
  PROVIDE (__tbss_size = 0);
  __global_pointer = .;
  _end = .;

  .region1 0x20000000 : { KEEP(*(.region1 .region1.*)) } :r1
  .region2 0x40000000 : { KEEP(*(.region2 .region2.*)) } :r2
  .region3 0x60000000 : { KEEP(*(.region3 .region3.*)) } :r3
}
