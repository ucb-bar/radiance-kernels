OUTPUT_ARCH( "riscv" )
ENTRY(_start)

SECTIONS
{
  /* shared memory addresses start at 0x0; loader does not put stuff in there
   * so it needs to be populated at runtime. */
  .shared_mem 0 (NOLOAD) : { KEEP(*(.shared)) }

  . = 0x10000000;
  .init : { KEEP(*(.init)) }
  .text.init : { KEEP(*(.text.init)) }
  . = ALIGN(0x1000);
  .tohost : { *(.tohost) }
  . = ALIGN(0x1000);
  .text : { *(.text .text.*) }
  /* RUNG 2c: move .data/.rodata/.bss HIGH and change NOTHING else.  fb_padded passes 9/10 with
     its second RW segment at 0x...036000; fb_unpadded (0/8), fb_rung2 (fail) and FA (7/7 fail)
     all have it low.  This relocates only the data sections -- same code, same size, no blob --
     so a pass isolates placement as the variable in one change. */
  . = 0x10030000;
  .data : { *(.data .data.*) }
  . = ALIGN(0x1000);
  .rodata : { *(.rodata .rodata.*) }

  /* exception handling.  We don't need this, but the C++ compiler generates
   * them.  Tuck it away after .text */
  .eh_frame_hdr   : { *(.eh_frame_hdr) *(.eh_frame_entry .eh_frame_entry.*) }
  .eh_frame       : ONLY_IF_RO { KEEP (*(.eh_frame)) *(.eh_frame.*) }

  /* There's no support for TLS, edata or global_pointer currently.
   * Define them so that vx_start.S and vx_syscalls.c don't complain. */

  /* thread local storage */
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

  /* `*(.bss)` does NOT match -fdata-sections names like .bss._ZL16schedule_context, so every
     zero-init static was ORPHAN-placed by lld into the gap after .tohost (0x10001060) -- low, and
     outside every LOAD segment.  That is why relocating .data alone left .bss behind, and why
     schedule_context was never loaded.  Match the subsections too. */
  .bss : { *(.bss .bss.*) }
  __global_pointer = .;

  _end = .;
}
