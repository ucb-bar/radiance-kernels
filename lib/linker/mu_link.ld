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
  .text : { *(.text) }
  . = ALIGN(0x1000);
  .data : { *(.data) }
  . = ALIGN(0x1000);
  .rodata : { *(.rodata) }

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

  .bss : { *(.bss) }
  __global_pointer = .;

  _end = .;
}
