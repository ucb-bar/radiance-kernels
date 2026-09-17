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
  /* FA_MAIN_EARLY: give main a slot at the FRONT of .text.  FA's main lands at 0x100047f8,
     about 18 KB into the image; fpga_bringup's is about 8 KB in.  That is the only dimension on
     which the two kernels differ, so if the bytes are present (fa_img) and main still never
     executes, the question is whether the far address can be fetched at all.  This moves main to
     roughly 0x10002000 and changes nothing else. */
  .text : { *(.text.faearly) *(.text .text.*) }
  . = ALIGN(0x1000);
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

  /* FA: -fdata-sections emits .bss._ZL<name>, which *(.bss) does not match, so lld
     orphan-places every zero-init static.  In this image that put C_scale_factors at
     0x10001060 and schedule_context at 0x10001880, past the RW segment FileSiz of 0x48,
     i.e. NOBITS and never written -- they hold stale DRAM.  This is the ONE-LINE half of
     the fix; .bss is still NOBITS, so it also needs zeroing at startup. */
  /* Emit the range symbols.  Neither script defined _edata, which is why mu_start.S's
     "clear BSS segment" block could not have linked and is commented out.  With the wildcard
     above, .bss is a real segment at the end of the image, so __bss_start.._end is a range that
     actually contains it -- under the shared script it did not, and zeroing that range was a
     fix that would compile, run, and do nothing. */
  PROVIDE(__bss_start = .);
  .bss : { *(.bss .bss.*) }
  PROVIDE(__bss_end = .);
  __global_pointer = .;

  _end = .;
}
