/* base rv64 layout; entry at 0x8000_0000 */
ENTRY(_start)

/* optional: define program headers for rv64 regions (not strictly required) */
PHDRS {
  text PT_LOAD FLAGS(R X);
  data PT_LOAD FLAGS(R W);
}

/* 512 KiB for a tiny demo; tune as needed */
__rom_base  = 0x80000000;
__ram_base  = 0x80000000;
__stack_top = 0x80080000;

SECTIONS
{
  . = __rom_base;

  .text : {
    KEEP(*(.text.start))
    *(.text .text.*)
    *(.rodata .rodata.*)
  } :text

  . = ALIGN(4K);
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

  /* rv32 mirrored segments (auto-appended by the script) */
}

