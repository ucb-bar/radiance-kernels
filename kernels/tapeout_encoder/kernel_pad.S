/* EN_KERNEL_ALIGN: linked between fa_stage.mu.o and kernel.mu.o (EN_FA_FIRST=1). Starts the
   kernel code at a 16 KB boundary (the L0i is 16 KB direct-mapped), so a change in the attention
   code's size does not move the other stages' code in the instruction cache. */
    .section .text.en_kernel_pad, "ax", @progbits
    .balign 16384
    .globl en_kernel_pad
en_kernel_pad:
    .dword 0
