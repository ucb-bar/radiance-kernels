/* minimal crt0: set stack and jump to main */
    .section .text.start, "ax"
    .globl _start
_start:
    la   sp, __stack_top
    call main
1:  j    1b

