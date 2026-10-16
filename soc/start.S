/* minimal crt0: set stack, jump to main, then report main's return value to fesvr */
    .section .text.start, "ax"
    .globl _start
_start:
    la   sp, __stack_top
    call main
    /* HTIF exit: tohost = (code << 1) | 1.  Without this the run only ends through the
       RTL stop() on GPU idle, which VCS honors but FireSim does not synthesize. */
1:  ld   t0, tohost          /* wait for fesvr to consume any pending message */
    bnez t0, 1b
    slli a0, a0, 1
    ori  a0, a0, 1
    sd   a0, tohost, t1
2:  j    2b

