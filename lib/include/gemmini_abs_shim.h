#ifndef GEMMINI_ABS_SHIM_H
#define GEMMINI_ABS_SHIM_H
// Force-included (via -include in common.mk) ahead of gemmini.h.
//
// gemmini.h's scale_and_sat() calls abs(q) where the accumulator type acc_t is uint64_t.
// Once real libc headers are on the include path (radiance-kernels supplies newlib via
// MU_LIBC_INCLUDE, needed for any kernel that includes gemmini.h), abs() is ambiguous
// between C's abs(int) and C++'s abs(long)/abs(long long). Providing an exact-match
// overload for the unsigned accumulator type resolves the call unambiguously WITHOUT
// patching the gemmini submodule. abs() of an unsigned value is identity, which is exactly
// what the intended computation is.
//
// Guarded for non-assembly only: common.mk applies -include to the link step too, which
// compiles tohost.S; __ASSEMBLER__ keeps this a no-op there.
#ifndef __ASSEMBLER__
static inline unsigned long long abs(unsigned long long x) { return x; }
static inline unsigned long      abs(unsigned long x)      { return x; }
#endif

#endif // GEMMINI_ABS_SHIM_H
