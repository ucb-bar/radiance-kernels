#ifndef __MU_INTRINSICS_H__
#define __MU_INTRINSICS_H__

inline float mu_fexp(float arg) {
    float output;
    asm volatile ("fexp.s %0, %1" :: "r"(output), "r"(arg));
    return output;
}

// TODO: half?

#endif // __MU_INTRINSICS_H__