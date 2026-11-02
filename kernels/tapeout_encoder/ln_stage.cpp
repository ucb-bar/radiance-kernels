// ln_stage.cpp: the LayerNorm stages (en_st_lnq1 / en_st_lnq2 / en_st_res1ln) compiled from
// kernel.cpp with x54-x63 reserved (Makefile), so their launches fit the renamer
// budget at 4 warps per core (EN_LN_OCC4; scripts/regs.py).  Without EN_LN_OCC4 this unit is empty.
#ifdef EN_LN_OCC4
#define EN_LN_TU 1
#include "kernel.cpp"
#endif
