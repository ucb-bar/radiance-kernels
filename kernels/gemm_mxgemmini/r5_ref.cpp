#include <stdint.h>
#include <mu_schedule.h>
#include <mu_intrinsics.h>
#include "mxgemm.data.r5.h"
static const uint8_t A_lut[64][16] = {0};
static const uint8_t B_lut[64][16] = {0};
static const uint8_t C_lut[64][16] = {0};
#include "mxgemm_lib.hpp"
extern "C" int vx_printf(const char *, ...);
constexpr GemmConfig CFG{ .TILE_M=16, .TILE_N=16, .TILE_K=32, .DATATYPE=GemmDatatype::FP8, .QUANT_OUTPUT=false };
void mxgemm_entry(void *a, uint32_t tid, uint32_t th, uint32_t tb){
  auto Cg = reinterpret_cast<uint8_t*>(0x40000000);
  mxgemm<CFG>(CFG.TILE_M, CFG.TILE_N, CFG.TILE_K, Cg, tid, th, tb);
  if(tid==0 && tb==0){ gemmini_fence();
    volatile uint16_t* C = (volatile uint16_t*)0x40000000;
    vx_printf("MXOUT\n");
    for(int i=0;i<MATMUL_M*MATMUL_N;i++) vx_printf("%04x\n",(unsigned)(C[i]&0xffff));
    vx_printf("MXEND\n"); }
}
int main(){ mu_schedule(mxgemm_entry, nullptr, 2); return 0; }
