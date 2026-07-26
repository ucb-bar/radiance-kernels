// AFTER = PROPER WEIGHT-STATIONARY: one TM=256 x TN=64 tile.
// The mesh loop FSM loads each B[k] tile (64x64 fp8 = 4KB) ONCE into SMEM per k-tile
// (32 k-tiles) and streams ALL 256 M-rows through it -> each weight byte DMA'd from DRAM
// EXACTLY ONCE. Effective weight-tile move-ins = 32 (== the 128KB weight, 1x).
#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <stdint.h>
extern "C" uint32_t __mu_num_warps = 2;
#include "data"
static const uint8_t A_lut[64][16]={0}; static const uint8_t B_lut[64][16]={0}; static const uint8_t C_lut[64][16]={0};
#include "mxgemm_lib.hpp"

constexpr GemmConfig CFG{ .TILE_M=256, .TILE_N=64, .TILE_K=64, .DATATYPE=GemmDatatype::FP8, .QUANT_OUTPUT=false };

#define VERIFY_COUNT (MATMUL_M*MATMUL_N)
__global uint16_t C_raw[VERIFY_COUNT]={0};
static const uint16_t* gold_raw=&C_out_bf16[0][0];

struct KernelArgs { uint8_t *C; };
static void kernel_body(void* raw,uint32_t tid,uint32_t tpb,uint32_t tbid){
  auto*a=reinterpret_cast<KernelArgs*>(raw);
  mxgemm<CFG>(MATMUL_M, MATMUL_N, MATMUL_K, a->C, tid, tpb, tbid);
}
static KernelArgs kernel_args;
#define VERIFY_LANES (MU_NUM_THREADS*MU_NUM_CORES)
__global uint32_t lane_errors[VERIFY_LANES]={0};
static void verify_body(void*,uint32_t tid,uint32_t tpb,uint32_t){ uint32_t e=0;
  for(uint32_t i=tid;i<VERIFY_COUNT;i+=tpb) if(C_raw[i]!=gold_raw[i]) e++; lane_errors[tid]=e; mu_fence(); }
static inline uint32_t hart_id(){uint32_t i;asm volatile("csrr %0, mhartid":"=r"(i)::"memory");return i;}
int main(){ kernel_args={reinterpret_cast<uint8_t*>(reinterpret_cast<uint32_t>((uint16_t*)C_raw))};
  mu_schedule(kernel_body,&kernel_args,2); mu_barrier(0,MU_NUM_CORES); mu_fence();
#ifndef DRAIN_ITERS
#define DRAIN_ITERS 0u
#endif
  for(volatile uint32_t d=0;d<DRAIN_ITERS;d++) asm volatile("":::"memory");
  mu_schedule(verify_body,nullptr,1); mu_barrier(0,MU_NUM_CORES); if(hart_id()!=0){for(;;){}} mu_fence();
  uint32_t t=0; for(uint32_t i=0;i<VERIFY_LANES;i++) t+=lane_errors[i];
  uint32_t code=t?((t<<1)|1u):0u; asm volatile(".insn i 0x73,0,x0,%0,0"::"r"(code):"memory"); return 0; }
