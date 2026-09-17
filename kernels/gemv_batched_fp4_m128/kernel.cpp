// TinyLlama BATCHED DECODE projection on the MX mesh (fp4, weight-stationary).
//
//   out[M, N] = X[M, K] @ W[K, N],   K = 2048,  N = 128 (one projection tile),
//   M = batch of decode steps (or heads) = 128.
//
// This is the key decode lever. At M=1 the same projection is a SIMT GEMV
// (~8 cyc/MAC, weight-BANDWIDTH-bound: each weight feeds
// exactly one MAC). Batching B decode steps raises M so every weight loaded into
// the 16x16 systolic array feeds M MACs -> arithmetic intensity climbs to ~M and
// the projection becomes COMPUTE-bound on the MX mesh instead of BW-bound.
//
// The tile is SQUARE here (TILE_M=TILE_N=128). The stock mxgemm_lib only
// allowed square tiles; this copy differentiates the A/B scale-factor counts
// (SCALE_FACTORS_PER_TILE_A/_B) so the smaller M=32/64 variants (TILE_M != TILE_N) verify correctly.
//
// This variant is fixed at M=128 (fp4); the committed `data` carries that shape.
#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <stdint.h>
#ifndef MX_NUM_WARPS
#define MX_NUM_WARPS 2
#endif
extern "C" uint32_t __mu_num_warps = MX_NUM_WARPS;
#include "data"
static const uint8_t A_lut[64][16]={0}; static const uint8_t B_lut[64][16]={0}; static const uint8_t C_lut[64][16]={0};
static const uint8_t *A_in = &A_in_hw[0][0];

#include "mxgemm_lib.hpp"
constexpr GemmConfig GEMM_CFG{ .TILE_M=MATMUL_M, .TILE_N=MATMUL_N, .TILE_K=128, .DATATYPE=GemmDatatype::FP4, .QUANT_OUTPUT=false };
struct KernelArgs { uint8_t *C; uint32_t M,N,K; };
void kernel_body(void* raw,uint32_t tid,uint32_t tpb,uint32_t tbid){ auto*a=reinterpret_cast<KernelArgs*>(raw); mxgemm<GEMM_CFG>(a->M,a->N,a->K,a->C,tid,tpb,tbid); }
static KernelArgs kernel_args;
#define VERIFY_LANES (MU_NUM_THREADS*MU_NUM_CORES)
__global uint32_t lane_errors[VERIFY_LANES]={0};
static void verify_body(void*,uint32_t tid,uint32_t tpb,uint32_t){ uint32_t e=0; for(uint32_t i=tid;i<VERIFY_COUNT;i+=tpb) if(C_raw[i]!=gold_raw[i]) e++; lane_errors[tid]=e; mu_fence(); }
static inline uint32_t hart_id(){uint32_t i;asm volatile("csrr %0, mhartid":"=r"(i)::"memory");return i;}
int main(){ kernel_args={reinterpret_cast<uint8_t*>(reinterpret_cast<uint32_t>((uint16_t*)C_raw)),MATMUL_M,MATMUL_N,MATMUL_K};
  mu_schedule(kernel_body,&kernel_args,MX_NUM_WARPS); mu_barrier(0,MU_NUM_CORES); mu_fence();
#ifndef DRAIN_ITERS
#define DRAIN_ITERS 0u
#endif
  for(volatile uint32_t d=0;d<DRAIN_ITERS;d++) asm volatile("":::"memory");
  mu_schedule(verify_body,nullptr,1); mu_barrier(0,MU_NUM_CORES); if(hart_id()!=0){for(;;){}} mu_fence();
  uint32_t t=0; for(uint32_t i=0;i<VERIFY_LANES;i++) t+=lane_errors[i];
  uint32_t code=t?((t<<1)|1u):0u; asm volatile(".insn i 0x73,0,x0,%0,0"::"r"(code):"memory"); return 0; }
