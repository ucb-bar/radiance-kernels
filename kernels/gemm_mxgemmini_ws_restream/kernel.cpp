// BEFORE = NAIVE M-OUTER RE-STREAM: compute the SAME C[256,64] as 4 M-blocks of TM=64.
// Each block runs its own full K-loop and RE-DMAs the weight B from DRAM.
// Effective weight-tile move-ins = 4 blocks x 32 k-tiles = 128 (== the 128KB weight, 4x).
// This is the "re-DMA B per K-tile / per M-block" dataflow the earlier re-streaming impl used.
#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <stdint.h>
extern "C" uint32_t __mu_num_warps = 2;
#include "data"
static const uint8_t A_lut[64][16]={0}; static const uint8_t B_lut[64][16]={0}; static const uint8_t C_lut[64][16]={0};
// re-pointable A base (data array was renamed to A_in_data)
static const uint8_t *A_in = &A_in_data[0][0];
#include "mxgemm_lib.hpp"

#define WS_MB   64
#define WS_MT   4     // 4 M-blocks -> M=256
constexpr GemmConfig CFG{ .TILE_M=WS_MB, .TILE_N=64, .TILE_K=64, .DATATYPE=GemmDatatype::FP8, .QUANT_OUTPUT=false };

#define VERIFY_COUNT (MATMUL_M*MATMUL_N)
__global uint16_t C_raw[VERIFY_COUNT]={0};
static const uint16_t* gold_raw=&C_out_bf16[0][0];

// M-outer re-stream driver. thread-0 drives the mesh; whole block cooperates on move-out.
template <GemmConfig C>
static void mxgemm_restream(uint32_t dim_k, uint8_t *C_gmem, uint32_t tid, uint32_t tpb) {
    const auto warps = tpb / MU_NUM_THREADS;
    const uint32_t bm = C.TILE_M, bn = C.TILE_N;
    // config once (dims identical every block); models the config-once "reuse" variant
    if (tid==0) configure_mxgemmini<C>(bm, bn, dim_k);

    for (uint32_t m=0; m<WS_MT; m++) {
        if (tid==0) {
            A_in = &A_in_data[0][0] + (uint32_t)m*WS_MB*MATMUL_K;         // this block's A rows
            const uint8_t *A_sc = &A_scales_tiled[0][0] + (uint32_t)m*MATMUL_GK*WS_MB; // block-contiguous A scales
            const uint8_t *B_sc = &B_scales_col[0][0];                    // shared weight scales
            gemmini_flush(0);                                             // reset accumulator only
            int tk=0;
            copy_gmem_to_smem_async<C>(bm, bn, dim_k, 0,0, tk);
            load_scale_factors(calculate_scale_factor_smem_addr<false>(tk),
                calculate_scale_factor_gmem_addr<C,false>(A_sc, tk, bm, bn), C.SCALE_FACTORS_PER_TILE_A());
            load_scale_factors(calculate_scale_factor_smem_addr<true>(tk),
                calculate_scale_factor_gmem_addr<C,true>(B_sc, tk, bm, bn), C.SCALE_FACTORS_PER_TILE_B());
            load_lut<C>();
            mu_fence_smem(); gemmini_fence();
            for (; ((uint32_t)tk*C.TILE_K)<dim_k; tk++) {
                const auto odd=(tk&1); const auto last=(((uint32_t)tk+1)*C.TILE_K)>=dim_k;
                gemmini_mxquant_config_mvout(
                    rad_device_to_host_address(reinterpret_cast<uint32_t>(&C_scale_factors[0])),
                    C.PE_TILES_I(),C.PE_TILES_J(),C.PE_TILES_K(), odd,odd, QUANT_LUT_UPDATE_GRANULARITY);
                if(!last) copy_gmem_to_smem_async<C>(bm,bn,dim_k,0,0,tk+1);
                matmul_tile_async<C>(tk,last);
                load_scale_factors(calculate_scale_factor_smem_addr<false>(tk+1),
                    calculate_scale_factor_gmem_addr<C,false>(A_sc,tk+1,bm,bn), C.SCALE_FACTORS_PER_TILE_A());
                load_scale_factors(calculate_scale_factor_smem_addr<true>(tk+1),
                    calculate_scale_factor_gmem_addr<C,true>(B_sc,tk+1,bm,bn), C.SCALE_FACTORS_PER_TILE_B());
                mu_fence_smem(); gemmini_fence();
            }
            gemmini_fence();
        }
        mu_barrier(1, warps);
        auto C_smem=reinterpret_cast<const __shared uint8_t*>(C.SPAD_DEST()*DIM);
        uint8_t *C_tile = C_gmem + (uint32_t)m*C.TILE_M_QUANT()*C.TILE_N_QUANT()*C.OUT_ELEM_SIZE();
        copy_smem_to_gmem_simt<C.TILE_M_QUANT(),C.TILE_N_QUANT(),C.OUT_ELEM_SIZE()>(C_smem,C_tile,tid,tpb);
        mu_barrier(2, warps);
    }
}

struct KernelArgs { uint8_t *C; };
static void kernel_body(void* raw,uint32_t tid,uint32_t tpb,uint32_t){ auto*a=reinterpret_cast<KernelArgs*>(raw);
  mxgemm_restream<CFG>(MATMUL_K, a->C, tid, tpb); }
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
