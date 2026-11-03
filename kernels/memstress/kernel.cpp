// memstress: GMEM / SMEM access-pattern stress kernel with per-warp cycle records.
//
// Every participating warp runs two phases over a per-core (or shared) region:
//   phase S: MS_INSTR/MS_NWARPS warp-level stores of value(addr), then `fence` (L0d writeback,
//            skipped when MS_FENCE=0, which leaves the lines resident and makes phase L an L0d hit)
//   phase L: the same addresses loaded back and checked against value(addr)
// Timestamps (mcycle) are taken after a core barrier before each phase and after each phase, and
// written to a record block in GMEM that the rv64 host reads after all cores have finished:
//   REC[g][w]  (64 B, at MS_REC_BASE + (g*8+w)*64): magic t0 t1 t1f t2 t3 sum_lane0 -
//   ERR[g][w] (4 B each, at MS_REC_BASE + 0x1000): nonzero if any lane of the warp read wrong data
// where g = cluster*2 + core is the global core index and w the warp.
//
// Address of warp-instruction k (k = i*MS_NWARPS + w, i the warp's loop index), lane l:
//   MS_PATTERN 0  lane stride:  off = ((k*16 + l) * MS_STRIDE) mod MS_WS
//                 stride 4 = one 64 B line per instruction, 8 = 2 lines, 64 = 16 lines
//   MS_PATTERN 1  random word:  off = (hash(k*16 + l) * 4) mod MS_WS
//   MS_PATTERN 2  broadcast:    off = (k * MS_STRIDE) mod MS_WS, same for every lane
#include <stdint.h>
#include <mu_intrinsics.h>
#include <mu_schedule.h>

#ifndef MS_SPACE
#define MS_SPACE 0          // 0 GMEM, 1 SMEM
#endif
#ifndef MS_PATTERN
#define MS_PATTERN 0
#endif
#ifndef MS_STRIDE
#define MS_STRIDE 4
#endif
#ifndef MS_WS
#define MS_WS (2u << 10)    // working set per core in bytes, power of two
#endif
#ifndef MS_NWARPS
#define MS_NWARPS 8
#endif
#ifndef MS_INSTR
#define MS_INSTR 4096       // warp-level memory instructions per core per phase
#endif
#ifndef MS_CORES
#define MS_CORES 0xFu       // bit (cluster*2 + core) set: that core participates
#endif
#ifndef MS_FENCE
#define MS_FENCE 1          // 1: every warp fences between the store and the load phase (writes the
                            //    L0d back, so the load phase measures the L1/DRAM path)
                            // 0: no fence, so the load phase re-reads whatever stayed in the L0d
                            // 2: only warp 0 fences.  The taped-out RTL cannot execute a fence from
                            //    more than one warp of a core at all, so this is the only setting in
                            //    which both RTLs run the same flushing workload
#endif
#ifndef MS_PUBLISH
#define MS_PUBLISH 1        // 0: skip the end-of-kernel flush/publish entirely (no host read-back);
                            //    used to separate a mid-kernel flush from the end-of-run flushes
#endif
#ifndef MS_L1_FLUSH
#define MS_L1_FLUSH 0       // 1: the cluster L1 is incoherent (radiance L1IncoherentKey): publishing also
                            //    writes its flush register, which writes its dirty lines to the GPU L2
#endif
#define MS_L1_FLUSH_REG 0x82000u   // cluster-local: cluster peripherals (0x80000) + 0x2000, + 0x100 per bank
#ifndef MS_L1_BANKS
#define MS_L1_BANKS 1       // cluster L1 banks (radiance L1BanksKey); each has its own flush register
#endif
#ifndef MS_SHARED_REGION
#define MS_SHARED_REGION 0  // 1: every core uses the same region; 0: private region per core
#endif
#ifndef MS_GMEM_BASE
#define MS_GMEM_BASE 0x18000000u   // inside the window the kernel image and record block live in
#endif
#define MS_REC_BASE  0x1F000000u
#define MS_ERR_BASE  (MS_REC_BASE + 0x1000u)
#define MS_DONE_BASE (MS_REC_BASE + 0x2000u)   // one word per core, magic once its records are out
#define MS_DONE_MAGIC 0xD09E0000u
#define MS_ITERS     (MS_INSTR / MS_NWARPS)
#ifndef MS_UNROLL
#define MS_UNROLL 1          // loads issued back-to-back per batch: the memory-level parallelism
#endif
#define MS_STR(x) #x
#define MS_XSTR(x) MS_STR(x)
#define MS_PRAGMA_UNROLL _Pragma(MS_XSTR(unroll MS_UNROLL))
#define MS_PRAGMA_UNROLL_FULL _Pragma("unroll")

static inline uint32_t rd_mcycle() {
  uint32_t c;
  asm volatile("csrr %0, 0xB00" : "=r"(c));
  return c;
}
// mcycle read that cannot issue before `dep` is available (branch on it first)
static inline uint32_t rd_mcycle_after(uint32_t dep) {
  uint32_t c;
  asm volatile("beq %1, zero, 1f\n1:\n\tcsrr %0, 0xB00" : "=r"(c) : "r"(dep));
  return c;
}
static inline uint32_t hash32(uint32_t x) {
  x *= 0x9E3779B1u; x ^= x >> 15; x *= 0x85EBCA77u; x ^= x >> 13;
  return x;
}
// The stored value is derived from its own address, so a load that returns the wrong line, a stale
// line or an unwritten line is caught.
//
// MS_HASH 0 (default) uses the address itself.  MS_HASH 1 uses hash32, which costs two 32-bit
// multiplies.  This core builds rocket's iterative MulDiv with mulUnroll=1 and splits a 16-lane
// multiply into two 8-lane slices, so one `mul` takes about 33 cycles; two of them per load put
// 60-70 cycles of arithmetic on the critical path of the timed loop, which is more than the memory
// access itself.  Measurements taken with MS_HASH 1 measure the multiplier, not the memory system.
#ifndef MS_HASH
#define MS_HASH 0
#endif
// MS_CONTIG: how the unroll index walks addresses.
//
//   0 (default)  warp w's accesses are at ((i+j)*MS_NWARPS + w) * 64, so the unroll partner is
//                MS_NWARPS*64 away and the loop step is MS_UNROLL*MS_NWARPS*64.  Both are multiples
//                of 128 for the usual warp counts, which means every access a given warp makes is
//                congruent to w*64 mod 128.  On a machine that interleaves DRAM channels at 32 B
//                with four channels (period 128 B), an even warp then only ever reaches
//                addr[6:5] = 0,1 and an odd warp only 2,3.  All four channels are covered only if
//                the warps stay balanced, which they do not have to.
//   1            warp w's accesses are at (((i/MS_UNROLL)*MS_NWARPS + w)*MS_UNROLL + j) * 64, so the
//                MS_UNROLL accesses of one iteration are CONTIGUOUS.  At MS_UNROLL=2 each warp
//                covers 128 contiguous bytes per iteration and reaches all four channels by itself.
//                Same total footprint and same number of accesses; only the order changes.
#ifndef MS_CONTIG
#define MS_CONTIG 0
#endif
#ifndef MS_ASM
#define MS_ASM 0
#endif
// MS_STREAM 1 (with MS_ASM 2): each warp streams its own contiguous MS_ITERS*64 B chunk once,
// 32 lines per block, instead of re-reading one resident block.  Every load is a new line, so the
// load phase is a pure DRAM read stream once the chunk exceeds the caches.
#ifndef MS_STREAM
#define MS_STREAM 0
#endif
// MS_STORE 0 skips the store phase (and its fence).  The load phase then reads whatever the
// backing store holds, so it verifies nothing; use it with MS_ASM 2 to measure read bandwidth
// without the store phase's write-backs.
#ifndef MS_STORE
#define MS_STORE 1
#endif
// MS_SKEW (with MS_STREAM): stream s = core*MS_NWARPS + warp starts MS_SKEW*s bytes past its chunk.
// Chunks are multiples of the cluster L1's 8 KiB set span, so without a skew every stream sits on
// the same set index at the same time, and the L1 MSHR file admits one primary miss per set index.
#ifndef MS_SKEW
#define MS_SKEW 0
#endif
// MS_REGION_STRIDE: core g adds (g / MS_REGION_GROUP) * MS_REGION_STRIDE to its base, so cores can
// be spread over GPU regions that map to different L2 slices and DRAM channels.  Under
// RadianceHBMConfig the GPU slices own cluster-local 0x0000_0000 and 0x4000_0000 (1 GiB each).
#ifndef MS_REGION_STRIDE
#define MS_REGION_STRIDE 0u
#endif
#ifndef MS_REGION_GROUP
#define MS_REGION_GROUP 2u
#endif
#ifndef MS_SUBWORD
#define MS_SUBWORD 0
#endif
#if MS_HASH == 1
static inline uint32_t value_of(uint32_t addr) { return hash32(addr ^ 0xA5A5A5A5u); }
#else
static inline uint32_t value_of(uint32_t addr) { return addr; }
#endif
// multiply-free scrambler, for the random access pattern
static inline uint32_t mix32(uint32_t x) {
  x ^= x >> 16; x += x << 3; x ^= x >> 5; x += x << 7; x ^= x >> 11;
  return x;
}

#if MS_CONTIG == 1
#define MS_IDX(i, j, w) ((((i) / MS_UNROLL) * MS_NWARPS + (w)) * MS_UNROLL + (j))
#else
#define MS_IDX(i, j, w) (((i) + (j)) * MS_NWARPS + (w))
#endif

static inline uint32_t offset_of(uint32_t k, uint32_t lane) {
#if MS_PATTERN == 0
  return ((k * 16u + lane) * MS_STRIDE) & (MS_WS - 1u);
#elif MS_PATTERN == 1
  return (mix32(k * 16u + lane) << 2) & (MS_WS - 1u);
#else
  (void)lane;
  return (k * MS_STRIDE) & (MS_WS - 1u);
#endif
}

static inline void mem_store(uint32_t addr, uint32_t v) {
#if MS_SPACE == 1
  asm volatile("sw.shared %1, 0(%0)" :: "r"(addr), "r"(v) : "memory");
#else
  asm volatile("sw %1, 0(%0)" :: "r"(addr), "r"(v) : "memory");
#endif
}
static inline uint32_t mem_load(uint32_t addr) {
  uint32_t v;
#if MS_SPACE == 1
  asm volatile("lw.shared %0, 0(%1)" : "=r"(v) : "r"(addr) : "memory");
#else
  asm volatile("lw %0, 0(%1)" : "=r"(v) : "r"(addr) : "memory");
#endif
  return v;
}


// A load loop with no address arithmetic in it.
//
// MS_PATTERN 0 with stride 4 walks consecutive 64-byte lines, so a warp's k-th access sits at
// `base + lane*4 + k*64`.  For a working set up to 4 KiB that whole `k*64` fits in a 12-bit load
// immediate, which means the block below needs one base register and no adds, ands or shifts at
// all.  The C loop it replaces spent four instructions per load (addi, andi, addi, lw) on a machine
// that admits one instruction per cycle, so it could not have beaten 4 cycles per 64-byte line
// whatever the memory system did underneath it.
//
//   MS_ASM 1  32 loads and 32 xors: every value is folded into a checksum, two instructions per
//             load.  This is the floor for a loop that actually looks at its data.
//   MS_ASM 2  32 loads and nothing else, one instruction per load.  Nothing consumes the values so
//             it verifies nothing; it measures how fast the L0d can be driven at all.
//
// Sixteen destinations, each written twice per block, put a WAW on any one of them sixteen loads
// apart, far enough that it never gates.  Sixteen destinations plus the base, the accumulator and
// the loop counter also stay inside the ~31 architectural registers a warp gets at eight-warp
// occupancy: Rename.scala allocates one physical register per distinct architectural register a
// warp ever writes, out of a file of 256 shared by every warp, and never frees it.
//
// Global memory only, and MS_WS must be at least 2048 so the 32 offsets stay inside the region.
#ifndef MS_ASM_REGS
#define MS_ASM_REGS 16
#endif
#if MS_ASM == 1 && MS_ASM_REGS == 16
#define MS_ASM_BLOCK(p, acc, v)                                                   \
  asm volatile(                                                                   \
      "lw %0, 0(%17)\n\t" \
      "lw %1, 64(%17)\n\t" \
      "lw %2, 128(%17)\n\t" \
      "lw %3, 192(%17)\n\t" \
      "lw %4, 256(%17)\n\t" \
      "lw %5, 320(%17)\n\t" \
      "lw %6, 384(%17)\n\t" \
      "lw %7, 448(%17)\n\t" \
      "lw %8, 512(%17)\n\t" \
      "lw %9, 576(%17)\n\t" \
      "lw %10, 640(%17)\n\t" \
      "lw %11, 704(%17)\n\t" \
      "lw %12, 768(%17)\n\t" \
      "lw %13, 832(%17)\n\t" \
      "lw %14, 896(%17)\n\t" \
      "lw %15, 960(%17)\n\t" \
      "xor %16, %16, %0\n\t" \
      "xor %16, %16, %1\n\t" \
      "xor %16, %16, %2\n\t" \
      "xor %16, %16, %3\n\t" \
      "xor %16, %16, %4\n\t" \
      "xor %16, %16, %5\n\t" \
      "xor %16, %16, %6\n\t" \
      "xor %16, %16, %7\n\t" \
      "xor %16, %16, %8\n\t" \
      "xor %16, %16, %9\n\t" \
      "xor %16, %16, %10\n\t" \
      "xor %16, %16, %11\n\t" \
      "xor %16, %16, %12\n\t" \
      "xor %16, %16, %13\n\t" \
      "xor %16, %16, %14\n\t" \
      "xor %16, %16, %15\n\t" \
      "lw %0, 1024(%17)\n\t" \
      "lw %1, 1088(%17)\n\t" \
      "lw %2, 1152(%17)\n\t" \
      "lw %3, 1216(%17)\n\t" \
      "lw %4, 1280(%17)\n\t" \
      "lw %5, 1344(%17)\n\t" \
      "lw %6, 1408(%17)\n\t" \
      "lw %7, 1472(%17)\n\t" \
      "lw %8, 1536(%17)\n\t" \
      "lw %9, 1600(%17)\n\t" \
      "lw %10, 1664(%17)\n\t" \
      "lw %11, 1728(%17)\n\t" \
      "lw %12, 1792(%17)\n\t" \
      "lw %13, 1856(%17)\n\t" \
      "lw %14, 1920(%17)\n\t" \
      "lw %15, 1984(%17)\n\t" \
      "xor %16, %16, %0\n\t" \
      "xor %16, %16, %1\n\t" \
      "xor %16, %16, %2\n\t" \
      "xor %16, %16, %3\n\t" \
      "xor %16, %16, %4\n\t" \
      "xor %16, %16, %5\n\t" \
      "xor %16, %16, %6\n\t" \
      "xor %16, %16, %7\n\t" \
      "xor %16, %16, %8\n\t" \
      "xor %16, %16, %9\n\t" \
      "xor %16, %16, %10\n\t" \
      "xor %16, %16, %11\n\t" \
      "xor %16, %16, %12\n\t" \
      "xor %16, %16, %13\n\t" \
      "xor %16, %16, %14\n\t" \
      "xor %16, %16, %15\n\t" \
    : "=&r"(v[0]), \
      "=&r"(v[1]), \
      "=&r"(v[2]), \
      "=&r"(v[3]), \
      "=&r"(v[4]), \
      "=&r"(v[5]), \
      "=&r"(v[6]), \
      "=&r"(v[7]), \
      "=&r"(v[8]), \
      "=&r"(v[9]), \
      "=&r"(v[10]), \
      "=&r"(v[11]), \
      "=&r"(v[12]), \
      "=&r"(v[13]), \
      "=&r"(v[14]), \
      "=&r"(v[15]), "+r"(acc)                                                           \
    : "r"(p) : "memory")
#endif
#if MS_ASM == 2 && MS_ASM_REGS == 16
#define MS_ASM_BLOCK(p, acc, v)                                                   \
  asm volatile(                                                                   \
      "lw %0, 0(%17)\n\t" \
      "lw %1, 64(%17)\n\t" \
      "lw %2, 128(%17)\n\t" \
      "lw %3, 192(%17)\n\t" \
      "lw %4, 256(%17)\n\t" \
      "lw %5, 320(%17)\n\t" \
      "lw %6, 384(%17)\n\t" \
      "lw %7, 448(%17)\n\t" \
      "lw %8, 512(%17)\n\t" \
      "lw %9, 576(%17)\n\t" \
      "lw %10, 640(%17)\n\t" \
      "lw %11, 704(%17)\n\t" \
      "lw %12, 768(%17)\n\t" \
      "lw %13, 832(%17)\n\t" \
      "lw %14, 896(%17)\n\t" \
      "lw %15, 960(%17)\n\t" \
      "lw %0, 1024(%17)\n\t" \
      "lw %1, 1088(%17)\n\t" \
      "lw %2, 1152(%17)\n\t" \
      "lw %3, 1216(%17)\n\t" \
      "lw %4, 1280(%17)\n\t" \
      "lw %5, 1344(%17)\n\t" \
      "lw %6, 1408(%17)\n\t" \
      "lw %7, 1472(%17)\n\t" \
      "lw %8, 1536(%17)\n\t" \
      "lw %9, 1600(%17)\n\t" \
      "lw %10, 1664(%17)\n\t" \
      "lw %11, 1728(%17)\n\t" \
      "lw %12, 1792(%17)\n\t" \
      "lw %13, 1856(%17)\n\t" \
      "lw %14, 1920(%17)\n\t" \
      "lw %15, 1984(%17)\n\t" \
    : "=&r"(v[0]), \
      "=&r"(v[1]), \
      "=&r"(v[2]), \
      "=&r"(v[3]), \
      "=&r"(v[4]), \
      "=&r"(v[5]), \
      "=&r"(v[6]), \
      "=&r"(v[7]), \
      "=&r"(v[8]), \
      "=&r"(v[9]), \
      "=&r"(v[10]), \
      "=&r"(v[11]), \
      "=&r"(v[12]), \
      "=&r"(v[13]), \
      "=&r"(v[14]), \
      "=&r"(v[15]), "+r"(acc)                                                           \
    : "r"(p) : "memory")
#endif
#if MS_ASM == 1 && MS_ASM_REGS == 8
#define MS_ASM_BLOCK(p, acc, v)                                                   \
  asm volatile(                                                                   \
      "lw %0, 0(%9)\n\t" \
      "lw %1, 64(%9)\n\t" \
      "lw %2, 128(%9)\n\t" \
      "lw %3, 192(%9)\n\t" \
      "lw %4, 256(%9)\n\t" \
      "lw %5, 320(%9)\n\t" \
      "lw %6, 384(%9)\n\t" \
      "lw %7, 448(%9)\n\t" \
      "xor %8, %8, %0\n\t" \
      "xor %8, %8, %1\n\t" \
      "xor %8, %8, %2\n\t" \
      "xor %8, %8, %3\n\t" \
      "xor %8, %8, %4\n\t" \
      "xor %8, %8, %5\n\t" \
      "xor %8, %8, %6\n\t" \
      "xor %8, %8, %7\n\t" \
      "lw %0, 512(%9)\n\t" \
      "lw %1, 576(%9)\n\t" \
      "lw %2, 640(%9)\n\t" \
      "lw %3, 704(%9)\n\t" \
      "lw %4, 768(%9)\n\t" \
      "lw %5, 832(%9)\n\t" \
      "lw %6, 896(%9)\n\t" \
      "lw %7, 960(%9)\n\t" \
      "xor %8, %8, %0\n\t" \
      "xor %8, %8, %1\n\t" \
      "xor %8, %8, %2\n\t" \
      "xor %8, %8, %3\n\t" \
      "xor %8, %8, %4\n\t" \
      "xor %8, %8, %5\n\t" \
      "xor %8, %8, %6\n\t" \
      "xor %8, %8, %7\n\t" \
      "lw %0, 1024(%9)\n\t" \
      "lw %1, 1088(%9)\n\t" \
      "lw %2, 1152(%9)\n\t" \
      "lw %3, 1216(%9)\n\t" \
      "lw %4, 1280(%9)\n\t" \
      "lw %5, 1344(%9)\n\t" \
      "lw %6, 1408(%9)\n\t" \
      "lw %7, 1472(%9)\n\t" \
      "xor %8, %8, %0\n\t" \
      "xor %8, %8, %1\n\t" \
      "xor %8, %8, %2\n\t" \
      "xor %8, %8, %3\n\t" \
      "xor %8, %8, %4\n\t" \
      "xor %8, %8, %5\n\t" \
      "xor %8, %8, %6\n\t" \
      "xor %8, %8, %7\n\t" \
      "lw %0, 1536(%9)\n\t" \
      "lw %1, 1600(%9)\n\t" \
      "lw %2, 1664(%9)\n\t" \
      "lw %3, 1728(%9)\n\t" \
      "lw %4, 1792(%9)\n\t" \
      "lw %5, 1856(%9)\n\t" \
      "lw %6, 1920(%9)\n\t" \
      "lw %7, 1984(%9)\n\t" \
      "xor %8, %8, %0\n\t" \
      "xor %8, %8, %1\n\t" \
      "xor %8, %8, %2\n\t" \
      "xor %8, %8, %3\n\t" \
      "xor %8, %8, %4\n\t" \
      "xor %8, %8, %5\n\t" \
      "xor %8, %8, %6\n\t" \
      "xor %8, %8, %7\n\t" \
    : "=&r"(v[0]), \
      "=&r"(v[1]), \
      "=&r"(v[2]), \
      "=&r"(v[3]), \
      "=&r"(v[4]), \
      "=&r"(v[5]), \
      "=&r"(v[6]), \
      "=&r"(v[7]), "+r"(acc)                                                           \
    : "r"(p) : "memory")
#endif
#if MS_ASM == 2 && MS_ASM_REGS == 8
#define MS_ASM_BLOCK(p, acc, v)                                                   \
  asm volatile(                                                                   \
      "lw %0, 0(%9)\n\t" \
      "lw %1, 64(%9)\n\t" \
      "lw %2, 128(%9)\n\t" \
      "lw %3, 192(%9)\n\t" \
      "lw %4, 256(%9)\n\t" \
      "lw %5, 320(%9)\n\t" \
      "lw %6, 384(%9)\n\t" \
      "lw %7, 448(%9)\n\t" \
      "lw %0, 512(%9)\n\t" \
      "lw %1, 576(%9)\n\t" \
      "lw %2, 640(%9)\n\t" \
      "lw %3, 704(%9)\n\t" \
      "lw %4, 768(%9)\n\t" \
      "lw %5, 832(%9)\n\t" \
      "lw %6, 896(%9)\n\t" \
      "lw %7, 960(%9)\n\t" \
      "lw %0, 1024(%9)\n\t" \
      "lw %1, 1088(%9)\n\t" \
      "lw %2, 1152(%9)\n\t" \
      "lw %3, 1216(%9)\n\t" \
      "lw %4, 1280(%9)\n\t" \
      "lw %5, 1344(%9)\n\t" \
      "lw %6, 1408(%9)\n\t" \
      "lw %7, 1472(%9)\n\t" \
      "lw %0, 1536(%9)\n\t" \
      "lw %1, 1600(%9)\n\t" \
      "lw %2, 1664(%9)\n\t" \
      "lw %3, 1728(%9)\n\t" \
      "lw %4, 1792(%9)\n\t" \
      "lw %5, 1856(%9)\n\t" \
      "lw %6, 1920(%9)\n\t" \
      "lw %7, 1984(%9)\n\t" \
    : "=&r"(v[0]), \
      "=&r"(v[1]), \
      "=&r"(v[2]), \
      "=&r"(v[3]), \
      "=&r"(v[4]), \
      "=&r"(v[5]), \
      "=&r"(v[6]), \
      "=&r"(v[7]), "+r"(acc)                                                           \
    : "r"(p) : "memory")
#endif

// Push the records out to DRAM so the rv64 host can read them back, and mark the core done.
// The flush is issued from a single warp because a multi-warp `fence` trips an assertion on the
// taped-out RTL; it is repeated with a delay between, because that RTL starts the sweep before the
// stores have drained and needs more than one pass (the reason for ledger row 4).
//
// The host only sees any of this when the run keeps going after the GPU retires, i.e. with
// +gpu_finish_keeps_sim=1, which exists only on this branch.  On the taped-out RTL the harness stops
// the simulation 1k cycles after the last warp dies, and the only number that run yields is its own
// end time, which is the whole kernel's cycle count.
static inline void ms_spin(uint32_t n) {
  while (n--) asm volatile("" ::: "memory");
}
// With MS_L1_FLUSH the L1 flush follows each L0d flush: the L1 holds new requests while its sweep runs,
// so the records leave the L1 before the DONE word reaches it, and the second sweep writes DONE.
static inline void ms_l1_flush() {
#if MS_L1_FLUSH
  // a cluster register: the shared-memory store path reaches the cluster bus, a plain store goes to GMEM
MS_PRAGMA_UNROLL_FULL
  for (uint32_t b = 0; b < MS_L1_BANKS; b++)
    asm volatile("sw.shared %0, 0(%1)" :: "r"(1u), "r"(MS_L1_FLUSH_REG + 0x100u * b) : "memory");
  ms_spin(500);
#endif
}
static inline void ms_publish(uint32_t g) {
  mu_fence();
  ms_spin(500);
  mu_fence();
  ms_l1_flush();
  *(volatile uint32_t *)(MS_DONE_BASE + g * 4u) = MS_DONE_MAGIC | g;
  mu_fence();
  ms_spin(500);
  mu_fence();
  ms_l1_flush();
}

static void entry(void *, uint32_t, uint32_t, uint32_t) {
  uint32_t cluster;
  asm volatile("csrr %0, %1" : "=r"(cluster) : "i"(MU_CSR_CLUSTER_ID));
  const uint32_t core = vx_core_id();
  const uint32_t g = cluster * 2u + core;
  const uint32_t w = vx_warp_id();
  const uint32_t lane = vx_thread_id();
  if (((MS_CORES >> g) & 1u) == 0u || w >= MS_NWARPS) return;

#if MS_SPACE == 1
  const uint32_t base = MS_SHARED_REGION ? 0u : core * 0x10000u;   // one bank pair per core
#else
  const uint32_t base = MS_GMEM_BASE + (MS_SHARED_REGION ? 0u : g * MS_WS) +
                        (g / MS_REGION_GROUP) * MS_REGION_STRIDE;
#endif
  const uint32_t bar0 = 1u + core * 4u;   // core-local barrier ids (the count is per id, cluster-wide)

  mu_barrier(bar0, MS_NWARPS);
  const uint32_t t0 = rd_mcycle();
  uint32_t last = 0;
#if MS_STORE
  for (uint32_t i = 0; i < MS_ITERS; i += MS_UNROLL) {
MS_PRAGMA_UNROLL_FULL
    for (uint32_t j = 0; j < MS_UNROLL; j++) {
      const uint32_t addr = base + offset_of(MS_IDX(i, j, w), lane);
      mem_store(addr, value_of(addr));
      last = addr;
    }
  }
#endif
  const uint32_t t1 = rd_mcycle_after(last);
#if MS_STORE && MS_FENCE == 1
  mu_fence();
#elif MS_STORE && MS_FENCE == 2
  if (w == 0) { mu_fence(); } else { asm volatile("nop"); }
#endif
  const uint32_t t1f = rd_mcycle();

  mu_barrier(bar0 + 1u, MS_NWARPS);
  const uint32_t t2 = rd_mcycle();
  // Issue MS_UNROLL loads before consuming any of them.  Checking each value where it lands puts a
  // dependent instruction between every pair of loads, and on an in-order core that measures
  // load-to-use latency instead of throughput: with the check inline, unrolling changed nothing and
  // the core held exactly one load in flight whatever the unroll factor.
  uint32_t sum = 0, err = 0;

#if MS_SUBWORD == 2
  // Sensitivity check for the test below: reproduce the OLD, broken mapping by hand and confirm the
  // lanes really do collide, so that err == 0 in MS_SUBWORD=1 means something.
  //
  // It has to be inline asm with literal offsets.  Writing it in C does not work: the compiler
  // folds the pointer arithmetic back into frame-index references and then applies the (now
  // correct) word-granular scaling, so no aliasing is ever created and the control silently passes.
  //
  // Lane L's frame sits 4 bytes above lane L-1's, so putting byte i of a word at +16*i -- which is
  // what scaling the whole offset used to do -- makes lane L's byte i land on lane L+4's byte i-1.
  // Expect a NONZERO error count here.
  {
    volatile unsigned int pad[64];
    unsigned char *base = (unsigned char *)&pad[0];
    const uint32_t t0v = 0x40u + lane * 4u;
    uint32_t r0, r1, r2, r3;
    asm volatile(
      "sb.global %4, 0(%8)\n\t"
      "sb.global %5, 16(%8)\n\t"
      "sb.global %6, 32(%8)\n\t"
      "sb.global %7, 48(%8)\n\t"
      "lbu.global %0, 0(%8)\n\t"
      "lbu.global %1, 16(%8)\n\t"
      "lbu.global %2, 32(%8)\n\t"
      "lbu.global %3, 48(%8)\n\t"
      : "=&r"(r0), "=&r"(r1), "=&r"(r2), "=&r"(r3)
      : "r"(t0v), "r"(t0v + 1u), "r"(t0v + 2u), "r"(t0v + 3u), "r"(base)
      : "memory");
    if (r0 != (t0v & 0xffu)) err += 1u;
    if (r1 != ((t0v + 1u) & 0xffu)) err += 1u;
    if (r2 != ((t0v + 2u) & 0xffu)) err += 1u;
    if (r3 != ((t0v + 3u) & 0xffu)) err += 1u;
    sum ^= r0 ^ r1 ^ r2 ^ r3;
  }
#endif
#if MS_SUBWORD == 1
  // Sub-word stack traffic from every lane at once.  Under the lane-interleaved stack a byte offset
  // inside a word must NOT be scaled by the stride, or lane L's byte lands in lane L+4's slot and
  // the lanes silently overwrite each other.  Each lane writes a value only it can produce, reads
  // the bytes back, and checks them; any cross-lane leak shows up as a nonzero err.
  {
    volatile unsigned char b[4];
    volatile unsigned short h[2];
    const unsigned char tag = (unsigned char)(0x40u + lane);
    for (uint32_t rep = 0; rep < 64u; rep++) {
      b[0] = tag; b[1] = (unsigned char)(tag + 1u);
      b[2] = (unsigned char)(tag + 2u); b[3] = (unsigned char)(tag + 3u);
      h[0] = (unsigned short)(0x1000u + lane); h[1] = (unsigned short)(0x2000u + lane);
      if (b[0] != tag || b[1] != (unsigned char)(tag + 1u) ||
          b[2] != (unsigned char)(tag + 2u) || b[3] != (unsigned char)(tag + 3u) ||
          h[0] != (unsigned short)(0x1000u + lane) ||
          h[1] != (unsigned short)(0x2000u + lane)) {
        err += 1u;
      }
      sum ^= (uint32_t)b[0] ^ ((uint32_t)h[1] << 8);
    }
    // The case above lets the compiler split the array into separate scalars, one word slot each.
    // Index it with a value it cannot fold so the array stays one frame object and the bytes are
    // reached by immediate: that is the sgemm_gemmini_dma shape, sb at 0x20/0x21/0x22(sp), where a
    // scaled byte offset used to land in lane 4's and lane 8's slots.
    volatile unsigned char blob[8];
    const uint32_t k = lane & 3u;
    for (uint32_t rep = 0; rep < 64u; rep++) {
      for (uint32_t i = 0; i < 8u; i++) blob[i] = (unsigned char)(0x80u + lane + i);
      for (uint32_t i = 0; i < 8u; i++)
        if (blob[i] != (unsigned char)(0x80u + lane + i)) err += 1u;
      sum ^= (uint32_t)blob[k];
    }
  }
#endif

#if MS_ASM >= 1
  // Every warp walks the same 32 lines, which is the point: the working set is resident, so this
  // measures how fast the core can drive a hitting L0d and nothing else.
#if MS_STREAM
  uint32_t p = base + w * (MS_ITERS * 64u) + (g * MS_NWARPS + w) * MS_SKEW + lane * 4u;
#else
  const uint32_t p = base + lane * 4u;
#endif
  uint32_t vv[MS_ASM_REGS];
  // Do NOT let the compiler unroll this.  The block is already 32 loads; unrolling the loop around
  // it turns the kernel into straight-line code with no instruction reuse, and since a 32 B L0i line
  // holds only four 8-byte Muon instructions, every fourth fetch becomes a compulsory miss.  Left to
  // itself clang flattened 24 iterations into 7.9 KB of loads (782 lw.global, 247 L0i lines walked
  // once), and fetch, not the data path, set the measured throughput.
#pragma clang loop unroll(disable)
  for (uint32_t i = 0; i < MS_ITERS; i += 32u) {
    MS_ASM_BLOCK(p, sum, vv);
#if MS_STREAM
    p += 32u * 64u;
#endif
  }
#if MS_ASM == 1
  // The block reads the same 32 addresses every time and value_of(addr) is the address, so the
  // expected checksum is one block's worth xored MS_ITERS/32 times, i.e. 0 for an even count.
  uint32_t expect = 0;
  for (uint32_t k = 0; k < 32u; k++) expect ^= (p + k * 64u);
  if ((MS_ITERS / 32u) % 2u == 0u) expect = 0u;
  err = (sum != expect) ? 1u : 0u;
#else
  err = 0u;   // nothing consumes the loaded values in this mode; it verifies nothing
#endif
#else
  for (uint32_t i = 0; i < MS_ITERS; i += MS_UNROLL) {
    uint32_t v[MS_UNROLL];
    uint32_t a[MS_UNROLL];
MS_PRAGMA_UNROLL_FULL
    for (uint32_t j = 0; j < MS_UNROLL; j++) {
      a[j] = base + offset_of(MS_IDX(i, j, w), lane);
      v[j] = mem_load(a[j]);
    }
MS_PRAGMA_UNROLL_FULL
    for (uint32_t j = 0; j < MS_UNROLL; j++) {
      sum ^= v[j];
      err += (v[j] != value_of(a[j])) ? 1u : 0u;
    }
  }
#endif
  const uint32_t t3 = rd_mcycle_after(sum);

  // One error word per warp, not per lane: the host pays about 1300 cycles for every GPU-DRAM word
  // it reads back, so 16 words per warp dominated the run.  Lane 0 clears the word, then any lane
  // with a mismatch overwrites it; lanes of a warp execute in lockstep, so the clear is ordered
  // before the overwrites, and any nonzero value means at least one lane saw wrong data.
  volatile uint32_t *errs = (volatile uint32_t *)(MS_ERR_BASE + (g * 8u + w) * 4u);
  if (lane == 0) { *errs = 0; } else { asm volatile("nop"); }
  if (err) { *errs = err; } else { asm volatile("nop"); }
  if (lane == 0) {
    volatile uint32_t *rec = (volatile uint32_t *)(MS_REC_BASE + (g * 8u + w) * 64u);
    rec[1] = t0; rec[2] = t1; rec[3] = t1f; rec[4] = t2; rec[5] = t3; rec[6] = sum; rec[7] = 0;
    rec[0] = 0xC0DE0000u | (g << 4) | w;
  } else {
    asm volatile("nop");
  }

#if MS_PUBLISH
  mu_barrier(bar0 + 2u, MS_NWARPS);
  if (w == 0) {
    ms_publish(g);
  } else {
    asm volatile("nop");
  }
#endif
}

int main() {
  mu_schedule(entry, nullptr, MS_NWARPS);
  return 0;
}
