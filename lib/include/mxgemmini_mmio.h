/** mxgemmini_mmio.h
 *  ================
 *
 *  Macro definitions that implement MMIO interface between Muon - MxGemmini.
 *  Include it *after* gemmini.h so that it correctly overrides APIs such as
 *  ROCC_INSTRUCTION_RS1_RS2.
 *
 *  Note that all of these addresses are within the GPU-local address space,
 *  different from the CPU-global space.
 *
 *  @cleanup: These only differ marginally from gemmini_mmio.h.  Merging the
 *  two would be ideal.
 */

#define GEMMINI_SF_MEM 0x00088000
#define GEMMINI_SF_MEM_A (GEMMINI_SF_MEM + 0x2000)
#define GEMMINI_SF_MEM_B GEMMINI_SF_MEM

#define GEMMINI_CTRL 0x00084000
#define GEMMINI_RS1_ADDR  (GEMMINI_CTRL + 0x10)
#define GEMMINI_RS2_ADDR  (GEMMINI_CTRL + 0x18)
#define GEMMINI_INST_ADDR (GEMMINI_CTRL + 0x0)

#undef ROCC_INSTRUCTION_RS1_RS2
#define ROCC_INSTRUCTION_RS1_RS2(x, rs1, rs2, funct) { \
    *((volatile uint64_t *) GEMMINI_RS1_ADDR) = (rs1); \
    *((volatile uint64_t *) GEMMINI_RS2_ADDR) = (rs2); \
    *((volatile uint32_t*) GEMMINI_INST_ADDR) = (0x7B) | (0 << 7) | (3 << 12) | (1 << 15) | (2 << 20) | ((funct) << 25); \
}
