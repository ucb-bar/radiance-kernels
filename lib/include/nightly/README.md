# Nightly RTL kernel headers

These headers support Muon + MxGemmini kernels on the "nightly" Radiance RTL. The kernels are in
`kernels/nightly`. Do not use them on the taped-out part.

## RTL

The nightly RTL is:

* Radiance: branch `split-l2` (radiance `main` plus the split host/GPU L2, one DRAM channel per L2
  slice, and the configs `RadianceHBMConfig` (2 SMs) and `RadianceSingleSMHBMConfig` (1 SM)).
  The kernels need these fixes, which are on `main`:
  * Muon FPPipe: the shared CVFPU returns packets of different operation groups out of order.
    The fix tags the fp32-to-bf16 convert and allows one operation group in flight per pipe.
  * Muon SFUPipe: `fence.s` waits for the shared-memory queue of its own warp only.
  * CollectorNode: one request fires when all 16 lanes are valid (no per-source state).
  * GemminiTile requantizer input: a beat is valid only when its request fires.
* Gemmini: branch `gemmini-mx-cleanup` with these fixes:
  * Scratchpad: back-pressure shared-memory reads on room in the DMA queue.
  * ExecuteController: pop each operand read response when the mesh accepts it.
  * LoopMatmulStCSpad: the store's row step comes from its own loop, not the global bounds.
  * StoreController: a DRAM store may carry its own J bound in rs1[63:56] (MXFP4 C stores).
  * LoopMatmul: do not force the A/B load loops when they are idle.

## Headers

| Header | Contents |
|---|---|
| `pb.h` | printBuf mailbox, shared by the host and the GPU |
| `device.h` | GPU launch, per-core start and end timestamps, barriers |
| `host.h` | Host launch, result wait, integer helpers (the host Rocket has no FPU) |
| `mx.h` | MxGemmini commands: loads, loops, scale memory, requantizer, stores |
| `verify.h` | GPU-side comparison of a result tensor with a golden |

## Build and run

1. Use the muon LLVM toolchain with the stack word stride (`llvm/llvm-muon`).
2. In a kernel directory, run `make`. The build writes `build/1sm/<kernel>.soc.elf` and
   `build/2sm/<kernel>.soc.elf`.
3. Run the 1sm ELF on `RadianceSingleSMHBMConfig` and the 2sm ELF on `RadianceHBMConfig`. The
   host prints the mesh utilization and the result check.
