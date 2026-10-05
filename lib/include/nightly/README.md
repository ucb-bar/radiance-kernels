# Nightly RTL kernel headers

These headers support Muon + MxGemmini kernels on the "nightly" Radiance RTL. The kernels are in
`kernels/nightly`. Do not use them on the taped-out part.

## RTL

The nightly RTL is:

* Radiance: the split-l2 branch merged with origin/main (311895c), plus these fixes:
  * Muon FPPipe: the shared CVFPU returns packets of different operation groups out of order.
    The fix tags the fp32-to-bf16 convert and allows one operation group in flight per pipe.
  * Muon SFUPipe: `fence.s` waits for the shared-memory queue of its own warp only.
  * CollectorNode: one request fires when all 16 lanes are valid (no per-source state).
  * GemminiTile requantizer input: a beat is valid only when the response channel is ready.
  * `RadianceSingleSMHBMConfig` (1 SM). `RadianceHBMConfig` is the 2-SM configuration.
* Gemmini: branch `gemmini-mx-cleanup` at 0901baa, plus these fixes:
  * Scratchpad: back-pressure on the DMA queue.
  * ExecuteController: pop each operand read response when the mesh accepts it.
  * StoreController and LoopMatmulStCSpad: store strides and the J bound of the store.
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
