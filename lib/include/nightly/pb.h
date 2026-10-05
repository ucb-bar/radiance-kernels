/*
 * nightly/pb.h: printBuf mailbox shared by the Rocket host and the Muon kernels (nightly RTL).
 *
 * printBuf is a small uncached SRAM in every cluster: cluster-local 0x80000 on the GPU
 * (lw.shared / sw.shared), 0x4008_0000 + 0x10_0000 * cluster on the host.  Neither side caches
 * it, so it is the one channel both sides see immediately.  Slots are 8 bytes; the GPU writes
 * each 32-bit value to both halves so a 64-bit host read sees it whichever half it takes.
 *
 * Slot map (per cluster):
 *   0      LAUNCH    host -> GPU: NIGHTLY_LAUNCH_MAGIC starts the kernel
 *   1      ARG       host -> GPU: optional 32-bit argument (device address of an argument block)
 *   2, 3   START     GPU -> host: mcycle at kernel start, per core
 *   4, 5   END       GPU -> host: mcycle after the kernel's final barrier, per core
 *   6      STATUS    GPU -> host: kernel-defined status word (0 = none)
 *   8..15  PHASE     GPU -> host: kernel-defined phase stamps (mcycle), cluster core 0
 *   16..63 free
 */
#ifndef NIGHTLY_PB_H
#define NIGHTLY_PB_H

#include <stdint.h>

#define NIGHTLY_PB_DEV_BASE      0x80000u
#define NIGHTLY_PB_HOST_BASE(cl) (0x40080000ull + 0x100000ull * (uint64_t)(cl))

#define NIGHTLY_PB_LAUNCH  0u
#define NIGHTLY_PB_ARG     1u
#define NIGHTLY_PB_START   2u   /* + core */
#define NIGHTLY_PB_END     4u   /* + core */
#define NIGHTLY_PB_STATUS  6u
#define NIGHTLY_PB_PHASE   8u   /* + phase index, 0..7 */
#define NIGHTLY_PB_HEARTBEAT 48u /* kernel progress value, printed by the host while it waits */

#define NIGHTLY_LAUNCH_MAGIC 0x6E1A0C11u

#endif
