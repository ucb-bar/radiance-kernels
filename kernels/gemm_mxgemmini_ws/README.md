# gemm_mxgemmini_ws

**Proper weight-stationary** MX GEMM: one `TM=256 x TN=64` tile.

The mesh-loop FSM loads each `B[k]` tile (`64x64` fp8 = 4 KB) into SMEM **once** per
k-tile (32 k-tiles) and streams all 256 M-rows through it, so every weight byte is
DMA'd from DRAM **exactly once** (32 weight-tile move-ins = the 128 KB weight, 1x).

This is the read-once "after" for the weight-reuse study; the re-streaming baseline
it is measured against is [`gemm_mxgemmini_ws_restream`](../gemm_mxgemmini_ws_restream).
