# gemm_mxgemmini_ws_restream

The **naive M-outer re-stream** baseline for the weight-reuse study: it computes the
same `C[256, 64]` as four `TM=64` M-blocks, each running its own full K-loop and
re-DMAing the weight `B` from DRAM.

Effective weight-tile move-ins = 4 blocks x 32 k-tiles = 128 (the 128 KB weight read
**4x**). The read-once counterpart that DMAs each weight byte exactly once is
[`gemm_mxgemmini_ws`](../gemm_mxgemmini_ws).
