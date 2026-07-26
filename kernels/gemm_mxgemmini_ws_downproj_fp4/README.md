# gemm_mxgemmini_ws_downproj_fp4

The fp4 **down-projection** in read-once weight-stationary form: one
`TM=256 x TN=64` tile with each weight byte DMA'd from DRAM exactly once.

The fp4 (e2m1) analogue of [`gemm_mxgemmini_ws`](../gemm_mxgemmini_ws), sized for the
MLP down-proj.
