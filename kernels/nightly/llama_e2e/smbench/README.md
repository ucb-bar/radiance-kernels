# smbench: the llama_e2e SOFTMAX phases on cyclotron

A device-only kernel that runs the SOFTMAX phases of `../kernel_vla.cpp` (SmolVLA) and `../kernel.cpp` (LLM decode)
on synthetic scores under [cyclotron](../../../../../chipyard/generators/radiance/cyclotron), no FireSim needed.
The NEW code is extracted from the two sources at build time (`build/softmax_vla.h`, `build/softmax_llm.h`: the
`// ---- SOFTMAX:` section up to `// ---- EWQ / SILU`), so the bench tests what the kernels compile.  OLD is the
previous SmolVLA phase (a copy in `kernel.cpp`, a warp per output group) and `kernel.cpp`'s `softmax_phase<false>`.

| case | shapes | as in |
|---|---|---|
| self | SK 512, 2 x 192 rows, mask table (prefix 0..204, suffix 256..256+t) | SmolVLA expert self-attention |
| cross | SK 256, 2 x 192 rows, prefix 0..204 | SmolVLA expert cross-attention, prefix |
| decode | SK 1024, 16 x 16 rows (8 valid), query position 1000 | TinyLlama decode (gen_decode.py) |

```
make bench                      # build/smbench_mode{0,1,2,3}.elf (BENCH_CASES=7: all three cases)
cyclotron bench.toml --binary-path build/smbench_mode3.elf            # OLD vs NEW: every code / scale byte
cyclotron bench.toml --timing --binary-path build/smbench_mode2.elf   # NEW, cycles
```
`BENCH_MODE` 0 empty, 1 OLD, 2 NEW, 3 OLD + NEW + compare; `DEFS="-DBENCH_MODE=2 -DBENCH_CASES=4"` times one case.
Cyclotron's `mcycle` does not advance: time a build by the simulator's cycle count.  `bench.toml` is cyclotron's
`config.toml` with absolute timing includes, `num_warps = 4` (OCC 4) and a larger timeout.

Cyclotron limits found on the way: barrier ids >= 8 are not barriers (they are other Neutrino tasks: a kernel waits
forever), and `__mu_cluster_from_pb = 1` (nightly/device.h) makes mu_start.S wait for a host that is not there, so
the bench defines the runtime symbols itself.

## Results (2026-10-29, cyclotron 9b774a5, --timing)

| case | OLD cycles | NEW cycles | speed-up | bytes |
|---|---|---|---|---|
| self | 2,507,911 | 984,855 | 2.55x | identical |
| cross | 1,321,025 | 602,264 | 2.19x | identical |
| decode | 612,676 | 566,994 | 1.08x | identical |

Calibration: OLD costs ~4,840 cycles per SmolVLA row here against ~4,230 on FireSim (SmolVLA prefill esoftmax,
1 SM), and ~2.45 M per 512 decode rows against 2.94 M (TinyLlama decode softmax).  Decode stays SMEM-bound in the
model: the new phase issues fewer SMEM instructions (14.5 K vs 22.7 K) but each splits into ~2x more bank requests
(cyclotron models 4 banks; quant_group's lane rotation spreads over 16).
