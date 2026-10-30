# rpbench: the llama_e2e ROPE phase on cyclotron

A device-only kernel that runs the unfused ROPE phase (`../rope_fast.h`: `rope_qk_fast` + `rope_v`) on synthetic
prefill inputs under cyclotron, against a reference, for one case per build (`gen_data.py`; M = 128 tokens at P0 = 128
of an SK = 256 cache, ATTN layout):

| case | shape | reference (OLD) |
|---|---|---|
| gemma2 | hd 256, 8 / 4 heads | `rope_fast.h` + `rope_v` as committed (warp per unit, gw of ngw) |
| tinyllama | hd 64, 32 / 4 heads | the same |
| deepseek | hd 128, 12 / 2 heads, q/k/v bias, key center | the fused `rope_q/k/v_unit<true>` of `../kernel.cpp`, every unit in turn |
| llama7b | hd 128, 32 / 32 heads | `rope_fa` + `rope_v_fa` |

kernel.cpp's helpers and ROPE code are extracted from `../kernel.cpp` at build time (`build/helpers.h`), the committed
`rope_fast.h` / `rope_v` from `git show HEAD:` (`build/rope_old.h`), so the bench tests what the kernels compile.

Clusters are simulated on cyclotron's one: `SIM_NC` clusters, `SIM_TB` the one this build plays (`-1`: all in turn).
A phase takes its slowest cluster's time.

```
make PY=<python with numpy> CASE=deepseek build/1sm/rpbench.radiance.elf DEFS="-DBENCH_MODE=3 -DSIM_NC=4"
cyclotron bench.toml --binary-path build/1sm/rpbench.radiance.elf      # OLD vs NEW, every output word
make PY=... sweep     # every case: build/rp_<case>_m<mode>_nc<nc>_tb<tb>.elf (mode 3 check, modes 0 / 1 / 2 timing)
cyclotron bench.toml --timing --binary-path build/rp_gemma2_m2_nc4_tb0.elf
```
`BENCH_MODE` 0 empty, 1 OLD, 2 NEW, 3 OLD + NEW + compare.  Time a build by the simulator's cycle count minus mode 0's
(mcycle does not advance).

## Results (2026-10-29, --timing; the slowest cluster, M cycles)

All four cases: Q, K, V codes and scales identical to the reference (mode 3, 4 clusters).

| case | 1 SM old / new | 2 SM old / new | 4 SM old / new |
|---|---|---|---|
| gemma2 | 3.83 / 2.49 | 2.89 / 1.35 | 2.89 / 0.74 (3.9x) |
| tinyllama | 1.34 / 1.44 | 0.77 / 0.76 | 0.55 / 0.42 |
| deepseek (vs fused units run as a phase) | 1.88 / hang* | 1.00 / hang* | 0.60 / 0.42 |
| llama7b (vs rope_fa) | 6.82 / 7.14 | 3.49 / 3.65 | 1.78 / 1.88 |

The old Gemma-2 phase had 16 Q, 8 K and 16 V units, all dealt from warp 0: beyond 16 warps the extra clusters idled
(FireSim: 4.98 / 3.78 / 3.68 M per layer at 1 / 2 / 4 SM).  Its K units also recomputed every key's row once per 64-d
pass.  Calibration: cyclotron's old Gemma-2 numbers are ~0.77x FireSim's.  Llama-7B keeps `rope_fa` (its Q units
share a token's cos / sin over 16 heads).

*cyclotron's timing model deadlocks on these two builds (DeepSeek NEW, 1 cluster and cluster 1 of 2): after ~1.15 M
cycles every gmem request is rejected as queue-full with nothing in flight, until the 400 M timeout.  Functionally
they finish (and the 4-cluster check is bit-exact); FireSim times them.
