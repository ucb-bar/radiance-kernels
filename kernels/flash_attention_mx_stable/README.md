# flash_attention_mx_stable

**Stability-oriented** MX-FP8 flash attention. Same algorithm and same sources as
[`../flash_attention_mx`](../flash_attention_mx/README.md), different objective: this directory
optimizes for **staying bit-correct under schedule perturbation** and accepts whatever utilization
that costs. The peak-utilization variant lives in the sibling directory and is explicitly *not*
robust.

```
Q,K,V (mvin) --[mesh: QK^T]--> S (accmem -> SMEM)
  --[SIMT: online softmax -> requantize -> pack scales]--> P (e4m3) + P scales
  --[mesh: PV]--> O (accmem) --[SIMT: finalize]--> O (bf16)
```

## Why two directories

The two goals conflict, and conflating them produced weeks of false results.

* **Peak utilization** (sibling) wants maximum overlap: mesh `QK(i)`/`PV(i)` under SIMT
  softmax/requant/pack `(i-1)` and DMA mvin `(i+1)`, so `T_tile = max(stages)`. That overlap is
  where the utilization comes from -- and it is also what exposes a latent race.
* **Stability** (here) is willing to give the overlap back. A slow, boring, correct kernel is the
  deliverable. **Do not optimize cycles in this directory.**

## The gate

`FA_PHASE<k>` (see `mxgemm_core.hpp`) delays **cluster 1 only** by `k * 64` dependent
**read-only** MMIO round-trips (~2.4k cycles per step) at the top of every tile. It computes
nothing and touches no data, so it can only move the schedule. Therefore:

> **A configuration that fails any `FA_PHASE<k>` was never correct.**

Target -- bit-correct on **all** of:

* `NT6` (12 tile-images) and `NT8` (16)
* the **real per-invocation tile count**. For TinyLlama-1.1B, one head at S=2048 causal with
  `Sq=64`/`Sk=256` is `sum_{m=1..32} ceil(m/4)` = **144 tiles**; the harness counts tiles per
  cluster across 2 clusters, so that is **~72 tiles/cluster = NT72**. The GPU is reset between
  kernel invocations, so 144 tiles is the exposure that matters -- not a whole forward pass. This
  is the strongest available claim because it *directly matches a real invocation*, and it matters
  because the observed corruption onset is at a **specific** tile (4, 5 and 7 have all been seen):
  a config clean at NT8 whose onset is at tile 30 passes every short gate and fails every real
  invocation.
* `FA_PHASE1/2/3` and `FA_PHASE_BOTH`

### *** THE GATE IS PASSED -- `FA_ST_NOOVL`, every point, scored with `fa_verify_tiles.py` ***

| gate point | tile-images | result |
|---|---|---|
| `NT6` (`stV6`) | 12 | **12 correct, 0 wrong** |
| `NT8` (`stV8`) | 16 | **16 correct, 0 wrong** |
| `NT24` (`stV24`) | 48 | **48 correct, 0 wrong** |
| **`NT72`** (`stV72`) -- one full TinyLlama head | **144** | **144 correct, 0 wrong** |
| `NT24` + `FA_PHASE1` (`stV24p1`) | 48 | **48 correct, 0 wrong** |
| `NT24` + `FA_PHASE2` (`stV24p2`) | 48 | **48 correct, 0 wrong** |
| `NT24` + `FA_PHASE3` (`stV24p3`) | 48 | **48 correct, 0 wrong** |
| `NT24` + `FA_PHASE1` + `FA_PHASE_BOTH` (`stV24b1`) | 48 | **48 correct, 0 wrong** |
| `NT24` + `FA_PHASE2` + `FA_PHASE_BOTH` (`stV24b2`) | 48 | **48 correct, 0 wrong** |

Onset is `none(>N-1)` in **both clusters** on every row. Config:

```
FULL_ATTN2 FA_SP FA_SP_QOVL FA_SP_LEANCFG FA_SP_QKACC FA_SP_PKOVL FA_SP_QSPLIT
FA_SP_WCNT FA_SP_PAX FA_SP_CVTX  FA_ST_NOOVL
```

Cross-checked: `fa_rowdiag.py --onset` and `fa_verify_tiles.py` agree on every run. 57,256 cyc/tile,
**28.68%** utilization (vs the peak track's 35.83% at onset tile 13).

Two things this does **not** say. It is one seed (12345) -- and the simulator is timing-deterministic,
so a second seed only randomises uninitialised state and is *not* a second test of the schedule;
`FA_PHASE` and more tiles are. And `FA_PHASE` sweeps `k = 1,2,3`; a `k` beyond 3 has not been run.

The **sequential `FULL_ATTN2 FA_STEADY`** body is also clean at `NT24` (`stS24b`, 48/48) now that the
`FA_NTILES` hole is fixed -- a second, structurally unrelated de-overlapped body reaching the same
place, which is the corroboration that matters most here.

## Score configurations by ONSET TILE, not by pass/fail

The perf track established that the onset is a **deterministic function of the schedule**, not a
sampled race outcome: identical at NT16 and NT24 for one config, and at NT24 and NT72 for another.
So "passed at NT *n*" carries exactly the information "onset > *n*" -- which is why NT8 gave false
confidence for weeks -- while the onset tile is a real-valued observable that can be **ranked and
bisected against from a single run**. `fa_rowdiag.py --onset <trace>` prints it per cluster; it
excludes INCOMPLETE images explicitly, and it separates the un-overlapped path's **tile-0 prologue
defect** from a hazard onset, because conflating them would report "onset 0" for a config whose
hazard onset is actually unmeasured.

### Onsets measured here (seed 12345, `NT6` unless stated)

| config | perturbation | cl0 onset | cl1 onset |
|---|---|---|---|
| 36% config (`stD6p1`) -- the reference | `FA_PHASE1` | **4** | none(>5) |
| 36% + `FA_ST_CFGPRE` (`stE6p1`) | `FA_PHASE1` | none(>5) | **3** |
| 36% + `FA_ST_CFGPRE` (`stE6p2`) | `FA_PHASE2` | none(>5) | **1** |
| 36% + `FA_ST_CFGFENCE` (`stC6p1`) | `FA_PHASE1` | none(>5) | none(>5) |
| 36% + `FA_ST_CFGFENCE`, **NT24** (`stC24`) | none | **13** | **15** |
| **`FA_ST_NOOVL`** (`stV6`/`p1`/`p2`) | none / `P1` / `P2` | none(>5) | none(>5) |
| sequential `FULL_ATTN2 FA_STEADY` (`stS6`/`p1`) | none / `P1` | none(>5) | none(>5) |
| un-overlapped `FA_SP` (`stR6`/`p1`/`p2`/`p3`) | none / `P1` / `P2` / `P3` | tile-0 only, hazard none(>5) | same |

Two things to read off this. **`FA_ST_CFGPRE` makes the hazard *worse*** -- it pulls cl1's onset in
to 3 (`PHASE1`) and 1 (`PHASE2`), where the unfixed reference has cl1 clean at NT6. That is a
stronger refutation than "it didn't help". And **the un-overlapped `FA_SP` body's steady state
survives `FA_PHASE1`, `FA_PHASE2` *and* `FA_PHASE3`** at NT6 with only its tile-0 prologue defect --
so de-overlapping does something real. NT24 runs for `FA_ST_NOOVL` (`stV24`, `stV24p1`, `stV24p2`),
the fully de-overlapped body (`stM24`), the reference (`stD24`) and the sequential body (`stS24`) are
the measurements that turn "none(>5)" into a number.

## S IS THE SITE -- confirmed positively, per-tile and per-cluster, under FA_PHASE

`FA_SP_DUMPS` writes a per-row XOR checksum of **S** the instant `S(t)` becomes resident, before the
softmax touches it. Q, K, V and their MX scales are loop-invariant in `FA_SP`, so `S(t)` must be
bit-identical for every `t`. Measured on `stG6p2` = 36% config + `FA_SP_DUMPS` + `FA_PHASE2`, whose
O onset is cl0 = 4:

| | S row-checksums | O |
|---|---|---|
| cluster 0 | identical for tiles 0-3, then **all 64 rows change at tile 4 and stay changed** | onset 4 |
| cluster 1 | **identical at every tile** | clean at every tile |

S goes wrong in exactly the cluster and exactly the tile where O goes wrong, and nowhere else, in the
same run. That is a *positive* localization rather than the previous convex-hull inference, and it
rules out softmax, requant, pack, PV and finalize for this failure.

### The wrong S is a DRIFT, not a fixed mis-addressing -- which kills one story and names another

Three further facts, all from the same checksums at zero extra cost:

* **It is not a permutation.** 0 of 64 wrong row-checksums appear anywhere in the correct S; there
  is no rotation `k` for which the wrong S is the correct S rotated. So S is not being *read from
  the wrong place* with the right data -- it is being *computed differently*.
* **Tile 4's S differs from tile 5's S.** A fresh corruption each tile, not one latched error.
* **It gets monotonically worse:** O Frobenius 3.567 (t0-t3) -> **84.635** (t4) -> **114.566** (t5),
  reproducing the shape of the pre-existing `FA_PHASE2` record (108.6 / 113.1 / 118.2 / 119.4 /
  119.4 -- progressive, saturating).

All 64 rows at once + fresh each tile + monotonically worsening + never recovering + one cluster is
the signature of a **monotonically drifting index that starts slipping at the onset tile**, not of a
one-shot latch and not of a per-row ordering violation (which would corrupt a *subset* of rows).

That fits `ScaleFactorMem`'s odometer exactly: `counter_i/j/k_runtime` (`ScaleFactorMem.scala:70-104`)
advance on mesh scale reads, re-zero **only** by completing a full sweep of the *live* `loop_bound_*`
registers, and have no reset path. Once a sweep fails to land on zero, every later gemm reads scale
rows further off -- progressively worse, never recovering. **But `FA_ST_CFGPRE` refutes the obvious
trigger**: fencing on both sides of `CONFIG_SCALE_MEM` does not stop it (and makes it worse). So if
it is the odometer, the slip is *not* caused by a bound change landing mid-sweep.

The remaining way to slip that odometer is for one matmul to perform a **different number of
scale-enabled reads** than `bound_i x bound_j x bound_k x 16` -- a read replayed or dropped under
scale-port/SMEM contention. That would be triggered by exactly the contention `FA_PHASE` perturbs,
and it would be **immune to any software fence**, because a fence orders *commands* and not the
mesh's internal read stream. It also predicts that de-overlapping pushes the onset out, which is
what the `NT6` phase results and the perf track's `ACCRS`+`PREPK` onset of 17 both show.
> #### PRE-REGISTERED CRITERION, REVISED BEFORE MEASURING -- and the revision matters
>
> I first wrote this test as *"count `read_req.fire && scaling_enable` per matmul and compare against
> `bound_i x bound_j x bound_k x 16`"*, with a match reading as refutation. **That criterion is wrong
> and would have killed a live hypothesis on a null result.** From the peak track's static arithmetic
> over the `Sq=32` shapes (`PE_M=PE_N=PE_K=16`, `mxgemm_core.hpp:318-320`):
>
> | cfg | M,N,K | bounds | `bi*bj*bk*16` |
> |---|---|---|---|
> | QK full | 64,256,128 | 4,16,8 | 8,192 |
> | QK half | 32,256,128 | 2,16,8 | **4,096** |
> | PV full | 64,128,256 | 4,8,16 | 8,192 |
> | PV half | 32,128,256 | 2,8,16 | **4,096** |
>
> The half-tile bound product *equals* its mesh-cycle count -- one scale read per row-feed, sweep
> complete -- and `fa_mm_acc` re-issues `gemmini_mxquant_config_mvout` with `C.PE_TILES_I/J/K()` on
> **every** call, so each matmul latches its own bounds instead of inheriting the previous ones. Per
> 64 query rows the totals are **identical** across shapes: 16,384 scale reads, 768 A-scale writes.
> **So "the count matches" is the EXPECTED result on both shapes and carries no information about the
> odometer.**
>
> **The hypothesis is a RATE hypothesis, not a COUNT hypothesis.** What `Sq=32` changes is not the
> read count but the number of matmuls per 64 rows, **2 -> 4**, and hence the number of
> `CONFIG_SCALE_MEM` issuances and bounds *changes*:
>
> * drift **per scale read** => `Sq=32` behaves like `Sq=64` (identical total reads);
> * drift **per matmul / per bounds change** => `Sq=32` accumulates at **twice the rate** and should
>   fail about twice as early.
>
> Measured, and consistent with the second: `Sq=64` fails at tiles 7-17 while `Sq=32` dies at
> **half-tile 1-2** -- sooner by well more than 2x.
>
> **THE PROBE, RESOLVED AND WORKING** (so nobody re-derives the hierarchy walk). FSDB capture:
> `/tmp/fa_fsdb_go.sh s16 200000 130000 12345`. Note `+dump-start` did **not** clip anything here --
> the FSDB spans from time 0 regardless, which is convenient (it covers the healthy early matmuls too)
> but means budgeting disk for the whole run: 95 MB by cycle 100k, 474 MB for the full 200k.
> Scope, cluster 0 (`cluster_prci_domain`; cluster 1 is `cluster_prci_domain_1`, and note the module
> is `ScalingFactorMem` while the Scala file is `ScaleFactorMem.scala` -- searching the file name finds
> nothing in the netlist):
>
> ```
> TestDriver.testHarness.chiptop0.system.cluster_prci_domain.element_reset_domain_element
>   .tile_prci_domain_2.element_reset_domain_radiance_gemmini_tile_3.gemmini.spad.acc_mems_0
>   .scaleFactorMem.{counter_i_runtime, counter_j_runtime, counter_k_runtime,
>                    io_scaleMemCntl_loop_bound_{i,j,k}, read_row_addr_{act,w}}
> ```
>
> Cluster 1 differs only in `cluster_prci_domain_1` and `..._gemmini_tile_6`. The paths were resolved
> by walking instance names in `gen-collateral/*.sv` with plain `grep` (`AccumulatorMem` <- `Scratchpad`
> <- `Gemmini` <- `GemminiTile` <- `TilePRCIDomain_3` <- `RadianceCluster` <- `ClusterPRCIDomain`), which
> needs no special tooling.
>
> **READ THE SIGNALS WITH pynpi, NOT WITH THE `radiance-fsdb` MCP.** Two independent reasons:
> * *Correctness.* The MCP's `fsdb_signal_changes` returned changes from **outside** the requested
>   window with the first entry stamped at the window start -- see the tool caveat below, which nearly
>   produced a false confirmation of this very hypothesis. Its `fsdb_signal_value` point reads were
>   self-consistent, but a tool that silently mis-windows one query is not one to build a sequence on.
> * *Fit.* The observable below needs a **time grid**, and pynpi returns a whole per-cycle series in one
>   pass instead of N point queries:
>   `/home/eecs/yrh/.claude/skills/radiance-perf-viz/scripts/radiance_perf.sh --fsdb X.fsdb --out Y.npz`
>   which drives `radiance_perf_extract.py` under the Verdi-bundled python3.6
>   (`VERDI_NPI_HOME` defaults to `/ecad/tools/synopsys/verdi/V-2023.12-SP1-1`). Point the config at the
>   six signals above. The MCP is stdio (one server per client) and may or may not be connected; this
>   path does not depend on it at all.
>
> **Healthy baseline measured on `s16` (`Sq=32`, `NT16`), cluster 0.** `counter_k_runtime` advances
> `0 -> 1 -> 2 -> 3` at 191.739 / 192.763 / 193.787 Mps, i.e. **one increment every 512 cycles
> exactly** -- which is `bound_i x bound_j x 16 = 2 x 16 x 16 = 512` for the QK half-tile, so the
> odometer behaves as derived and the units are confirmed. Probe `counter_k_runtime` (the slowest of
> the three) for a compact per-matmul picture; `counter_i_runtime` changes every 16 reads.
>
> **REVISED OBSERVABLE:** the per-matmul **odometer state sampled at each bounds change** --
> `counter_i/j/k_runtime` immediately before and after every `CONFIG_SCALE_MEM` -- and not any total
> count. Confirmation is `counter_* != 0` at a bounds change; refutation is `counter_* == 0` at every
> bounds change up to and including the failing matmul. All six signals are `dontTouch`'d
> (`ScaleFactorMem.scala:73-75`, `ExecuteController.scala:163-165`), so they survive into the netlist
> and are directly probeable.

### Reading a 1-cluster (FPGA) run

The board carries 1 of the 2 clusters, so the same kernel yields **half** the tile-images (`NT6` = 6,
not 12) and cluster 1 is simply absent. `fa_rowdiag.py --onset` prints
`cl1: ABSENT (no O stores -- expected on a 1-cluster board)` rather than letting the empty case fall
through to a scary-looking `onset none(>-1)`. Verified against a synthetically 1-clusterized trace.
The surviving mechanism candidate is intra-cluster and `FA_PHASE_BOTH` already argued against an
inter-cluster contest, so a 1-cluster board is well matched to what is actually being tested.

## Status

| gate | best known |
|---|---|
| NT6 (12 images) | several configs, 12/12 |
| NT8 (16 images) | `FA_SP_ACCPAD`+`FA_SP_PREPK`, 16/16 |
| NT72 (~1 TinyLlama head) | **`FA_ST_NOOVL`, 144 of 144 tile-images** (`stV72`) |
| `FA_PHASE1/2/3` | `FA_ST_NOOVL`: `P1` and `P2` clean at NT6 **and NT24**; `P3`/`BOTH` in flight |

### `FA_ST_NOOVL` -- the first configuration to survive a large NT

| run | config | tiles | perturbation | result |
|---|---|---|---|---|
| `stV72` | `FA_ST_NOOVL` | **NT72** | none | **144/144, onset none(>71) both clusters** |
| `stV24` | `FA_ST_NOOVL` | NT24 | none | 48/48, onset none(>23) both |
| `stV24p1` | `FA_ST_NOOVL` | NT24 | `FA_PHASE1` | **48/48, onset none(>23) both** |
| `stV24p2` | `FA_ST_NOOVL` | NT24 | `FA_PHASE2` | **48/48, onset none(>23) both** |
| `stM24` | fully de-overlapped + `LEANCFG` | NT24 | none | 48/48, onset none(>23) both |
| `stD24` | **the 36% reference** | NT24 | none | **cl0 onset 13**, cl1 none(>23) |
| `stC24` | 36% + `FA_ST_CFGFENCE` | NT24 | none | **cl0 onset 13**, cl1 onset 15 |

NT72 is 144 tile-images = one full TinyLlama head at `Sq=64`/`Sk=256` causal, i.e. the exposure a real
kernel invocation actually sees. **Still outstanding before the gate is met:** `FA_PHASE3`,
`FA_PHASE_BOTH` (both `k`), and an explicit `NT8` -- all four launched (`stV24p3`, `stV24b1`,
`stV24b2`, `stV8`). Nothing measured so far fails, but the gate is not passed until those land.

Two side results from the same table. `FA_ST_CFGFENCE` leaves cl0's unperturbed onset at **13**,
identical to the unfixed reference -- so it is **not a fix**, and its clean `FA_PHASE1` run at NT6 was
a reshuffle. And the fully de-overlapped body (`stM24`) is clean to 23 as well, so the robustness is
not specific to `FA_ST_NOOVL`'s particular staging -- it tracks *removing the overlap*.

### What it costs

| config | cyc/tile (steady, NT24) | util | onset |
|---|---|---|---|
| 36% reference (`stD24`) | 45,827 | 35.83% | 13 |
| **`FA_ST_NOOVL`** (`stV24`) | **57,256** | **28.68%** | none(>71) |
| fully de-overlapped (`stM24`) | 63,952 | 25.68% | none(>23) |

`FA_ST_NOOVL` buys the exposure from tile 13 to beyond tile 71 for **25% more cycles per tile**
(35.83% -> 28.68% utilization). Per this directory's charter that is a good trade; per the sibling's
it is not, which is exactly why there are two directories.

Everything below was measured with the scripts and tools in this directory; the run set is
reconstructible from disk with **`./fa_runtable.sh /tmp/struns`** (tag, whether the sim reached
`$finish`, correct/wrong tile-image counts, and the flag set + RV32-segment sha recorded at build
time), and any single trace can be re-diagnosed with `fa_verify_tiles.py` (the verdict) or
`fa_rowdiag.py` (which stage). `/tmp/st_build.sh <tag> "<DEFINES>"` and
`/tmp/st_go.sh <tag> <elftag> <seed> <budget_cycles> 16420 <marks_per_tile>` build and launch in
**this** directory only. Marks per tile: 7 for the `FA_SP` bodies, **8** with `FA_ST_NOOVL` (it adds
a stage), 11 for the sequential `FULL_ATTN2 FA_STEADY` body.

### The de-overlap plan has a prerequisite: the un-overlapped path gets TILE 0 wrong

> **CORRECTION, and it is mine.** This section first said "the un-overlapped path is BROKEN". It is
> not. Once the `FA_NT2` runs *completed*, the scored result is **tile 0 wrong, tile 1 CORRECT at
> 3.5666%** -- at every rung of the ladder. The steady state of the un-overlapped body is bit-exact;
> only its **first** tile is wrong. The stronger claim came from reading a *partial* trace, whose
> first (and only) complete group is tile 0 -- so a tile-0-only defect is indistinguishable from a
> total failure until the run finishes. I killed seven NT6/NT8 runs on that partial evidence; their
> tiles 1-5 would have shown the steady state immediately. **Do not score a growing trace and act on
> it: `fa_verify_tiles.py` reports later groups as INCOMPLETE for exactly this reason, and the fix is
> to wait, not to reason around it.**

The first thing the suggested approach asks for -- *"no `(i+1)` prefetch, no `(i-1)` overlap"* -- is
reachable by simply **not** setting `FA_SP_QOVL` / `FA_SP_QKACC` / `FA_SP_QSPLIT` / `FA_SP_PKOVL`.
Preprocessing `FULL_ATTN2 FA_SP FA_SP_WCNT FA_NT6` confirms that gives exactly the intended shape --
seven stages, every one bracketed by `FAP_BAR`, every mesh op issued *and drained* inside a single
stage with all six warps parked, no cross-tile anything:

```
for t:  MARK
  S0 [w0] fa_cfg<QKF>; fa_mvin_A(Q); fa_scl(Q scales)      BAR(3)
  S1 [w0] fa_gf; fa_mm<QKF> -> SP_C; fa_gfl                BAR(4)
  S2 [all] online_softmax_block  (in place over S)         BAR(5)
  S3 [all] requant_P_to_spad_tiled -> P8 + scales          BAR(6)
  S4 [all] pack_scales_to_sfmem (thread 0 internally)      BAR(7)
  S5 [w0] fa_cfg<PVF>; fa_mm<PVF> -> SP_C; fa_gfl          BAR(8)
  S6 [all] finalize_O -> GMEM                              BAR(9)
```

**It does not compute the right answer at tile 0, deterministically -- and it is right from tile 1
on.** Measured (2026-07-31, seed 12345, `fa_verify_tiles.py` against `golden_O_u16.npy`):

| build | cl0 t0 | cl0 t1 | cl1 t0 | cl1 t1 |
|---|---|---|---|---|
| `stN0` (`FA_NT2`, completed) | 24 NaN rows | **CORRECT 3.5666%** | 1 NaN row | **CORRECT 3.5666%** |
| `stN1`..`stN4` (each rung) | same | **CORRECT** | same | **CORRECT** |
| `stB6` / `stB6p1` / `stB6p2` / `stB6p3` / `stB8` | 153.879%, same 24 rows | -- | 79.026%, same row | -- |

Tile 0 is bit-identical across no skew, `FA_PHASE1`, `FA_PHASE2`, `FA_PHASE3`, `FA_PHASE_BOTH` and
`FA_NT8`, i.e. across a 2.4k-, 4.8k- and 7.2k-cycle skew of cluster 1. So this is **not** the race
-- it is a **prologue / warm-up** bug, and it is *separate from* the hazard this directory exists to
chase (that one is phase-sensitive and starts at a *later* tile). Two further facts that narrow it:

* **It is not `l`.** Fitting each non-NaN row to the best scalar multiple of golden leaves a
  **median 78% residual** (max 93%), so `O_unnorm` itself is wrong, not the softmax denominator.
  A pure `l` error would fit at ~0%.
* **The two clusters mostly agree.** Rows 1,2,3,17,18,19,33,63 have *identical* fitted scale and
  residual in both clusters; they differ only at rows 0, 16, 32 (the first row of each 16-row PE
  row-tile) plus the NaN rows. So the defect is largely deterministic and data-dependent, with a
  small timing-dependent component at PE-tile boundaries.

The NaN rows are bf16 `0x7FC0` and the shape is *"first 4 of 16 rows fine, last 12 wrong"* in PE
row-tiles 0 and 1, with row-tiles 2 and 3 clean -- worth keeping in view, because
`PE_TILES_I() == 4`.

**Consequence for the plan:** the fully de-overlapped body is *usable* -- its steady state is
bit-exact -- but it cannot pass a gate that scores every tile-image until tile 0 is fixed. Because
the defect is at tile 0, every experiment about it is an `FA_NT2` question: ~20 minutes per point
rather than ~75.

**The prologue is the suspect, and `FA_ST_PROLOGF` is the bounded test.** The prologue's K and V
weight-scale words are written by `fa_scl`, i.e. ordinary Muon SIMT stores to the gemmini's
scale-SRAM TL slave, and the only thing between them and tile 0's QK matmul -- which *reads* those
weight scales -- is `FAP_BAR(2)` = `mu_fence_smem()` + `vx_bar`. This file documents twice
(`FA_SP_QGF`, and the `1cce749` fix) that `mu_fence_smem()` is **not** a drain for an SF-SRAM scale
write, and that the working primitive is `gemmini_fence()` -- a load from the same gemmini TL port,
which orders every preceding store to it. Every configuration that gets tile 0 *right* happens to
have such a fence there already: `FA_SP_QKACC`'s priming `fa_gf()`. That is the accidental-slack
pattern this campaign keeps finding. `FA_ST_PROLOGF` makes it explicit and unconditional for one
MMIO round trip (~37 cyc) **once per kernel**.

### The upward ladder, at NT2 (the defect is at tile 0, so NT2 is enough)

All on `FULL_ATTN2 FA_SP FA_SP_WCNT`, seed 12345, cluster 0 / cluster 1 tile 0:

| rung | added | tile 0 | tile 1 | verdict |
|---|---|---|---|---|
| `stN0` | -- | WRONG (24 / 1 NaN rows) | CORRECT | 2 of 4 |
| `stN1` | `PKOVL` | WRONG, **same rows** | CORRECT | 2 of 4 |
| `stN2` | `+ QOVL` | WRONG, **same rows** | CORRECT | 2 of 4 |
| `stN3` | `+ QKACC` | WRONG, **same rows** | CORRECT | 2 of 4 |
| `stN4` | `+ QSPLIT` | WRONG, **same rows** | CORRECT | 2 of 4 |
| `stN5` | `+ LEANCFG PAX CVTX` (= the 36% config) | **CORRECT** | **CORRECT** | **4 of 4** |

So `PKOVL`, `QOVL`, `QKACC` and `QSPLIT` -- the four overlap flags -- do not affect tile 0 at all,
and every rung's steady state is bit-exact. The only difference from the verified-correct
configuration is `FA_SP_LEANCFG` + `FA_SP_PAX` + `FA_SP_CVTX`, and **the split is clean**:

| build | flags on `QOVL QKACC PKOVL QSPLIT WCNT` | cl0 t0 | cl1 t0 |
|---|---|---|---|
| `stN7` | `PAX` + `CVTX` | **153.879%, 24 NaN rows** | **79.026%, 1 NaN row** |
| `stN6` | `LEANCFG` | **CORRECT 3.5666%** | **CORRECT 3.5666%** |

**`FA_SP_LEANCFG` is a tile-0 correctness flag, not just the performance flag it is documented as.**
`PAX`/`CVTX` are irrelevant to tile 0, exactly as their bit-exactness-by-construction predicts.
`LEANCFG`'s only effect is to stop calling `configure_mxgemmini` per gemm, so **calling
`configure_mxgemmini` at all in this body is what breaks tile 0** -- and it does so identically
whether or not any overlap flag is set.

Two candidate sub-mechanisms, both inside `configure_mxgemmini`, neither yet pinned:

* it issues an extra `CONFIG_SCALE_MEM` per gemm -- the netlist defect above; and
* it issues a **pair of `gemmini_loop_ws_config_bounds` commands outside any
  `gemmini_loop_ws_spad` sequence**, and `LoopMatmul.scala:1122` writes those bounds into the
  `loop_being_configured` slot *without* setting its `configured` bit (`LoopMatmul.scala:1162`
  is what sets it) -- so a stray pair lands in the slot the next real `LOOP_WS` will use.
  Those two calls are redundant anyway: `gemmini_loop_ws_spad` re-issues the bounds itself.

`FA_ST_PROLOGF` (`stP2`) tests the drain hypothesis. If a drain does not fix tile 0, deleting the two
stray `loop_ws_config_bounds` calls is the next thing to try, and it is free.

**Note the shape of the inference, because it is the interesting part.** All three of those flags
are *per-tile* changes, so none of them can *cause* a tile-0-only failure by its own semantics --
`PAX` and `CVTX` are XOR permutations of an index (`PAX`'s reduction is an order-independent `fmax`),
and `LEANCFG` only deletes a redundant `configure_mxgemmini`. What they can do is move the *timing*
of the prologue-to-tile-0 hand-off. So the expected reading of `stN6`/`stN7` is **not** "flag X is
broken" but "flag X shifts a prologue race", which is why the fix under test is a drain
(`FA_ST_PROLOGF`) and not a flag.

`fa_rowdiag.py` in this directory is the tool these rows came from.

### FA_ST_CFGPRE is REFUTED -- by its own experiment

The `CONFIG_SCALE_MEM`-ordering mechanism below predicts the whole fingerprint and has a netlist
defect behind it, and it is **still not a sufficient fix**. Measured on the 36% config at NT6:

| build | flags added | verdict |
|---|---|---|
| `stE6p2` | `FA_ST_CFGPRE` + `FA_PHASE2` | **cluster 1 tile 1 = 119.367% WRONG** |
| `stE6p1` | `FA_ST_CFGPRE` + `FA_PHASE1` | **cluster 1 tile 3 = 91.435% WRONG** |

It fails at **two different `k`**, at different tiles, in the delayed cluster -- so it is not a
single unlucky alignment.

119.367% is the *same value* the pre-existing `FA_PHASE2` record reports for this hazard
(`mxgemm_core.hpp`: `... 118.1504% 119.3714% 119.3714%`), so it is the same failure, not a new one.
A `gemmini_fence()` immediately before the `CONFIG_SCALE_MEM` **and** one between it and the
`LOOP_WS` -- which together satisfy both halves of the ordering argument -- do not close it. That
is a fifth refuted mechanism for this campaign. The netlist finding stays on the record because it
is a true defect and it is the reason `FA_SP_LEANCFG` matters, but it is **not** this bug.

**The fences were verified to exist in the binary before the refutation was believed** -- this file
already records one "settle" that clang folded into 16 independent pipelined loads that guaranteed
nothing (`FA_CFGSETTLE2`'s note). Static `lw.shared` count over the linked GPU ELF:

| build | added | `lw.shared` |
|---|---|---|
| `stD6p1` | -- (control) | 56 |
| `stC6p1` | `FA_ST_CFGFENCE` (1 fence per matmul) | **60** |
| `stE6p1` | `FA_ST_CFGPRE` (2 fences per matmul) | **64** |

Monotone +4 per added fence pair, i.e. both fences are really emitted in both `fa_mm` and
`fa_mm_acc`. The refutation is of the mechanism, not of a fence that silently vanished.

## What is already ruled out -- do not re-derive

Established by measurement, several by refuting our own hypotheses:

* **The accumulator store is NOT the site**, by elimination: `ReservationStation.scala`'s STORE
  branch already implements the needed ordering (`deps_ex` for an `opa_is_dst` entry includes
  *"raw for st b <- ex a"* -- the store's accumulator source against the compute's accumulator
  destination), and `AccumulatorMem.scala:619-624` adds a same-row RAW interlock across all three
  write-pipeline stages, live in silicon and honored on both consumers. The failure survives both.
* **Every drain/pad fix is dead.** `FA_SP_ACCPAD` costs +1,694 cyc/tile for nothing; a pad at the
  RTL-derived 4-cycle bound gives 9/12 (so 4 is not the true drain); pad length is
  **non-monotone** in correctness (128 ok, 2,048 wrong, ~69k ok, ~106k wrong); `qmax` 2/12 vs
  `ymax` 12/12 differ *only* in pad length. The measured drain gap is real but is not this bug:
  `runningLoops` goes idle **141 cycles** before the last `acc_mems_0.io_write_valid`.
* **Inter-cluster relative phase is NOT the trigger.** `FA_PHASE_BOTH` -- identical delay in both
  clusters, relative phase back to ~0 -- still fails 10/12. A two-cluster contest is not required
  to explain this bug. The harness is exonerated independently: an ALU-only, MMIO-free,
  warp-0-only pad sweep also breaks it (N=2048 -> 7/11, N=8192 -> 6/10).
* **Fabric misrouting is excluded.** By instance-graph walk over all 3201 generated `.sv` files:
  the SMEM fabric, `clcbus`, `ScalingFactorMem`, `RWSplitterNode` and every source-shrinker are
  strictly **per-cluster**; at each genuinely shared point the clusters' source IDs are disjoint by
  hard-wired constant; and `TLCacheCork` erases cluster identity at the L2 boundary.
* **Correctness is non-monotone in the flag set**, so no flag has been shown broken by its own
  semantics: `2PBM+SMBMAX+PREPK` is 2/12; add `BANKA` and it is 12/12; `BANKA` alone is 7/12.
* **Nothing pollable reflects accmem completion.** The readable MMIO surface is exactly three
  fields (`0x08` cmd-ready, `0x20 io.busy`, `0x28 runningLoops`); every other offset is
  `RegField.w` and reads back constant zero with a normal ack. `0x28` is **issue-based** --
  `LoopMatmul.scala:497` goes idle when the last COMPUTE is *accepted into the reservation
  station* -- so no fixed pad can make it safe.
* ~~`matmul_in_progress` was dead-code eliminated by `num_counter = 0`.~~ **REFUTED
  2026-07-31** -- see the RTL finding below. It is live in the taped-out netlist. What is
  actually wrong is *what it gates*.

## Corruption fingerprint

Localized to **S**: every wrong cell lies inside V's per-column convex hull (0 of 8192 outside),
so P and l stay mutually consistent -- which exonerates PV, its operand spad, V's scales and
finalize. Successive wrong tiles share **0 of 4096 words**, so each tile is freshly corrupted
rather than inheriting one damaged resident operand. Values latch around 93-121% of golden.

## A fifth candidate, from the taped-out netlist: the scale-mem config is applied UNGATED

Read out of `RadianceTapeoutSimConfig/gen-collateral/ExecuteController.sv` on 2026-07-31 -- the
elaborated Verilog, not the Chisel, so this is what is in silicon:

```systemverilog
wire _GEN_6 = _cmd_q_io_deq_bits_0_cmd_inst_funct == 7'h1A;   // 0x1A = 26 = CONFIG_SCALE_MEM
always @(posedge clock) if (reset) ... else begin
  ...
  if (_GEN_6) begin                                           // <-- THE ENTIRE ENABLE
    loop_bound_i <= _cmd_q_io_deq_bits_0_cmd_rs1[41:33];
    loop_bound_j <= _cmd_q_io_deq_bits_0_cmd_rs1[50:42];
    loop_bound_k <= _cmd_q_io_deq_bits_0_cmd_rs1[59:51];
    scale_mem_read_act_sel <= ...[60];  scale_mem_read_w_sel <= ...[61];
  end
wire _GEN_26 = _GEN_6 & ~matmul_in_progress & ~_GEN_24;        // gates only the POP
```

**The interlock for exactly this hazard exists and is applied to the wrong signal.** The MX
scale-memory addressing registers are written from the funct field of the EX command queue's head
entry and nothing else -- not `_cmd_q_io_deq_valid_0`, not `control_state`, and **not
`matmul_in_progress`**. The `!matmul_in_progress` guard at `ExecuteController.scala:724` -- whose
own comment reads *"Registers are already updated at lines 135-144"* -- delays only the completion
back to the reservation station. The register write is eager and unconditional.

Why that is sufficient to produce the whole fingerprint: `ScaleFactorMem.scala:78-104` advances
`counter_i/j/k_runtime` on mesh scale reads and wraps them **only** when each equals
`loop_bound_* - 1` *against the live register*, and there is no reset path
(`scale_mem_counter_reset_flag` is computed in `ExecuteController` and never consumed by
`ScaleFactorMem` -- checked; the `mxgemm_core.hpp` claim is correct). The read rows are
`loop_bound_i*(counter_k>>1)+counter_i` (act) and `loop_bound_j*(counter_k>>1)+counter_j`
(weight), and the two gemms differ in `loop_bound_j` (QKF 16, PVF 8) but not `loop_bound_i`
(both 4). So one mis-timed bound change strands the odometer **permanently**, corrupts the
**weight** side only, and gives: onset at a specific tile, latching, progressively worse, never
recovering, one cluster (each cluster has its own gemmini), magnitude preserved (aliased E8M0
bytes span a narrow exponent range).

Two corollaries:

* **`matmul_in_progress` is live**, contrary to the entry crossed out above:
  `ExecuteController.sv:672` is a 6-input OR of `_mesh_io_tags_in_progress_{0..5}_rob_id_valid`
  off the `MeshWithDelays` tag queue.
* **The enable can be true while the EX queue is EMPTY.** `MultiHeadedQueue.scala:32` drives
  `io.deq.bits(i) := regs(wrappingAdd(raddr, i, entries))` -- an unconditional register-file read
  with no validity qualification -- and the EX queue is 8 deep here (`MultiHeadedQueue_1.sv` has
  `regs_0..regs_7`). So whenever the EX queue drains, `loop_bound_*` is re-driven by whatever
  command sat in that ring slot 8 commands ago; if that slot holds a `CONFIG_SCALE_MEM` the bounds
  silently revert to the *other* gemm's values for as long as the queue stays empty. The outcome
  then depends on a **modular** alignment of the command stream -- the only kind of thing that can
  be **non-monotone in an added delay**, which is the single most stubborn fact in this campaign
  (pad length: 128 ok, 2,048 wrong, ~69k ok, ~106k wrong). *This corollary is a hypothesis, not a
  measurement: it needs the waveform (does `loop_bound_j` ever change while `read_fire_d1` is
  high?), and no software flag tests it.*

Software consequence, and it is **not** what `FA_CFGSETTLE` implements: the sound rule is not
"separate the config from the matmul" but **"never issue a `CONFIG_SCALE_MEM` with a matmul
outstanding"** -- drain to `io.busy == 0` *immediately before* it, with no intervening gemmini
command. That is `FA_ST_CFGPRE` (`kernel.cpp`, ~37 cycles x2 per tile). `FA_ST_CFGFENCE` is the
weaker half (the fence *after* the config) and exists as the A/B control that separates the two.
Note `FA_SP_LEANCFG` **off** gives the same guarantee for free, because `configure_mxgemmini` ends
in a `gemmini_fence()` and is followed by the move-in and the scale stores -- which is exactly the
"why the baseline is safe" paragraph in `mxgemm_core.hpp`, now with a netlist-level reason.

## The other four live candidates -- two of them can be closed by reading the RTL

1. **SMEM-side visibility after the mvout** -- **substantially closed, no experiment needed.**
   The accumulator -> SMEM move-out goes through `spad_writer`, a `StreamWriter` over the TL
   ext-mem port (`Scratchpad.scala:302,477-490`), and

   ```scala
   io.busy := xactBusy.orR || (state =/= s_idle)                  // DMA.scala:407
   xactBusy_remove = ~Mux(tl.d.fire, (1.U << tl.d.bits.source), 0.U)   // DMA.scala:402
   ```

   A transaction clears **only when its TileLink D-channel response fires**, i.e. when the SMEM
   manager has acknowledged the write. `Scratchpad.io.busy` ORs that with the three write queues
   (`Scratchpad.scala:635`) and `Controller.io.busy` ORs in `spad.module.io.busy`
   (`Controller.scala:786`), which is MMIO `0x20`. So a `gemmini_fence()` that returns 0 does imply
   every accumulator -> SMEM write has been TL-acknowledged. This is *not* a hole software can be
   blamed for -- unless `io.busy` itself is lying, and the one documented way it can
   (`ReservationStation.scala:140`, a solitary PRELOAD reading as not-busy) remains open.
2. **Ordering across *different* accumulator rows** -- still untested. The interlock at
   `AccumulatorMem.scala:619-624` is per-row only.
3. **The requantizer path -- RULED OUT for this kernel.** Both gemms set `QUANT_OUTPUT = false`
   (`kernel.cpp`, `QKF`/`PVF`), so `configure_mxgemmini` passes C-datatype
   `GEMMINI_FORMAT_FULL = 3`, and `ExecuteController.scala:693-698` is
   `when (output_mx_format =/= 3.U) { enable_mxquant := true } .otherwise { false }`. That feeds
   `spad.module.io.enable_MXQuant` (`Controller.scala:242`), where
   `writeData_is_full_width := !is_garbage && !enable_MXQuant` (`Scratchpad.scala:443`) --
   so the move-out takes the full-width path and **the `MxRequantizer` is bypassed for the entire
   kernel**. (Which also means `CONFIG_SCALE_MEM`'s `rs1[62]` -- the only counter-reset bit that
   exists -- resets a counter in a block this kernel never uses: `MxRequantizer.scala:559`. It does
   **not** reach `ScaleFactorMem`'s odometer; independently re-verified.)
4. **The 4-entry un-backpressured spad read queue at `Scratchpad.scala:220` -- RULED OUT, twice
   over.** *Structurally:* `Scratchpad.scala:213` is
   `io.read.req.ready := q_will_be_empty && ext_mem.read_req.ready`, elaborated as
   `ScratchpadBank.sv:122`, so the bank accepts a new SMEM read **only when the response queue will
   be empty** -- at most one outstanding read per bank. A 4-entry `dma_q` cannot overflow behind a
   window of one. *Empirically:* the overflow assertion is **live in the taped-out netlist** --
   `ScratchpadBank.sv:123-130`, guarded only by `` `ifndef SYNTHESIS ``, emitting
   `$error("Assertion failed: DMA queue does not have enough entries")` plus `$fatal` -- and it has
   **never fired in any run in this directory, including every run that produced a corrupt tile**
   (`stE6p2`, `stN0`-`stN4`, `stB6`*). The `// TODO: do backpressure` comment is real but the
   hazard it warns about is unreachable at this issue rate.

   *Methodology note, because it nearly invalidated this check:* `fa_run.sh` uses
   `cmd 2>&1 > $T.log | grep ... > $T.out`. Bash applies those left to right, so **stderr goes to
   the grep and stdout goes to the `.log`** -- which is why the `[ISSUE]` trace (stderr) is what
   gets filtered, and why a VCS `$error` (stdout) lands in the `.log`. Grepping the `.out` for an
   assertion would silently find nothing forever.

**So of the four candidates, only #2 -- ordering across *different* accumulator rows -- is still
standing**, together with the one documented way `io.busy` can lie (`ReservationStation.scala:140`:
a solitary PRELOAD reads as not-busy).

## The current stability candidate: `FA_ST_NOOVL`

Since clearing the overlap flags lands on a broken path, the de-overlap is done by **subtracting the
overlap from the fully-featured `FA_SP_QSPLIT` body** -- the body every verified 12/12 in this
campaign was measured on. `FA_ST_NOOVL` (see the block above `fa_mm` in `kernel.cpp`) removes the
three things that straddle a stage boundary in `FA_SP_QSPLIT`:

1. QK(t+1)'s 8,210 mesh cycles running underneath `finalize(t)` (stage S6);
2. Q(t+1)'s 8 KB move-in DMA, issued in S4 and transferring on into S5/S6;
3. Q(t+1)'s 64 SF-SRAM scale stores running underneath the PV matmul (stage S5).

Stage S6 splits into **S6a** (warp 0 alone, everyone else parked: Q mvin -> *drain* -> Q scales ->
fence -> cfg QKF -> QK issue -> *drain*) and **S6b** (all six warps: finalize). Verified in the
preprocessed body: after the flag, the only concurrency left anywhere in the tile is the warp-0 SF
pack against the warps-1-5 requant convert in S4, which is SIMT-vs-SIMT and touches no mesh, no DMA
and no gemmini port. Every mesh operation is issued and drained inside one barrier-bracketed stage
with the other five warps at a barrier. It costs roughly +12k cyc/tile; that is the point.

It also happens to be the right experiment for the **one remaining** candidate (ordering across
different accumulator rows), because it puts the QK compute, the accumulator -> spad store and the
PV compute in three different stages each separated by a real drain.

`FA_ST_NOOVL` adds a stage, so it emits **8** marks per tile, not 7.

### `FA_ST_NOOVL` is three independent removals -- `FA_ST_OVL_QK` / `_DMA` / `_SCL` put them back

Collapsing them into one flag was right for getting a correct baseline and wrong for everything after.
With onset as the observable, re-admitting **one** overlap and watching whether the onset moves both
names the racing stage *and* buys cycles back when it doesn't move. Each `FA_ST_OVL_*` restores one:

| flag | puts back | worth |
|---|---|---|
| `FA_ST_OVL_QK` | QK(t+1)'s 8,210 mesh cycles under `finalize(t)` (no S6a drain; finalize in the `else`) | ~8.2k |
| `FA_ST_OVL_DMA` | Q(t+1)'s 8 KB move-in issued in S4, transferring into S5/S6 | smaller |
| `FA_ST_OVL_SCL` | Q(t+1)'s 64 SF-SRAM scale stores under the PV matmul | smaller |

All four shapes verified in the preprocessed body (the S5/S6 call-and-barrier sequence), and
`FA_ST_OVL_*` without `FA_ST_NOOVL` is an `#error`.

**The refactor is provably inert on the gate result:** rebuilding plain `FA_ST_NOOVL FA_NT24` after it
reproduces RV32-segment sha `82e73680240b60f2`, byte-identical to the gate-passing `stV24` image. So
the table above is not invalidated by the restructuring.

### Where 57,135 cyc/tile actually goes -- the per-stage table for `FA_ST_NOOVL`

From `stX24` (48/48 correct, so the timing is admissible), `fa_marks3.py --per 8 --mesh 16420`,
mean over 15 steady tiles, cluster 0:

| stage | cycles | what it is |
|---|---|---|
| s0 | 52 | top-of-tile mark |
| s1 | 885 | accumulator -> S(t) @ SP_C (`ACCRS` removed the pre-store drain) |
| s2 | **12,825** | softmax, cooperative, all six warps |
| s3 | 2,754 | requant pass A (with `PAX`) |
| s4 | 10,233 | warp 0 SF pack (with `PREPK`) \|\| warps 1-5 requant convert |
| **s6a** | **11,906** | *the stage `FA_ST_NOOVL` adds*: Q(t+1) prefetch **+** QK(t+1) issue **+** QK drain |
| s5 | 8,870 | PV matmul -- **warp 0 spins in a fence for ~8.6k of it** |
| s7 | 9,081 | finalize(t) -> GMEM |
| sum | 56,606 | (pooled interval 57,135) |

Two readings that set the remaining strategy. **S6a is 11,906**, of which ~8,210 is QK's own mesh time
and the balance ~3,700 is the serialized Q prefetch: an 8 KB move-in plus **64 strictly serial
SF-SRAM scale words at ~65 cyc each (~4,160)**. And **S5 leaves warp 0 idle for ~8.6k cycles** inside
a `gemmini_fence`. So the scale words have a free home, which is exactly what `FA_ST_OVL_SCL`
restores -- and it is a *scale-store relocation*, not a mesh/SIMT overlap, i.e. a different risk class
from the one `OVL_QK` just failed at.

### `FA_ST_OVL_QK` IS REFUTED ON BOTH AXES, and the de-overlap cost is therefore mostly structural

| run | correctness | cycles |
|---|---|---|
| `stY24` (`OVL_QK`, unperturbed) | **45 of 48 -- 3 wrong** | **VOID** (not all-correct) |
| `stY24p2` (`OVL_QK` + `PHASE2`) | **9 of 24** | **VOID** |

Putting QK(t+1)'s mesh work back underneath `finalize(t)` re-introduces the hazard, and because the
run is not all-correct its cycle number is unusable -- so the ~8.2k payback it was projected to return
**cannot even be claimed**. That is the single largest piece of the 11,429 cycles de-overlapping cost,
and it is **not recoverable**. *The de-overlap cost is mostly structural, not mostly recoverable.*
**Do not retry `OVL_QK` in any variant.**

### Utilization: target 30%, currently 28.74%

Need <= 54,733 cyc/tile. **`ACCRS`+`PREPK` bought only -121 cycles here** (57,256 -> 57,135 =
**28.74%**), not the ~2,349 they are worth on the peak body -- so these levers **do not transfer at
face value**, and any projection that adds up peak-body savings on this body is unsound. Robustness is
retained (`stX24` 48/48 unperturbed, `stX24p1` 48/48 at `PHASE1`, `stX24p2` 48/48 at `PHASE2`), so they
stay; they are nearly free rather than decisive. **Still ~2,402 cycles short.** Both were confirmed
wired into this body by preprocessing (`PREPK` was found *not* wired into the `Sq=32` body, so this was
checked rather than assumed):

* **`FA_SP_ACCRS`** (~-1,915) deletes the pre-store `fa_gfl` in S1. On the peak body that rests on
  `ReservationStation.scala`'s interlock; **here the argument is stronger and needs no interlock at
  all** -- verified in the preprocessed body that between S6a's `fa_gfl` and the next tile's
  `fa_store_acc` there is *no mesh operation whatsoever* (only a barrier, `finalize_O`, a barrier and
  the marks). So `ACCRS` removes a poll of an already-drained mesh.
* **`FA_SP_PREPK`** (-306 to -434), bit-exact by construction.

### One non-overlap lever left, and a distinction worth stating explicitly

The steer was *"do not spend slots on the group-of-two softmax (a +1,280 loss **once `FA_SM_2P` is the
baseline**)"*. That prohibition is conditional on `FA_SM_2P` already being present -- and **it is not in
this config**. The gate-passing flag set has no `FA_SM_2P`/`FA_SM_2PRAW`, so `s2 = 12,825` is the
*unoptimized* cooperative `online_softmax_block`. Adding the pair is recorded in the sibling README as
**45,582 -> 42,846, i.e. -2,736 cycles** -- more than the 2,402 still needed. So this is not the
prohibited item; it is a different, untried one on this body.

Why it is the right risk class: it is a **SIMT-only arithmetic restructuring** that touches no mesh, no
DMA and no gemmini port -- it replaces four `fence.s` *per row* with three *per warp*. And
`flash_mx_impl.hpp:1139` argues bit-exactness term by term (`m` via order-independent `fmax`; `l` by
reproducing `warp_tree_reduce`'s exact 16-leaf balanced pairing, which matters because bf16 addition is
not associative). If it really is bit-exact, correctness can only move by *schedule* perturbation --
and this body is already clean at `PHASE1/2/3` and `PHASE_BOTH`.

The known caveat, stated because it is the reason this needs measuring and not assuming: on the
**overlapped** body `2P`+`2PRAW` was the fastest thing measured and **did not bank** (12/12 at NT6 but
15/16 at NT8). That was with the hazard live. Whether it holds on a body whose hazard is gone is
exactly the open question. Bonus already observed at build time: it drops the per-warp register budget
from UPPER **219 to 156**, well clear of the renamer bracket.

Launched: `stQ24` (NT24, unperturbed) and `stQ24p2` (NT24 + `PHASE2`).

In flight: `stX24`/`p1`/`p2` (`ACCRS`+`PREPK` at NT24, unperturbed + `PHASE1` + `PHASE2`) for the cycle
number *and* confirmation that robustness is retained; `stY24`/`stY24p2` (`OVL_QK`) for the
biggest single payback and for whether the onset moves when the mesh/SIMT overlap comes back.
Deliberately **not** run: the group-of-two softmax (+1,280 once `FA_SM_2P` is baseline) and
`FA_SP_ACCPAD` (+1,694 for nothing that survives perturbation).

**It computes correctly** -- `stV2` (`FA_NT2`) is 3.5666% in both clusters at tile 0, so the
restructuring is bit-exact as intended and the register budget (UPPER 219, in the unresolved
`(216, 246]` band) does not trip the renamer. Its `FA_NT6` phase sweep (`stV6`, `stV6p1`, `stV6p2`,
and the built-but-unlaunched `stV6p3` / `stV6b1` / `stV6b2` / `stV8` / `stV24`) is the gate run.

## "The non-pipelined FULL_ATTN2 path" is not actually serialized -- check before assuming

The plan names the non-pipelined `FULL_ATTN2 FA_STEADY` body as the de-overlapped option. It has
**three** barriers per tile (`bar2` after QK, `bar3` after pack, `bar4` after PV) against the
`FA_SP` bodies' seven, and -- read off the preprocessed body -- **there is no barrier at the tile
boundary at all**. So in that body:

* `finalize_O(t)` (all six warps: SMEM reads + 4,096 GMEM stores) runs concurrently with tile
  `t+1`'s Q/K move-in DMA, its 320 SF-SRAM scale words, **and the QK matmul together with its
  accumulator -> SMEM move-out**;
* the PVF prefetch's V move-in DMA runs concurrently with requant and pack.

That is *more* mesh-versus-SIMT overlap at the tile boundary than `FA_SP_QSPLIT` has, not less.
It is still worth phase-testing (it is a completely different schedule, and it is verified correct
unperturbed), but it is not the serialized baseline, and `FA_ST_NOOVL` is closer to one.

## Suggested approach

**De-overlap deliberately, then bisect.** The hazard is a race, so serializing should close it:
the non-pipelined `FULL_ATTN2` path, barriers between every stage, one mesh op in flight, no
`(i+1)` prefetch, no `(i-1)` overlap. Barriers are cheap -- `mu_barrier` is 3 cycles. If a fully
serialized kernel is phase-robust, that gives both a correct baseline **and** a bisection handle:
re-introduce overlap one stage at a time until robustness breaks, which localizes the racing pair
far better than three days of flag A/B has managed.

## Building, running, verifying

**`FA_NT<n>` HAD A SILENT HOLE, NOW A BUILD ERROR.** `FA_NT12/16/24/36/72` were added to `FA_SPTILES`
(the `FA_SP` body) but **not** to `FA_NTILES` (the sequential `FULL_ATTN2 FA_STEADY` body), so
`FULL_ATTN2 FA_STEADY FA_NT24` fell through to the default and ran **four** tiles -- normal `$finish`,
no warning, and a perfectly clean 4-tile result that would have been quoted as a 24-tile pass.
Measured: `stS24` stopped at 428,097 cycles with 4 tile tops. The five cases are now present, and an
`FA_NT<n>` that falls through to the default is now `#error` -- verified by temporarily disabling the
`FA_NT24` case and confirming the build fails.

**`make kernel.s` / `make flash_attention_mx.s` silently does not rebuild** (broken rule, rc=2; a
byte-identical file size is the only tell, and a three-week-old assembly was nearly reported as a
fresh result on the sibling track). To read generated code, invoke the compiler directly with
`kernel-build-env.sh` sourced -- `clang++ -S` (or `-E -P` for the preprocessed body, which is what the
structural claims in this file were verified with) -- and then confirm the output contains a
shape-specific symbol before trusting it.

Otherwise identical to the sibling -- see [`../flash_attention_mx/README.md`](../flash_attention_mx/README.md)
for the build, the config flags, and the **verification traps**: the only sound scorer is
`fa_verify_tiles.py`; use `golden_O_u16.npy`; `TIMEOUT_CYCLES=N` yields only N/2 cycles; run
`fa_regs3.py` (not `fa_regs.py`) before every sim; hash the **RV32 segments**, never `.text`; and
`volatile` on a local means DRAM here. Every one of those exists because it produced a
false-confident number.
