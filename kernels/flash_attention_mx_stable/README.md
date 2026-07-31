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

**No configuration has passed this yet.** Status is tracked below rather than claimed.

## Status

| gate | best known |
|---|---|
| NT6 (12 images) | several configs, 12/12 |
| NT8 (16 images) | `FA_SP_ACCPAD`+`FA_SP_PREPK`, 16/16 |
| NT72 (~1 TinyLlama head) | **not yet measured** (~3.3M cycles, ~10 h sim) |
| `FA_PHASE1/2/3` | **none** -- every config measured fails at least one k |

Everything below was measured with the scripts and tools in this directory; the run set is
reconstructible from disk with **`./fa_runtable.sh /tmp/struns`** (tag, whether the sim reached
`$finish`, correct/wrong tile-image counts, and the flag set + RV32-segment sha recorded at build
time), and any single trace can be re-diagnosed with `fa_verify_tiles.py` (the verdict) or
`fa_rowdiag.py` (which stage). `/tmp/st_build.sh <tag> "<DEFINES>"` and
`/tmp/st_go.sh <tag> <elftag> <seed> <budget_cycles> 16420 <marks_per_tile>` build and launch in
**this** directory only. Marks per tile: 7 for the `FA_SP` bodies, **8** with `FA_ST_NOOVL` (it adds
a stage), 11 for the sequential `FULL_ATTN2 FA_STEADY` body.

### The de-overlap plan has a prerequisite nobody knew about: the un-overlapped path is BROKEN

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

**It does not compute the right answer, at tile 0, deterministically.** Measured (2026-07-31,
seed 12345, `fa_verify_tiles.py` against `golden_O_u16.npy`), on four builds -- no skew,
`FA_PHASE1`, `FA_PHASE2`, and `FA_NT8`:

| build | cluster 0 tile 0 | cluster 1 tile 0 |
|---|---|---|
| `stB6` (no skew) | 153.879%, 24 NaN rows (4-15, 20-31) | 79.026%, 1 NaN row (31) |
| `stB6p1` (`FA_PHASE1`) | **153.879%, same 24 rows** | **79.026%, same row** |
| `stB6p2` (`FA_PHASE2`) | **153.879%** | **79.026%** |
| `stB8` (`FA_NT8`) | **153.879%** | **79.026%** |

Bit-identical across all four, including across a 2.4k- and 4.8k-cycle skew of cluster 1. So this
is **not** the race -- it is a functional defect in the un-overlapped code path, and it is
*separate from* the hazard this directory exists to chase. Two further facts that narrow it:

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

**Consequence for the plan:** the de-overlap ladder has to be walked *upward* from a working
configuration rather than reached by clearing flags, and the first job is to find which flag the
un-overlapped path needs in order to compute correctly at all. That is an `FA_NT2` question (the
defect is at tile 0), i.e. ~20 minutes per point rather than ~75.

### The upward ladder, at NT2 (the defect is at tile 0, so NT2 is enough)

All on `FULL_ATTN2 FA_SP FA_SP_WCNT`, seed 12345, cluster 0 / cluster 1 tile 0:

| rung | added | cl0 t0 | cl1 t0 |
|---|---|---|---|
| `stN0` | -- | 153.879%, 24 NaN | 79.026%, 1 NaN |
| `stN1` | `PKOVL` | **153.879%, same 24** | **79.026%, same 1** |
| `stN2` | `+ QOVL` | **153.879%, same 24** | **79.026%, same 1** |
| `stN3` | `+ QKACC` | same NaN rows | same |
| `stN4` | `+ QSPLIT` | **153.879%, same 24** | **79.026%, same 1** |
| `stN5` | `+ LEANCFG PAX CVTX` (= the 36% config) | *in flight* | *in flight* |

So `PKOVL`, `QOVL`, `QKACC` and `QSPLIT` -- the four overlap flags -- are **not** what makes the
kernel compute correctly. The only remaining difference from the verified-correct configuration is
`FA_SP_LEANCFG` + `FA_SP_PAX` + `FA_SP_CVTX`, and `PAX`/`CVTX` are bit-exact by construction (both
are XOR permutations of a load/store index; `PAX`'s reduction is an order-independent `fmax`). That
points at **`FA_SP_LEANCFG`**, i.e. at `configure_mxgemmini` being *present* in the steady-state
loop -- which would make `LEANCFG` a **correctness** flag, not the performance flag it is
documented as. `stN6` (`LEANCFG` only) and `stN7` (`PAX`+`CVTX` only) split it; both in flight.

`fa_rowdiag.py` in this directory is the tool these rows came from.

### FA_ST_CFGPRE is REFUTED -- by its own experiment

The `CONFIG_SCALE_MEM`-ordering mechanism below predicts the whole fingerprint and has a netlist
defect behind it, and it is **still not a sufficient fix**. Measured on the 36% config at NT6:

| build | flags added | verdict |
|---|---|---|
| `stE6p2` | `FA_ST_CFGPRE` + `FA_PHASE2` | **cluster 1 tile 1 = 119.367% WRONG** |

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

Identical to the sibling -- see [`../flash_attention_mx/README.md`](../flash_attention_mx/README.md)
for the build, the config flags, and the **verification traps**: the only sound scorer is
`fa_verify_tiles.py`; use `golden_O_u16.npy`; `TIMEOUT_CYCLES=N` yields only N/2 cycles; run
`fa_regs3.py` (not `fa_regs.py`) before every sim; hash the **RV32 segments**, never `.text`; and
`volatile` on a local means DRAM here. Every one of those exists because it produced a
false-confident number.
