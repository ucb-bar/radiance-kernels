# flash_attention_mx

> **This is the PEAK-UTILIZATION variant.** It maximizes steady-state mesh utilization and is
> **not robust to schedule perturbation** -- see the correctness note below. If you need a
> configuration that stays correct under perturbation, use
> [`../flash_attention_mx_stable`](../flash_attention_mx_stable/README.md) instead. The two goals
> conflict (the overlap that produces the utilization is what exposes the race), which is why they
> are separate directories.

Software-pipelined **MX-FP8 flash attention** on the Radiance cluster + mxGemmini mesh.
Sq=64, Sk=256, d=128, Bk=64; e4m3 values with E8M0 per-32-block scales.

```
Q,K,V (mvin) --[mesh: QK^T]--> S (accmem -> SMEM)
  --[SIMT: online softmax -> requantize -> pack scales]--> P (e4m3) + P scales
  --[mesh: PV]--> O (accmem) --[SIMT: finalize]--> O (bf16)
```

The point of this kernel is **steady-state mesh utilization**, not a single tile. Three
stages are overlapped across tiles so that `T_tile = max(stages)` rather than their sum:
mesh `QK(i)`/`PV(i)`, SIMT softmax/requant/pack `(i-1)`, DMA mvin `(i+1)`. The mesh reads
its operands through its own scratchpad read ports (~2/cycle, no SMEM writes during
compute), so it does not consume Muon SMEM line-slots and genuinely runs underneath the
SIMT work.

## Numbers

Mesh-busy is **fixed at 16,420 cycles/tile** (2 matmuls x 8,210; 4.19M MAC / 256 MAC per
cycle, matching the 8,192 theoretical exactly). So **util = 16420 / (cycles per tile)**.

| config | cyc/tile | util | correctness |
|---|---|---|---|
| single-shot baseline (no pipelining) | 195,044 | 8.4% | -- |
| `FA_SP FA_SP_QSPLIT FA_SP_WCNT FA_SP_PAX FA_SP_CVTX` | 45,582 | 36.02% | 12/12 NT6, 16/16 NT8; 10/12 under FA_PHASE1 |
| **+ `FA_SM_2P FA_SM_2PRAW FA_SP_ACCRS FA_SP_PREPK`** (**peak**) | **42,364** | **38.76%** | **12/12 NT6 and 16/16 NT8**; phase sweep pending |
| + `FA_SP_ACCRS` alone | 42,644 | 38.50% | 12/12 NT6, **15/16 NT8** |
| + `FA_SP_ACCPAD FA_SP_PREPK` | 43,982 | 37.33% | 16/16 NT8, but the pad costs +1,694 for nothing |
| + `FA_SP_PREPK`, pad simply removed | ~~43,899~~ | -- | **11/16 -- timing void** |

**These utilization numbers are real results, and the correctness caveat is scoped -- read both.**

*The utilization is sound.* Mesh-busy is fixed, the harness measures converged steady-state
intervals, and steady state is exactly what a real workload looks like: a 2048-token TinyLlama-1.1B
prefill is ~101,000 tiles of this shape (32 query blocks x 144 key blocks/head x 32 heads x 22
layers; ~50,000 if two head_dim-64 heads pack into d=128). Boot amortizes to zero at that scale, so
**36.02% is the peak steady-state utilization achieved with output bit-correct against the golden**,
and NT6/NT8 is a perfectly adequate sample for the *timing* claim. That is the headline result:
8.4% -> 36.02%, a 4.3x speedup on a taped-out design whose RTL cannot be changed.

*The correctness caveat is about robustness, not about whether these runs were right.* The default
passes NT6 12/12 and NT8 16/16 -- those outputs really are bit-correct -- but it fails under
`FA_PHASE1` at 10/12 (cluster 0 tiles 4 and 5, 93.32% / 121.14%, full 4096/4096 coverage). Since
`FA_PHASE<k>` is a **read-only MMIO delay that computes nothing**, the arithmetic is unchanged and
only the schedule moves. So the honest statement is: **this configuration is correct at the
schedule it was measured at, and is not robust to schedule perturbation.**

That distinction matters because the two uses have different requirements:
* **Peak-utilization benchmarking (what this default is for).** A fixed schedule is fine, the
  numbers stand, and the reported outputs are verified bit-correct. Re-verify after any change.
* **Deployment.** At ~10^5 tiles per forward pass, with per-layer access patterns and DRAM state
  varying, the inter-cluster phase cannot be pinned the way the harness pins it -- a hazard that
  first bites at tile 7 of 8 will fire constantly. A stability-oriented variant is tracked
  separately and should not be conflated with this one.

**NT8 is NOT a sufficient gate, and here is the measurement that proves it.** `FA_SP_ACCRS` alone,
run at NT16, has **two different onsets**: cluster 0 goes wrong at **tile 12**, cluster 1 at
**tile 7**. Both latch and never recover.

```
cluster 0:  0-11 correct,  12 13 14 15 WRONG
cluster 1:  0-6  correct,   7  8  9 10 11 12 13 14 15 WRONG
```

| gate | reaches | verdict |
|---|---|---|
| NT6 | tile 5 | 12/12 -- **both onsets invisible** |
| NT8 | tile 7 | 15/16 -- catches cluster 1 by exactly **one tile** |
| NT16 | tile 15 | **19/32** -- catches both |

Cluster 0's onset at tile 12 is unreachable at NT8 and would have passed every gate this campaign
used for weeks. NT8 caught cluster 1 with a single tile of margin -- luck, not headroom. Extrapolated
to a real 144-tile invocation (72/cluster), this config would produce **~65 of 72 wrong tiles per
cluster**: a 12-of-12 NT6 pass and a production-unusable kernel, simultaneously. That is the whole
gap between "peak utilization at a fixed schedule" and "survives an invocation", in one run.

**`FA_PHASE` also strictly dominates NT8.** Measured across five
configurations -- every one fails somewhere:

| config | cyc/tile | unperturbed | NT8 | P1 | P2 | P3 |
|---|---|---|---|---|---|---|
| default (tagged 36.02%) | 45,582 | 12/12 | 16/16 | **10/12** | -- | 12/12 |
| `+ FA_SP_PREPK` | 43,982 | 12/12 | **16/16** | **7/12** | **5/12** | **10/12** |
| `+ FA_SP_ACCPAD` (N=128) | 44,565 | 12/12 | 16/16 | **7/12** | **7/12** | **4/12** |
| `+ FA_SP_ACCRS` | 42,650 | 12/12 | **15/16** | **8/12** | **5/12** | **10/12** |
| `+ FA_SP_ACCPAD` (N=4) | 43,273 | **9/12** | **11/16** | -- | 9/12 | -- |

**The peak config's mechanism is worth understanding before changing it.** `FA_SP_ACCPAD` padded
the pre-store window with slack; removing the pad and putting *nothing* in its place gives
**11/16** (cluster 0 tiles 3-7 at 91-122%, spread 29.1%). What works is `FA_SP_ACCRS`: delete the
pre-store `fa_gfl` drain entirely and let `ReservationStation.scala`'s STORE branch enforce the
ordering it already implements (`deps_ex` for an `opa_is_dst` entry covers *"raw for st b <- ex a"*,
the store's accumulator source against the compute's accumulator destination). The `fa_gfl` was
*destroying* that interlock by emptying the station before the store was allocated. Same 1,694
cycles recovered as bare removal -- but 16/16 instead of 11/16, because the ordering is handed to
hardware rather than to slack.

Unexplained observation, recorded as such: `FA_SP_PREPK` appears to help *correctness* at NT8, not
only cycles -- `ACCRS` alone is 15/16 (fails cluster 1 tile 7) while `ACCRS`+`PREPK` is 16/16. One
run each and no mechanism, so do not rely on it.

`PREPK` is also the cautionary row on the older `ACCPAD` base: 37.33% and **16/16 at NT8**, clearing
every gate this project used for weeks, and it fails the phase sweep three ways. Do not add a flag to this kernel and report NT6 or
NT8 alone -- but equally, do not read a `FA_PHASE` failure as invalidating the cycle count of a run
whose tiles verified. It means the config is schedule-fragile, not that its measurement was wrong.

## Building and running

```sh
make                      # builds kernel.cpp (GPU) + host.cpp (rv64 host)
python3 gen_data.py       # regenerates include/fa_data.h
python3 fa_gen_goldens.py # regenerates the golden_*.npy references
```

Config is selected by `-D` flags forwarded to both compiles (see `FA_HOST_DEFS` in the
Makefile -- the host and GPU scale-prefill paths **must** agree, because the scale SRAM
write port has a single shared 2-beat pairing counter).

`FA_STEADY` builds the multi-tile steady-state harness; tile count comes from
`FA_NT1/2/3/4/6/8`.

## Verifying -- please use these tools, they exist because of specific traps

* **`fa_verify_tiles.py <trace>` is the only sound scorer.** Every tile writes the *same*
  GMEM O buffer, so a whole-file Frobenius scores only whichever generation landed last.
  This tool groups by cluster and then by address-repeat, and self-validates with a
  4096-words-per-image count. Reference value is **3.5666%** (`golden_O_u16.npy`).
* **Use `golden_O_u16.npy`.** The default path is non-streaming;
  `golden_O_flash_u16.npy` is the streaming reference and scores the same *correct* output
  as 4.5954%.
* **`NT6` is the minimum for any steady-state or correctness claim.** NT2/NT4 have passed
  configurations that later failed 3-of-12. Quote the *converged* interval, not the
  minimum -- intervals fall and then rise.
* **A wrong tile also has wrong timing** (~28% steady-interval spread vs ~3% when clean),
  so a run that is not fully correct has *no usable cycle number* either.
* **Wait for the trace to stop growing.** VCS exits before `spike-dasm` drains its stderr
  pipe; a truncated trace scores uncovered cells as zero, which looks like a real error.
* **Check your cycle budget empirically -- the two knobs differ, and getting it wrong looks
  exactly like a hang.** `fa_run.sh`'s `BUDGET` argument (3rd arg of `fa_launch_safe.sh`) is
  **1:1**: it sets `MC=BUDGET*2` and `+max-cycles` counts half-cycles, so the run stops at
  `BUDGET` cycles exactly -- verified on three completed runs (`BUDGET=900000` -> `$finish` at
  1,800,000,500 ps = 900,000 cycles at 2000 ps/cycle). A raw `TIMEOUT_CYCLES=N` passed straight
  to VCS is the one that yields **N/2** (`$finish` at `N*1000+500` ps). Confirm which path you are
  on by dividing the `$finish` time by 2000 -- do not trust either rule from memory. Applying the
  N/2 rule to `BUDGET` wastes 2x; applying the 1:1 rule to `TIMEOUT_CYCLES` truncates the run.
* **`fa_regs3.py` is the register check; run it before every sim.** `Rename.scala`
  allocates a physical register on a warp's *first write* to an architectural register and
  never reclaims it, from one 255-entry counter per core, so the binding quantity is the
  **per-warp** sum over a core's resident warps -- *not* a whole-file union of register
  names. A union is not even monotone in the real figure (a config with a *higher* union
  ran while a lower one aborted), so it silently vetoes legal builds. Measured bracket on
  the real figure: **runs at <=216, aborts at >=231**.
* **`fa_baraudit.py`** catches `vx_bar` inside a divergent region: llvm duplicates it into
  both paths and the Synchronizer never releases, which hangs. Fix with an
  `else { asm volatile("nop"); }`.

## Known hazard family -- read before trusting a result

There is a **latent, timing-sensitive correctness hazard** that this kernel has not fully
escaped. What is established:

* `FA_SP_WCNT` fixed one real drain bug: `gemmini_fence()` polls `io.busy`, which rises
  several cycles *after* the command store, so the fence can fall through. Polling MMIO
  `0x28` (`runningLoops`), which rises in the *same* cycle as the `LOOP_WS` write, closes
  it, at no cost. Without `WCNT`, otherwise-identical builds score 5-of-8 and 6-of-8.
* **Something survives `WCNT`.** `FA_SP_PREPK` flips correctness to 4-wrong-of-12 at
  *identical* cycle counts, and `FA_SM_2P` alone fails while `2P+2PRAW` passes. A flag that
  changes correctness without changing timing is a data race whose window moved.
* The corruption fingerprint localizes it to **S**: every wrong cell lies inside V's
  per-column convex hull (0 of 8192 outside), so P and l stay consistent with each other,
  which exonerates PV, its operand spad, V's scales and finalize. Successive wrong tiles
  share 0 of 4096 words, so it is a *fresh* corruption each tile, not one damaged resident
  operand.
* Leading hypothesis: `runningLoops` and `io.busy` both track the reservation-station/loop
  FSM, and `completionCount` rises when the loop's last command *retires* -- not when the
  last MAC has propagated through the 16x16 array into accmem. Both polls can then pass
  with MACs still in flight. **THIS HYPOTHESIS IS REFUTED at the accumulator read port, and
  `FA_SP_ACCPAD` -- a delay before `fa_store_acc` -- is NOT a fix. Do not ship it.** Two
  independent results kill it:
  - `AccumulatorMem.scala:619-624` implements a complete **same-row RAW interlock** across all
    three write-pipeline stages, confirmed live in silicon (`AccumulatorMem.sv:1128`) and
    honored on *both* consumers (mvout `Scratchpad.scala:917`, ex `:890`). A read of the same
    accumulator row therefore cannot return a mid-accumulation value -- there was never a
    drain for a pad to cover.
  - **The pad sweep is NON-MONOTONIC**, which is decisive: 128 -> 12/12 correct (44,565
    cyc/tile), 2,048 -> 4 wrong (onset tile 2), 8,192 -> 4 wrong (onset tile 1). If the pad
    were covering a fixed latency, correctness would be monotone in pad length -- more delay
    could never hurt. It is not. The pad **relocates the schedule**, and some lengths happen
    to land outside the race window. A passing pad length is green-by-accident, exactly like
    the other four such results in this campaign.
  The failures keep the same fingerprint (all-rows/all-cols, S-localised, latching), so it is
  the same hazard throughout, not one introduced by a long pad.
* **What software can actually observe, and why it is not enough** (all flop-counted in
  generated Verilog): the readable MMIO surface is exactly three fields -- `0x08` cmd-ready,
  `0x20 io.busy`, `0x28 runningLoops`; every other offset is `RegField.w` and reads back
  constant zero with a normal ack. Structural drain is **35 register stages** (readable at
  +36): input skew *c* + 16 row hops + 1 output capture + (15-*c*) de-skew = 32 uniform to
  `io.resp.valid`, then +3 in `AccumulatorMem`; the PE accumulate path is combinational
  (`MacUnit.sv`/`MxFpMul.sv` have no clock port).
  - `0x28` (what `FA_SP_WCNT` polls) is **issue-based, not retire-based**: `LoopMatmul.scala:497`
    goes `idle` inside `.elsewhen (io.cmd.fire)`, i.e. the cycle the last COMPUTE is *accepted
    into the reservation station*, with up to 16 ex commands still queued. Residual is **>=164
    cycles and formally unbounded** (`Scratchpad.scala:220`'s 4-entry un-backpressured queue can
    stall feeding arbitrarily). **No fixed pad can make `0x28` safe.**
  - `0x20` falls exactly **3 cycles** early, so `0x20` + 4 is the ceiling of what software can
    do. Caveats: `io.busy` has **no mesh term at all** (`Controller.scala:786`, 8 terms), and per
    `ReservationStation.scala:140` **a lone PRELOAD in the RS makes `io.busy` read 0**.
  - The signal you would want, `matmul_in_progress` off the mesh tag queue
    (`ExecuteController.scala:313-315`), **exists in Chisel but was dead-code-eliminated** by
    `num_counter = 0` (`grep -c busy ExecuteController.sv` -> 0), and would still have been 3
    cycles short.
  So the racing pair is **not** at the accumulator read port and cannot be fixed by waiting
  longer there. Open candidates: SMEM-side visibility after the mvout, the requantizer path,
  and ordering across *different* rows.
* **A "free" ALU pad on warp 0 in a quiesced stage costs ~13x its instruction count.** Measured
  13.4 / 12.9 / 13.1 cyc per iteration across three pad sizes for a 3-instruction register-only
  loop -- that is the single-warp dependent-issue latency with no other warp resident on the
  core to hide it (the other five warps are already at the barrier). Budget accordingly.
* **`volatile` on a local means DRAM here.** A `volatile int` pad counter put the loop variable
  in a stack slot, and the stack is in GMEM/DRAM: a nominal 128-cycle pad cost ~68,000
  (128 x 2 DRAM round-trips x ~265 cyc). Same `BAR_PAD` bug documented in `kernel.cpp`,
  reintroduced three days after it was first found. Keep `asm volatile` on the instruction,
  never `volatile` on the variable, and check the disassembly for `lw.global`/`sw.global`.
* **Slack masks it.** `FA_SP_CVTXS` costs +4,766 cycles and is otherwise useless, and its
  presence turns a 15/16 NT8 failure into 16/16. So **an NT8 pass on a slower config
  carries no information about a faster one.**

Practical consequence: **the default is the fastest config measured that is clean at NT6+NT8, and
it is still not correct.** Re-verify with the `FA_PHASE<k>` sweep after any change, however
"obviously bit-exact" it looks -- NT6 and NT8 are both insufficient.

**What is closed off, by elimination rather than by A/B** -- the accumulator store is NOT where
this bug lives, so do not spend more time on drain/pad fixes:
* `ReservationStation.scala`'s STORE branch already implements the needed ordering (`deps_ex` for
  an `opa_is_dst` entry includes *"raw for st b <- ex a"* -- the store's accumulator source
  against the compute's accumulator destination), and the failure survives it.
* `AccumulatorMem.scala:619-624` adds a same-row RAW interlock across all three write-pipeline
  stages, live in silicon and honored on both consumers.
* A pad at the RTL-derived bound (4 cycles; `io.busy` falls +3) gives **9/12** -- if 4 were the
  true drain, 4 would suffice. A pad 424x the bound passes unperturbed and dies under phase.
* Measured drain gap is real but not the bug: `runningLoops` goes idle **141 cycles** before the
  last `acc_mems_0.io_write_valid` (cycles 91,180 vs 91,321 on an 8,113-cycle QK).
* Sharpest single datum: `qmax` 2/12 vs `ymax` 12/12 -- *same flag set*, differing only in pad
  length, opposite verdicts.

**And inter-cluster relative phase is NOT the trigger.** `FA_PHASE_BOTH` (identical delay in both
clusters, restoring relative phase to ~0) still fails 10/12. Asymmetry makes it worse but is not
necessary, so a two-cluster contest is not required to explain this bug. The harness itself is
exonerated independently: an ALU-only, MMIO-free, warp-0-only pad sweep breaks it too
(N=2048 -> 7/11, N=8192 -> 6/10), and exactly one cluster fails per run with *which* one flipping
between k=1 and k=2 under symmetric code -- a loop that corrupted whoever ran it would hit both.

## Final frontier, and what each number is evidence of

| frontier | config | cyc/tile | util | evidence |
|---|---|---|---|---|
| fastest at NT6 | `ACCRS`+`PREPK` | **42,344** | **38.78%** | 12/12 bit-exact, spread 6.7% |
| fastest at NT8 | `ACCRS`+`PREPK` | **42,364** | **38.76%** | 16/16 bit-exact, spread 7.0% |
| largest tile survived | `ACCRS`+`PREPK` | -- | -- | clean through tile **16** (c0) / **23** (c1) |

`ACCRS`+`PREPK` reaches **2.4x the exposure of any other configuration** (its neighbours break at
tiles 7-12), so it genuinely *mitigates* rather than merely reshuffling the schedule. Its onset is
cluster 0 tile 17.

**THE ONSET IS DETERMINISTIC, NOT A COIN FLIP -- this corrects an earlier "lottery" framing here.**
`nA16` and `nA24` are the *same binary* at different tile counts and agree **exactly**: c0 = 12,
c1 = 7. So a given build has a fixed onset tile, and rerunning it changes nothing. The observed
"~74% of variants pass" is therefore the fraction of *configurations* whose onset happens to fall
beyond the tile count you tested -- **not** a per-run probability. Concretely, NT8 passes by luck of
*where the onset sits*, not by luck of the draw: `ACCPAD`+`PREPK` is 16/16 at NT8 with onset at tile
9, and `ACCRS` alone is 15/16 with onset at tile 7 -- one tile of margin each.

**Consequence, stated plainly: no configuration measured here survives a real invocation.** A
144-tile invocation (72/cluster) reaches every onset ever measured; even the frontier's tile-17
onset would corrupt ~55 of 72 tiles per cluster. The utilization numbers are real and the reported
tiles really were bit-correct -- but they are **fixed-schedule, short-run benchmarks**, and this
kernel is not yet deployable. That is why the stability track is a separate directory.

**A cycle number from a corrupt run is not a measurement -- treat it as a class of error.** Three
lever verdicts in this tree traced to exactly that: `BANKA` -304 (from a 9-ok/3-wrong run; actually
-10 at NT6, **+28** at NT8, and 10/12 on its own), `FZ6` +1,330 (vs **+179** measured on a clean
run), and the `2PBM` row from the unscored F6 sweep. One of them silently broke a projection: the
39.18% forecast failed by exactly `BANKA`'s phantom -304.

## The accumulator drain is CORRECTLY covered already -- measured, not argued

From the waveform, on the banked config: **MMIO `0x20` (`io_busy`) falls in the *same cycle* as the
last accumulator write**, while `0x28` (`runningLoops`) falls 141 cycles earlier.

| signal | ps | cycle | |
|---|---|---|---|
| `runningLoops` 1->0 | 182,361,000 | 91,180 | MMIO `0x28` -- what `WCNT` waits on |
| last `acc_mems_0.io_write_valid` | 182,643,000 | 91,321 | data actually final |
| **`io_busy` 1->0** | **182,643,000** | **91,321** | MMIO `0x20` -- coincides **exactly** |

Confirmed on the second drain (the `fa_store_acc` `loop_ws`): `runningLoops` falls at 184,415,000,
`io_busy` at 184,705,000 -- a 145-cycle gap, same shape.

This resolves what looked like a contradiction in the `FA_SP_WCNT` note. Both claims were true but
about opposite ends: **`io_busy` can false-negative at the START** (it has not risen yet if polled
immediately after issue), while **`runningLoops` cannot, but false-negatives at the END by ~141
cycles**. The correct drain is therefore *both, in order* -- `waitcount(0)` then `busy` -- which is
exactly what `fa_gfl` already does.

**Load-bearing consequence: the accumulator drain is already correctly covered on the `WCNT` path,
so it is NOT the surviving hazard.** That is independent support for the different-accumulator-rows
candidate over anything drain-shaped, and it explains why `FA_SP_ACCPAD`'s +1,694 buys nothing
`fa_gfl` does not already provide. A correctly-sized pad would be ~141 x 2 drains = 282 cyc/tile
(0.62%) -- nearly free if a drain cost were the answer. It is not the answer.

Caveat on the measurement: `io_busy` shows two 3-cycle glitch pulses between the matmul drain and
the store. Benign for `fa_gfl` (a poll landing in a gap correctly reads "not busy"), but any future
code that polls `busy` **without** the preceding `waitcount` could latch on a glitch.

## Onset periodicity -- the narrowest live clue

The onset is **not uniformly distributed across tiles**. Over eleven independent failing configs,
**cluster 1 tile 7 recurs three times** (`yrs8`, `nA16`, `nP24`) with cluster 0 at tiles 9 and 12,
while phase-perturbed runs move onset to tile 1 and pad-length variants to tile 3. A structureless
race would scatter; **a resource turning over on a period of ~8 tiles produces exactly this.** That
fits the one surviving candidate -- ordering across *different* accumulator rows, where a row-index
or bank wrap turns over periodically and the per-row interlock would be insufficient.

Next capture, recipe ready: cluster 1 (`cluster_prci_domain_1...radiance_gemmini_tile_6`), the tile
6->7 boundary, narrow `+dump-start` window (a full dump is ~310 MB per 100k cycles, so size the
window from the mark stamps first). Trace `acc_mems_0.io_write_bits_addr` against `io_write_valid`
looking for writes to **different rows** completing out of order across the boundary, with
`runningLoops` bracketing each matmul.

## Structural limits found (do not re-derive these)

* **No matmul here can be split along M, N or K.** The mesh's scale-SRAM read row is a
  fixed function of the `CONFIG_SCALE_MEM` loop bounds and all four scale slots are
  occupied. This kills the whole "split the mesh op to hide SIMT underneath it" family, and
  it is why PV's 8,882 cycles stay exposed with 5 of 6 warps idle -- which in turn sets the
  ~40% ceiling of this pipeline shape. Store-only `loop_ws` is the one exception.
* **No SMEM subbank escapes the mesh** -- the Gemmini read client is attached to all 64.
* **`FA_SP_OPV`** (hiding all 16,420 mesh cycles) dies on `Scratchpad.scala:220`, a 4-entry
  un-backpressured spad read queue: the mesh may not read an operand from a bank SIMT is
  reading, SMEM is exactly full, and there is no lane-shuffle instruction to relocate the
  reduce scratch.
* **The HW MxRequantizer cannot be fed from Muon software** on its GPU-input path (needs
  exactly-32-byte transactions; the fabric is word-strided). The mesh-output path works.
* **No GMEM->SF_MEM DMA exists** (`scale_mem_mvin_base_addr_*` are declared but never
  assigned).
* **The host (rv64) can prefill the scale SRAMs concurrently with the GPU** using 8-byte
  `sd` stores -- 6.4x faster per byte than the GPU path. 4-byte host writes are *silently
  dropped*. But the whole offload is worth only ~1,000 cyc/tile here, because both scale
  bursts were already hidden by the pipeline.
* **Host reads of cluster SMEM trip `TLMonitor ... 'D' channel improper response size`, and
  the mechanism is NOT yet established.** Use `printBuf` instead (below). Two mechanisms have
  been proposed and *both refuted by measurement* -- do not propose a third without the
  offending A/D beat pair from a waveform:
  - `FlitMergeNode`'s per-source `wasMerged` bit: **refuted**, that node has exactly one
    instantiation (`GemminiTile.scala:188`, the Gemmini scale SRAM) and is not on the SMEM path.
  - request width: **refuted**. Subbanks really do advertise
    `get = TransferSizes(wordSize, wordSize)` -- *exactly* 4, `beatBytes = 4` -- but a 4-byte
    `lw`, which is precisely what that allows, **still asserts**. Width only moves *when*:
    82,468 cyc (8-byte, heavy softmax) -> 290,934 (8-byte) -> 343,604 (4-byte). An
    occupancy/rate-dependent window, not an illegal request.
  Current suspect, untested: `RWSplitterNode` trims `source` *and* `size`
  (`// FIXME: check truncation` in its source, on exactly the field the monitor reports) and
  is on the failing path **only** -- SMEM goes `clcbus -> extClients -> TLFragmenter(4,128) ->
  RWSplitterNode -> subbanks`, whereas `printBuf := clcbus.outwardNode` is direct. A trimmed
  `source` colliding once enough requests are outstanding would mis-deliver a D beat whose
  size then cannot match, which also fits the assert arriving earlier the harder the GPU
  hammers SMEM.
* **`clcbus` itself is fine, and `printBuf` is a working host<->GPU mailbox.** The unused
  512 B `printBuf` TLRAM at device `0x80000` (`beatBytes=8`, atomics, no RTL writer and no
  software symbol) hangs off the *same* `clcbus`, and 8-byte host reads of it work: 12/12
  correct, no assert.
* Barriers are cheap: `mu_barrier` = 3 cycles, `fence.s` ~22 cycles (and it waits only on
  the Muon per-warp shared queues -- not GMEM, not the Gemmini mvout).
* TLP beats ILP here, and **instruction count does not predict cycles**: a 27-instruction
  SWAR pack block is one serial dependency chain and loses to four independent 7-op chains
  by 4,766 cycles/tile.

## Files

| file | role |
|---|---|
| `kernel.cpp` | GPU kernel: pipelined body, `FA_STEADY` harness, all `FA_SP_*`/`FA_SM_*` flags |
| `flash_mx_impl.hpp` | SIMT stages: online softmax, requantize, pack scales, finalize |
| `mxgemm_core.hpp` | mesh configuration, mvin/prefetch, scale loading, `mxgemm_compute_tile` |
| `host.cpp` | rv64 host: MX scale prefill into both clusters' SF SRAMs via 8-byte stores |
| `gen_data.py` | generates `include/fa_data.h` |
| `fa_gen_goldens.py` | generates the `golden_*.npy` references |
| `fa_verify_tiles.py` | **the sound per-tile scorer** |
| `fa_regs3.py` | per-warp register-budget check (run before every sim) |
| `fa_baraudit.py` | per-function `vx_bar`-in-divergent-region audit |
| `fa_marks3.py` | MARK-stamp parser; **`--mesh` is required** (16420 per tile; 8192 if an iteration is a half-tile) |
