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
  station* -- so no fixed pad can make it safe. `matmul_in_progress` off the mesh tag queue
  (`ExecuteController.scala:313-315`) is the signal one would want, and it was dead-code
  eliminated by `num_counter = 0`.

## Corruption fingerprint

Localized to **S**: every wrong cell lies inside V's per-column convex hull (0 of 8192 outside),
so P and l stay mutually consistent -- which exonerates PV, its operand spad, V's scales and
finalize. Successive wrong tiles share **0 of 4096 words**, so each tile is freshly corrupted
rather than inheriting one damaged resident operand. Values latch around 93-121% of golden.

## Live candidates for the real mechanism

None tested yet. In rough order of suspicion:

1. **SMEM-side visibility after the mvout** -- the accumulator read port is exonerated, but what a
   later SMEM read observes is not.
2. **Ordering across *different* accumulator rows** -- the interlock above is per-row only.
3. The **requantizer** path.
4. An intra-cluster race decided by **GMEM-return timing** through the 4-entry un-backpressured
   spad read queue at `Scratchpad.scala:220`.

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
