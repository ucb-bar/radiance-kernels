# fa_mx_hostsf — Rocket-host offload sandbox for the MX-FP8 flash-attention kernel

`flash_attention_mx.cpp` and `flash_mx_impl.hpp` here are a **frozen snapshot** of the main
`kernels/flash_attention_mx/` versions (taken 2026-07-26).  Other agents edit those two files
continuously and the build scripts inject `#define`s into the source in place, so two builds made
30 minutes apart in the main directory are not necessarily the same kernel.  Freezing them is what
makes the A/B tables below apples-to-apples.  The only edit ever made to the snapshot is the inert
`#ifndef FA_NOPACK` guard around `pack_scales_to_sfmem` (see the hazard section).

`host.cpp` and `mxgemm_core.hpp` are **not** frozen: `hs_build.sh` copies them from
`kernels/flash_attention_mx/` on every invocation, so the main directory stays the single source of
truth for the two files this campaign owns.

## Harness

    bash hs_build.sh <TAG> "FULL_ATTN2 FA_STEADY FA_NT4 FA_NOSCALES FA_HOSTHS FA_HOSTCFG"
    bash hs_run.sh   <TAG> 700000 [SEED]        # -> /tmp/hsruns/<TAG>.{out,log}
    python3 fa_marks_extract.py <TAG>           # marks, per-tile deltas, steady slope, CPROF
    python3 fa_verify_tiles.py  <TAG>           # PER-TILE correctness (see below)

Rules that cost time to learn:

* `+max-cycles=N` gives **N/2 core cycles**; too small looks exactly like a hang.
* `+dramsim` shifts totals by ~35k cycles.  Never compare across that flag.
* The trace must be read only after its **file size stops growing** — VCS exits before the trace
  pipe drains, and a truncated trace scores `Frobenius ≈ sqrt(fraction uncovered)`, which looks
  exactly like data corruption (8192/8192 → 3.5666%, 7930/8192 → 18.1%).  Both verifiers now print
  an explicit INCOMPLETE warning; heed it instead of believing the number.
* `tmask` in the `[ISSUE]` trace is **one nibble per lane**, not one bit — an all-lanes-active store
  prints `tmask=0x1111111111111111`.  The natural `(tmask >> lane) & 1` keeps only lanes 0,4,8,12.
* The simulator is timing-deterministic: identical cycle counts and identical failure times across
  `+ntb_random_seed` values.  A seed only randomises uninitialised state, so it changes *error
  metrics* of a racy build, never its schedule.  Validate correctness on ≥2 seeds anyway.

## Correctness must be checked PER TILE

`fa_verify_out.py` folds every store into one image, so for an N-tile `FA_STEADY` run it only ever
scores the **last** tile of the last cluster to write a cell.  `fa_verify_tiles.py` splits the O
store stream per cluster and starts a new tile image whenever an address repeats (`finalize_O`
writes each of the 4096 words exactly once per tile), then scores all 2N images independently.
3.5666% vs `golden_O_u16.npy` is correct; `golden_O_flash_u16.npy` is the wrong golden for
`FULL_ATTN2` (it is the streaming online-softmax reference).

This distinction is not academic — re-scoring earlier runs that had all reported a clean "bit-exact
3.5666%" under the merged check:

| run  | merged check | per-tile check |
|------|--------------|----------------|
| hsb4 | 3.5666%      | 8/8 images correct |
| hsc4 | 3.5666%      | 8/8 correct |
| hse4 | 3.5666%      | **7/8** — one image wrong |
| hsg4 | 3.5666%      | **6/8** — worst 5.50% |
| hsi4 | 3.5666%      | **6/8** — cluster 1 tiles 1 and 3 at 3.88% / 3.98% |

### Why not `kernels/flash_attention_mx/fa_pertile.py`

That tool buckets O stores by the most recent MARK index and **does not separate the two clusters**
(`grep clid fa_pertile.py` finds nothing).  Both clusters write MARKs and both write the same O
addresses, interleaved in one trace, so cluster 1's MARK store advances the bucket cursor while
cluster 0 is still mid-`finalize_O`.  Every bucket therefore mixes clusters and generations.

Run on `st0` — the **unmodified baseline**, whose whole-file verify is 8192/8192 at 3.5666% and whose
per-tile verify here is 8/8 images at exactly 4096/4096 words each — it reports:

    after m[10] covered 6176/8192 Frobenius 51.91%     after m[12] covered 2272/8192  85.28%
    after m[21] covered 5344/8192           59.37%     after m[23] covered 3456/8192  76.12%
    after m[32] covered 5536/8192           57.63%     after m[34] covered 3520/8192  75.47%
    after m[43] covered 5760/8192           55.53%     after m[44] covered 2880/8192  80.45%

i.e. it calls a known-good baseline 52-85% wrong.  `fa_verify_tiles.py` is self-validating instead:
grouping by cluster and then by address-repeat yields **exactly** 4096 words per image, and the 8
images account for exactly the 32,768 store words the whole-file verify sees.  If the grouping were
wrong that accounting could not close.  Any "steady-state correctness bug" diagnosed with the
mark-bucketing tool should be re-scored before it is believed.

## The scale-SRAM hazard that bounds this whole approach

The gemmini scale SRAM only accepts 8-byte writes (`GemminiTile.scala:286` asserts `size == 3`),
but an RV32 Muon lane can only emit 4-byte stores — `store64_shared`
(`lib/include/mu_intrinsics.h:27-32`) is literally two 4-byte stores.  So the SF leg of the
gemmini tile's slave xbar sits behind `FlitMergeNode(from = 4, to = 8)`
(`GemminiTile.scala:186-189`), which pairs consecutive 4-byte requests into one 8-byte request.
Its pairing state — the beat counter, the merged-request register, and the per-source `wasMerged`
bit that decides the size a **response** is reported with (`FlitMergeNode.scala:33-100`) — is
global to the port.

The Rocket host is the opposite: a 4-byte host write into the cluster is silently dropped, so the
host must use `sd`, which takes the *non-merging* path through the same node.  Consequences,
all measured on this tree:

* **Host SF writes confined to boot are safe.**  Every `FA_NOSCALES`-only build (host prefills once,
  GPU never loads MX scales) has been clean in ~20 runs.  The host's SF writes all *precede* the
  GPU's first `pack_scales_to_sfmem`, so the two never share the node's state.
* **A per-tile host SF refill needs two things, and then it works.**  As originally scheduled,
  `FULL_ATTN2 FA_STEADY FA_NT4 FA_NOSCALES FA_HOSTHS` hangs and then trips
  `"'D' channel contains improper response size"` at the host→cluster `extReqXbar` monitor
  (`RadianceCluster.scala:112`) at **exactly** 330,243,000 ps — two independently built images ×
  two seeds, all four identical.  It takes BOTH of the following to make it clean, and neither alone
  is enough:
    1. confine every host push to a window in which the GPU provably issues no gemmini command
       (`[PVDONE >= t .. QK_READY = t+1]` and `[QKDONE >= t+1 .. V_READY = t+1]`), and
    2. put a store-to-load `fence` in front of every mailbox poll, so no host store into the cluster
       is still in flight when the host issues a load into it.
  With both, the configuration runs a complete `FA_NT4` to 810,409,000 ps with **no assertion and
  8/8 tile-images correct** at 83,517 / 86,482 cyc/tile.  That is the streaming-honest number: the
  host re-pushes all 4,608 scale bytes every tile, as a real FA must when K_j/V_j change, and it
  costs only 636 cyc/tile on cluster 0 over keeping them resident.
* **Adding `FA_HOSTCFG` on top brings the deadlock back** (163,895,000 ps).  Having the host *also*
  drive the gemmini command port at +0x84000 puts a second stream of host traffic into the same
  `tlSlaveXbar` at points that interleave with the GPU's own 4-byte SF writes, and no schedule I
  tried survives it.  So the host can own the scale SRAMs, or the command port, but not both —
  unless the GPU is taken off the SF port entirely (`FA_HOSTPACK`, in flight).
* `FA_HOSTPACK` (take the GPU off the SF port entirely) **delays but does not remove** the
  `FA_HOSTCFG` deadlock: it gets through ~3 tiles instead of ~1 and then asserts at 490,027,000 ps.
  Combined with the fact that `FA_HOSTHS` alone is clean, that pins the trigger on the host driving
  the **command port**, not the scale SRAM.
* An earlier campaign reported a clean 84,549 cyc/tile for `FA_HOSTHS` with the *old* schedule.  That
  build carried four extra instrumentation stores per tile; removing the *probe* uncovered the race
  in the code being probed.
* **Lane-parallel GPU scale writes (`FA_LANESC`) are simply invalid.**  16 lanes writing SF
  concurrently produce non-consecutive pairs and hit the merge node's own assertion,
  `FlitMergeNode.scala:62 assert(in.a.bits.address === mergedReq.address + byteOffset)`, at
  77,173,000 ps.  Not a timing issue — a contract violation.
* The structurally clean way to do a per-tile refill is therefore to **remove the GPU as an SF
  requestor entirely**: `-DFA_HOSTPACK` (host) plus `-DFA_NOPACK` (kernel body) has the host read
  the 512 runtime E8M0 P scales out of `SCALE_SMEM`, pack them 4-per-word and write them into
  `SF_MEM_A` itself, so every request reaching the merge node is an 8-byte host store that never
  merges and there is no shared state left to corrupt.

## Measured result (RadianceTapeoutSimConfig, frozen snapshot, no `+dramsim`, seed 12345)

Steady state: `FA_STEADY FA_NT4`, slope = `(T[3]-T[1])/2` so boot + icache warm-up cancel exactly;
mesh-busy is fixed at 16,420 cyc/tile, so util = 16420/slope.  Both clusters are reported because
the kernel is not finished until the slower one is.  **Correctness is the per-tile check over all
2N tile-images** (`fa_verify_tiles.py`), never the whole-file one.

| steady-state build | cl0 cyc/tile | cl0 util | cl1 cyc/tile | cl1 util | per-tile correctness |
|---|---|---|---|---|---|
| `FULL_ATTN2` — baseline, GPU loads all MX scales | 97,398 | 16.86% | 96,243 | 17.06% | **8/8 correct** |
| `+ FA_NOSCALES` — host prefills the scale SRAMs | 82,451 | 19.91% | 85,294 | 19.25% | **5/8** — cl1 tiles 1,2,3 at 112.64 / 113.45 / 113.45% |
| `+ FA_NOSCALES` seed 777 | 82,451 | 19.91% | 86,240 | 19.04% | 5/8 — *identical* failure and Frobenius |
| `+ FA_NOSCALES FA_EARLYV` **seed 777** | 82,881 | 19.81% | 83,321 | 19.71% | 6/6 correct — slope bit-identical to seed 12345 |
| **`+ FA_NOSCALES FA_EARLYV`** | **82,881** | **19.81%** | **83,321** | **19.71%** | **8/8 correct** |
| **`+ FA_NOSCALES FA_HOSTHS`** — host re-pushes all 4,608 B EVERY tile | **83,517** | **19.66%** | **86,482** | **18.99%** | **8/8 correct** |
| `+ FA_NOSCALES FA_EXPMVIN` (explicit V mvin only, no hoist) | 82,975 | 19.79% | — | — | 4/4 correct (2 tiles measured) |
| `+ FA_HOSTHS FA_HOSTCFG` — per-tile refill + host cfg/mvin | 79,115 | 20.75% | 81,359 | 20.18% | 6/8 — and it deadlocks, see below |
| `+ FA_HOSTHS FA_HOSTCFG FA_EARLYV` | 79,977 | 20.53% | — | — | **0/4** |
| `FA_HOST_ASEL0` — deliberate act-half collision (diagnostic) | 83,600 | 19.64% | — | — | tiles >= 1 at 64.5187% on **both** clusters |
| `FA_NS_PAD` — blunt 3,000-nop pre-fence pad (diagnostic) | 157,864 | 10.40% | — | — | 4/4 correct — pad costs ~75k, useless as a fix |
| `FA_LANESC` — lane-parallel GPU scale writes | — | — | — | — | **ABORTS**: `FlitMergeNode.scala:62` at 77,173,000 ps |

**Two verified host-offload configurations, both 8/8 correct on a complete `FA_NT4` run:**

* *scales resident* (`FA_NOSCALES FA_EARLYV`): 97,398 -> 82,881 cyc/tile, **16.86% -> 19.81%**
  (slower cluster 96,243 -> 83,321, 17.06% -> 19.71%).  -14,517 cyc/tile, -14.9%.
* *per-tile refill, streaming-honest* (`FA_NOSCALES FA_HOSTHS`): 97,398 -> 83,517,
  **16.86% -> 19.66%** (slower cluster 96,243 -> 86,482, 17.06% -> 18.99%).  The host re-pushes all
  4,608 scale bytes every tile and that costs only **636 cyc/tile** more than keeping them resident,
  i.e. MX scale movement is essentially free once it is the host's job.

Both beat the baseline by ~2.6-3.0 points of mesh utilisation with every tile-image bit-correct.

Single shot (`FULL_ATTN2` without `FA_STEADY`; metric is the last MARK, i.e. end of kernel):

| single-shot build | cl0 | cl0 util | cl1 | cl1 util | saving vs baseline |
|---|---|---|---|---|---|
| baseline | 125,091 | 13.13% | 123,351 | 13.31% | — |
| `+ FA_NOSCALES` | 118,677 | 13.83% | 119,504 | 13.74% | **-6,414 / -3,847** |
| `+ FA_NOSCALES FA_EARLYV` | 124,644 | 13.17% | 124,733 | 13.16% | -447 / +1,382 (a wash) |
| `+ FA_HOSTHS FA_HOSTCFG` | 120,971 | 13.57% | 120,194 | 13.66% | +4,120 / +690 |

All single-shot rows are 2/2 correct — a one-tile run only ever executes the tile that always works.
Note the honest shape of this: **`FA_EARLYV` costs ~6k at single shot and is only free in steady
state**, and `FA_HOSTHS`/`FA_HOSTCFG` cannot pay at single shot by construction (tile 0 always
issues its own commands; the host has nothing to replay yet).  Per-tile offload is a steady-state
item only.

### Two independent bugs, kept separate

**(a) Wrong output, fingerprint 112.6418% / 113.4523%.**  Appears the moment `FA_NOSCALES` removes
the GPU's ~12.6k-cycle `load_scale_factors`, which used to sit between the operand move-in issue and
the matmul's leading `gemmini_fence`.  It is deterministic (bit-identical on seeds 12345 and 777),
hits only whichever cluster loses the race, and reproduces in the main kernel
(`kernels/flash_attention_mx`, cluster 0 tiles 2 and 3, same 112.6418% / 113.4523%).  Fixed by
issuing the PV operand move-in with explicit `gemmini_extended_mvin` commands instead of the
loop-FSM path — the documented H8 phantom-completion bug.  `FA_EXPMVIN` shows the explicit move-in
alone is sufficient; hoisting it above `bar2` (the other half of `FA_EARLYV`) is not required.
This is **not** the SF_MEM_A act-half collision: forcing that collision on purpose
(`FA_HOST_ASEL0`) gives 64.5187% on both clusters, a different and cluster-symmetric fingerprint.

**(b) Cluster-fabric deadlock, whenever the host writes the scale SRAM *after* the GPU has.**
Deterministic and seed-independent in every variant tried:

| build | assertion | at |
|---|---|---|
| `FA_NOSCALES FA_HOSTHS` | `'D' channel contains improper response size` | 330,243,000 ps (2 images x 2 seeds, identical) |
| `+ FA_HOSTCFG`, refill rescheduled into ROCC-quiet windows | same | 179,473,000 ps (2 seeds, identical) |
| `+ store-to-load `fence`` in `mbox_get` | same | 163,895,000 ps |
| `+ FA_HOSTPACK FA_EARLYV` | same | 214,127,000 ps |
| `+ FA_HOSTPACK` (no `FA_EARLYV`) | none through >= 2 tiles | — |

Every `FA_NOSCALES`-only build (host writes the SF SRAMs once at boot, before the GPU's first
`pack_scales_to_sfmem`) has been clean in ~20 runs.  See the merge-node section above for what the
waveform shows.

### Effect on the main (pipelined) kernel

`kernels/flash_attention_mx` at md5 `8de04db0`, built `FULL_ATTN2 FA_STEADY FA_NT4 FA_NOSCALES`:

| build | cl0 cyc/tile | cl1 cyc/tile | per-tile correctness |
|---|---|---|---|
| `FA_NOSCALES` | 70,762 (23.20%) | 71,552 (22.95%) | 6/8 — cl0 tiles 2,3 at **112.6418 / 113.4523%** (the same fingerprint as the frozen snapshot) |
| `FA_NOSCALES FA_EARLYV` | 71,010 (23.12%) | 69,716 (23.55%) | 5/8 — cl0 tile 2 down to **6.8648%**, 2 images truncated |

So `FA_EARLYV` removes the 112.64% failure mode from the main kernel too (112.64% -> 6.86%) at no
cost in the slope, but the main kernel has a **residual** error of its own on top of it.  Whoever owns
that file should apply the explicit PV move-in (`FA_EXPMVIN` is the minimal form) and then chase the
remaining 6.86% separately -- it is a different bug, not this one.

### Phase decomposition of the offloaded steady state

`FA_CFGPROF`, tile 2, cluster 0.  CPROF's own stores add ~5k/tile, so read this as an attribution,
not a budget:

    QK prefetch (host issues cfg+mvin, GPU only waits)                    1,707
    QK matmul   lead-fence 137 | cfg 1,102 | issue 314 | mesh 8,860      10,647
    bar2 + softmax                                                      17,211
    PV prefetch                                                            763
    requant + pack + bar3                                               20,718
    PV matmul   lead-fence 1,520 | cfg 1,494 | issue 2,224 | mesh 8,626  13,864
    bar4 + finalize_O                                                   20,615

The two mesh trail-fences are 17,486 of the 16,420 nominal mesh-busy, i.e. essentially all mesh time
is already exposed.  What remains is SIMT work (softmax + requant ~24k), barriers (~17k) and
`finalize_O`.  The GPU-side gemmini command issue a further host offload could take is only
137+1,102+314+1,520+1,494+2,224 = **6,791 cyc/tile**, and most of the PV part is back-pressure on
`gemminiIO.ready` rather than store latency — moving the issuer does not remove back-pressure,
because the GPU would still block on the reply.  That is why the matmul-issue offload was
measured-and-rejected rather than built.

## Build matrix

| define | meaning |
|--------|---------|
| `FA_NOSCALES` | GPU does not load MX scales at all; the host prefills them once at boot.  Also enables the host-side code in `host.cpp` (the Makefile forwards the `-D` list). |
| `FA_HOSTHS` | host re-pushes all 4,608 scale bytes every tile (what a streaming FA needs), gated on per-tile `QKDONE`/`PVDONE` mailbox counters |
| `FA_HOSTCFG` | host also replays the 12-command gemmini CONFIG + move-in stream that the GPU recorded on tile 0 |
| `FA_HOSTPACK` + `FA_NOPACK` | host also packs the runtime P scales, so the GPU never writes the scale SRAM |
| `FA_STEADY`, `FA_NT<n>` | run the tile body n times and report the slope, so boot/icache/first-touch cancel |
| `FA_CFGPROF` | CPROF stamps inside prefetch/compute (config / move-in / scale-wait, lead-fence / issue / trail-fence) |
| `FA_HOST_TIMING` | host timeline echo — **instrumentation, ~1.6k cyc/tile, never in a reported number** |
| `FA_HOSTPROBE`, `FA_HOSTCFG_NOMVIN`, `FA_HOSTHS_VEARLY` | measurement-only probes |

Goldens and `include/fa_data.h` are not committed here; regenerate with `fa_gen_data.py` /
`fa_gen_goldens.py` (`--Sq 64 --Sk 256 --d 128 --block_n 64 --seed 0`).
