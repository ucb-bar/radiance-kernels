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
* **A per-tile host SF refill while the GPU still writes SF is not safe.**
  `FULL_ATTN2 FA_STEADY FA_NT4 FA_NOSCALES FA_HOSTHS` hangs and then trips
  `"'D' channel contains improper response size"` at the host→cluster `extReqXbar` monitor
  (`RadianceCluster.scala:112`) at **exactly** 330,243,000 ps — two independently built images ×
  two seeds, all four identical.  Adding `FA_HOSTCFG` moves the failure rather than removing it
  (179,473,000 ps, again seed-independent).  Reordering the refill into GPU-ROCC-quiet windows does
  not remove it either: the schedule change alters host instruction timing and the failure simply
  lands somewhere else.  It is a lottery across builds, which is why an earlier campaign reported a
  clean 84,549 cyc/tile for this configuration — that build carried four extra instrumentation
  stores per tile, and removing the *probe* uncovered the race in the code being probed.
* **Lane-parallel GPU scale writes (`FA_LANESC`) are simply invalid.**  16 lanes writing SF
  concurrently produce non-consecutive pairs and hit the merge node's own assertion,
  `FlitMergeNode.scala:62 assert(in.a.bits.address === mergedReq.address + byteOffset)`, at
  77,173,000 ps.  Not a timing issue — a contract violation.
* The structurally clean way to do a per-tile refill is therefore to **remove the GPU as an SF
  requestor entirely**: `-DFA_HOSTPACK` (host) plus `-DFA_NOPACK` (kernel body) has the host read
  the 512 runtime E8M0 P scales out of `SCALE_SMEM`, pack them 4-per-word and write them into
  `SF_MEM_A` itself, so every request reaching the merge node is an 8-byte host store that never
  merges and there is no shared state left to corrupt.

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
