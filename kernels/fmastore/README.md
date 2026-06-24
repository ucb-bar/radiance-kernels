# fmastore

A compute-bound microbenchmark for Muon's **out-of-order issue**. It is the
FU-latency dual of `pointerchase` (which measures memory latency): here the only
long-latency edge per chain is the FMA, and there is no inner-loop memory load,
so the machine is FMA-latency-bound rather than memory-latency-bound (the GEMV
problem).

## Idea

Each thread runs `K` independent **depth-2 produce→consume chains**, laid out
**blocked** (chain 0's pair, then chain 1's pair, …) — *not* software-interleaved:

```
produce_0 ; consume_0 ; produce_1 ; consume_1 ; … ; produce_{K-1} ; consume_{K-1}
```

- `produce`: a depth-1 self-recurrent fmadd, `acc[i] = acc[i]*m + b`.
- `consume`: reads the just-produced `acc[i]`. Adjacent to its producer, so it
  is the dependency edge that stalls an in-order machine.

Because the layout is blocked, the *ready* `produce` of chain `i+1` sits **behind**
the *stalled* `consume` of chain `i`. Only a machine that issues out of order can
reach it. So:

- **in-order / RS depth 1** → stalls at every consume → latency-bound.
- **deep RS (OOO)** → issues the younger ready produces past the stalled consume
  → the `K` chains' latencies overlap → throughput climbs to the issue/FU limit.

This is the opposite of the accumulator-interleave pattern, where the *compiler*
floats the ready work to the front and even an in-order machine wins. Here the
**hardware RS** is the thing under test.

## Why this isolates OOO cleanly (vs. GEMV)

- No inner-loop loads → no memory-latency domination, no LSU-token / RS-admission
  pathology.
- The `produce` is self-recurrent (`rd == rs1`), so its WAW is redundant with the
  RAW recurrence (same scoreboard counter) and the blocking spaces successive
  writes `2K` instructions apart → WAW never gates admission. The verified hot
  loops show **zero** instructions from the ordering barriers and **no register
  spills** through `K=8`.

## Consumer variants (`CONSUMER`)

- `store` (default): `consume` is `sh.shared` into SMEM. A store has **no rd**, so
  it creates no WAW/WAR at admission (`Hazard.scala` gates on `hasRd`) and drains
  on the **LSU port**, not the FMA port — the cleanest dependent consumer, and it
  lets the FMA pipe run flat-out. Stores hit **on-chip SMEM** (`__shared`), so they
  stay fast and do not turn this into a DRAM-bandwidth test. Layout is **word-strided
  / chain-major** (`smem[i*CHAIN_STRIDE + tid*2]`): each thread writes a 4-byte
  (one-bank) slot per chain, so the warp's 16 lanes hit 16 distinct 4-B SMEM banks —
  conflict-free and coalesced for **any K** (fp16 sits in the low half of its word;
  SMEM is never read back). This keeps the store path from bottlenecking the sweep.
- `fmadd`: `consume` is a dependent fmadd into a side accumulator
  (`sink[i] = acc[i]*m + sink[i]`). Same FMA port as produce → consumer competes
  for issue bandwidth. Use this to contrast against the off-port store consumer.

## Parameters

| macro                | meaning                                  | default |
|----------------------|------------------------------------------|---------|
| `FMASTORE_CHAINS`    | `K`, number of independent chains        | 4       |
| `FMASTORE_CONSUMER`  | `0` = store to SMEM, `1` = dependent fmadd | 0 (store) |
| `FMASTORE_NUM_WARPS` | `W`, occupancy; `1` isolates ILP from TLP | 1       |
| `FMASTORE_ITERS`     | **per-core total produce/consume pairs** (fixed work budget), split across the `W` warps and `K` chains → `ITERS/(W·K)` outer iters per warp, so aggregate per-core work is invariant across **both** `W` and `K` (cycles comparable along both axes); runtime → no fold | 4096 |
| `FMASTORE_UNROLL`    | outer-loop unroll factor — the only hot-loop branch (back-edge) fires once per `UNROLL` iters instead of every iter (fewer control-flow stalls, higher IBUF occupancy); decoupled from `K`/`W` | 8 |

Constraints (enforced by `static_assert`): `ITERS % (W·K) == 0` and `(ITERS/(W·K)) % UNROLL == 0`. Keep `ITERS` large enough that the busiest corner (max `W·K`) still has a healthy `ITERS/(W·K)` outer-iteration count for steady state (default 4096 → 64 at `W=K=8`).

## Build

```bash
# from repo root
source env.sh && cd kernels/fmastore && make
```

`make` builds variants named `kernel_<consumer>_k<K>_w<W>`:
- **store**: full occupancy matrix `CHAINS_LIST × WARPS_LIST` (default `1 2 4 8` × `1 2 4 8` = 16).
- **fmadd**: `CHAINS_LIST × FMADD_WARPS` (default `1 2 4 8` × `1` = 4), a `w1` comparison baseline.

Override any axis, e.g.:

```bash
make CHAINS_LIST="1 2 4 8" WARPS_LIST="1 2 4 8"     # store occupancy matrix
```

A single ad-hoc build:

```bash
make kernel.radiance.elf FMASTORE_CHAINS=6 FMASTORE_CONSUMER=0 FMASTORE_NUM_WARPS=1
```

## What to measure

- **cycles vs `K`** (ILP) at fixed deep RS, `W=1`: with work fixed, cycles fall ~`1/K`
  (more chains hide FMA latency) until the knee at `K* ≈ L × issue_width`, then flatten
  at the throughput-bound floor → the knee reads off `L`. Latency hiding via **ILP**.
- **cycles vs `W`** (occupancy/TLP) at fixed `K`: because per-core work is now
  occupancy-invariant, cycles drop as more warps hide FMA latency by interleaving.
  This is latency hiding via **TLP** — the complement of the ILP axis.
- **cycles vs RS depth** at fixed `K`,`W`: RS1 ≈ in-order (latency-bound) → deep RS
  (OOO, latency hidden). This curve is the headline OOO-issue result.
- Cross-check `store` vs `fmadd`: the off-port store consumer should let the FMA
  pipe reach a higher plateau. If a plateau falls *below* the FMA rate, suspect
  LSQ depth / store-port (separable by lowering store frequency).
