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
  stay fast and do not turn this into a DRAM-bandwidth test.
- `fmadd`: `consume` is a dependent fmadd into a side accumulator
  (`sink[i] = acc[i]*m + sink[i]`). Same FMA port as produce → consumer competes
  for issue bandwidth. Use this to contrast against the off-port store consumer.

## Parameters

| macro                | meaning                                  | default |
|----------------------|------------------------------------------|---------|
| `FMASTORE_CHAINS`    | `K`, number of independent chains        | 4       |
| `FMASTORE_CONSUMER`  | `0` = store to SMEM, `1` = dependent fmadd | 0 (store) |
| `FMASTORE_NUM_WARPS` | occupancy; `1` isolates ILP from TLP     | 1       |
| `FMASTORE_ITERS`     | loop trip count (runtime → no fold)      | 1024    |

## Build

```bash
# from repo root
source env.sh && cd kernels/fmastore && make
```

`make` builds the default sweep `kernel_<consumer>_k<K>` for
`CONSUMERS = store fmadd` × `CHAINS_LIST = 1 2 4 8`. Override either list, e.g.:

```bash
make CHAINS_LIST="1 2 3 4 6 8 12 16" CONSUMERS=store
```

A single ad-hoc build:

```bash
make kernel.radiance.elf FMASTORE_CHAINS=6 FMASTORE_CONSUMER=0 FMASTORE_NUM_WARPS=1
```

## What to measure

- **cycles vs `K`** at fixed deep RS: rises ~`K/L` then plateaus at the issue/FU
  limit; the knee is at `K* ≈ L·W` (FMA latency × issue width) → reads off `L`.
- **cycles vs RS depth** at fixed `K`: RS1 ≈ in-order (latency-bound) → deep RS
  (OOO, latency hidden). This curve is the headline OOO-issue result.
- Cross-check `store` vs `fmadd`: the off-port store consumer should let the FMA
  pipe reach a higher plateau. If a plateau falls *below* the FMA rate, suspect
  LSQ depth / store-port (separable by lowering store frequency).
