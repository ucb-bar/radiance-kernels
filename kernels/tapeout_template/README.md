# tapeout_template

A minimal kernel for the taped-out Radiance part (1 cluster, 2 cores) to copy and extend.
The Rocket host prints a hello line, launches a `uint32_t` vector add on the GPU, waits for the
tapeout epilogue, and checks the result.

| file | role |
|---|---|
| `template.h` | addresses, argument struct and constants shared by host and GPU |
| `kernel.cpp` | GPU: `main()` per core, then the entry function, which ends in `rad_tapeout_epilogue()` |
| `host.cpp` | Rocket: write inputs and arguments, `rad_host_run()`, check |

Shared helpers: `lib/include/rad_host.h` (host: buffered output, GPU DRAM access, `rad_host_run()`)
and `lib/include/rad_tapeout.h` (GPU: `rad_tapeout_begin()`, `rad_tapeout_epilogue()`).

## Build

```bash
make    # -> kernel.soc.elf (fused host + GPU image)
```

## Expected output

```
Hello from Rocket: tapeout template, vecadd of 4096 words
rad_host_run: started=2/2 done=2/2 gen0=0xac...... trace0=0x000000a3 gen1=0xac...... trace1=0x000000a3 host_cycles=...
check: wrong=0/4096 still_poison=0
PASS
```

## Rules for extending it

1. **End the entry function with `rad_tapeout_epilogue()`.** Do not return and let a core assert
   `finished`: on this part the finish edge fires the L0d flush unit, which wedges the cache, and
   at occupancy 2 or more `finished` never asserts. The epilogue drains L0d and L1 and parks the
   cores; the host then soft-resets the GPU.
2. **Pass arguments through memory the host writes** (`TT_ARGS_ADDR`). Zero-initialized statics
   are `.bss`, which is never loaded or zeroed on this platform. A static with a non-zero
   initializer (`.data`) is also safe.
3. **Do not call `mu_fence()` from every warp.** Only one warp per core may fence; the epilogue
   does the one fence that is needed.
4. **Keep barriers out of divergent branches.** Put them outside the branch and give the branch an
   explicit `else { asm volatile("nop"); }`.
5. **The host has no FPU.** Any floating-point code in `host.cpp`, including a `double` that
   only causes register spills, makes the host trap before it prints anything.
6. **Use the buffered `rad_put*()` helpers**, not `printf`: HTIF `printf` costs about 1 s per
   character on the U250.
7. **Stay in the `0x1F00_0000` window** for buffers, and clear of the epilogue drain scratch at
   `0x1F20_0000`..`0x1F34_0000`.
8. **printBuf keeps its contents across runs.** Clear any postbox slot the host reads, from the
   GPU, before it can be posted. `rad_tapeout_begin()` does this for the epilogue slots.

## Running

The FireSim metasim of the tapeout RTL runs this kernel, epilogue included, and prints `PASS`.

On the U250 the first run of an image after a different image fails with `started=0/2`, and the
next run of the same image passes; retrying inside the same program does not help. The likely
cause, not established, is that the GPU executes the image already in DRAM at SoC reset, before the
host loads the new one, and the soft-reset launch does not invalidate the instructions it cached.
Run a new image twice and read the second result. `rad_tapeout_begin()` and `rad_host_run()` make
this case show up as a failure instead of letting a stale postbox value pass as success.
