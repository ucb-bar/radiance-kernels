# Radiance Kernels

Kernels written for the [Radiance GPU](https://github.com/ucb-bar/radiance).

## Setup

You will need a **GCC** rv32 toolchain installed from
[riscv-gnu-toolchain](https://github.com/ucb-bar/riscv-gnu-toolchain/). You
might need to set ABI to be `ilp32` (not `ilp32d`). Export that path to an
environment variable `$(RISCV_TOOLCHAIN_PATH)` - there should be a directory
called `riscv32-unknown-elf` under that path. The purpose of this installation
is to retrieve headers & definitions from the GCC sysroot; the binary libraries
are not usable and no libraries (e.g. libm, crt) can be linked to due to ISA
differences.

There are two ways to install the Muon LLVM toolchain.

* Prebuilt: run `./scripts/llvm_prebuilt.sh`, which will decompress the
  existing archived LLVM binaries at `llvm/llvm-muon.tar.xz`. This is compiled
  on an Ubuntu 24.04 system with GLIBC 2.39, meaning there's a good chance it
  won't work on older systems. You will also need ZSTD. If for any reason the
  prebuilt toolchain doesn't work, use the second method.

* Build from scratch: run `./scripts/llvm.sh`. This will initialize the
  submodule located at `llvm/llvm-src`, which is not cloned by default due to
  its size. You will need `clang`, `clang++`, `ninja` installed in your system
  (more may be required) to build the Muon toolchain.

Once the toolchain is installed, compile the Muon runtime by running `make`
under `lib/`. There should now be `lib/libmuonrt.a`; if not, check previous
steps. As a further compiler sanity check, you can also inspect for any
`<unknown>`s in the assembly dump `lib/libmuonrt.dump`; all instructions in
executable sections should be 8 bytes and properly recognized by the
disassembler.

## Run ISA Tests

First initialize the submodule:
```bash
git submodule update --init muon-isa-tests
```

Then compile the tests:

```bash
autoconf
./configure
cd isa
make
```

This should generate a bunch of binaries under `isa` along with their dumps.
You can run the tests with [cyclotron](https://github.com/hansungk/cyclotron),
or straight from the `isa` directory by using `make run`. Note that you'll need
to go into `isa/Makefile` and change the `MUON_SIM` variable to point to the
compiled cyclotron binary.


## Run Kernels

Similar to the ISA tests, go to `kernels/<kernel_name>` and run `make`.
Use `*.elf` binaries for the `BINARY=` argument of the [Chipyard RTL
simulations](https://chipyard.readthedocs.io/en/latest/Simulation/Software-RTL-Simulation.html).

`*.soc.elf` binaries fuse both the host CPU and device GPU programs
into one, and they should be run on the SoC `CONFIG`s such as
`RadianceTapeoutConfig` and `RadianceSingleClusterConfig`.

`*.radiance.elf` binaries are GPU-only kernels that should be run on
host-less `CONFIG`s such as `MuonCoreTestConfig`.

See
[RadianceConfigs.scala](https://github.com/ucb-bar/radiance/blob/main/chipyard/RadianceConfigs.scala)
for the full list of configs.

## Kernel Optimization Guide

When optimizing Muon SIMT kernels, start by shaping the work so that each warp
owns a naturally contiguous slice of the problem. A common pattern is
warp-strided indexing: lane `l` in a warp handles element `base + l`, and
subsequent unrolled iterations advance by `MU_NUM_THREADS` rather than by `1`.
This keeps loads and stores contiguous across lanes and matches the memory
access style used in kernels such as `softmax` and `gemm_mxgemmini`.

For simple bandwidth-oriented kernels, avoid extra per-lane validity branches
in the hot loop when the problem shape allows it. If the input size is chosen
to be a clean multiple of the warp-owned chunk size, removing those branches
usually gives cleaner code generation and avoids paying repeated SIMT control
overhead.

Use two levels of unrolling deliberately:

* Inner-loop ILP unrolling increases the number of independent loads, ALU ops,
  and stores in flight within one loop body.
* Outer-loop unrolling amortizes loop branch overhead across multiple chunks,
  at the cost of larger code size and more instruction-cache pressure.

Simple unrolling is often not enough. If every unrolled iteration reuses the
same registers, the core can stall heavily on WAR and WAW hazards. The better
pattern is to give different unrolled lanes independent temporaries and arrange
the loop body more like software pipelining: issue a chunk of `lw.global`
operations back-to-back, then the arithmetic, then `sw.global` operations. In
practice, this tends to expose more memory-level parallelism and reduce short
dependency chains.

Inspect the generated `*.radiance.dump` before going straight to RTL. For a
kernel like `vecadd`, a good hot loop usually looks like:

* a dense block of back-to-back `lw.global`
* then the arithmetic block such as `fadd.s`
* then the `sw.global` block
* then one loop branch for the whole chunk

If the dump still shows many extra branches, scalarized address updates, or
interleaved use/def chains that defeat ILP, fix the source structure first.

Once the assembly looks reasonable, move to RTL and use the VCS log counters to
guide the next step. In particular:

* high WAR/WAW stalls usually mean the unrolled body still has too much
  register reuse or too little independence
* high scoreboard stalls usually mean short dependency chains are dominating
* high LSU stalls usually mean the kernel has become memory-bound and more ILP
  alone may not help much
* higher ILP can reduce total cycles even when reported IPC goes down, because
  the unrolled kernel may retire fewer dynamic instructions for the same work

Use a large enough problem size to amortize cold I-cache and startup effects
before drawing conclusions from performance counters. Small kernels can look
artificially bad because fixed overheads dominate the trace.

## Kernel Writing Pitfalls

### Deadlock due to branch duplication of `mu_barrier`

When putting threadblock barriers around thread-divergent branches
(`mu_barrier` in [mu_intrinsic.h](lib/include/mu_intrinsics.h)), be careful
about the compiler potentially duplicating the barrier to both branch paths and
resulting in a deadlock.  For example:

```
if (tid_in_threadblock == 0) {
    // do something
}
mu_barrier(0, NUM_WARPS);
```

may be transformed to:

```
if (tid_in_threadblock == 0) {
    // do something
    mu_barrier(0, NUM_WARPS);
} else {
    mu_barrier(0, NUM_WARPS);
}
```

This may result in a deadlock, since the thread-divergent warp 0 executes
`mu_barrier` twice due to SIMT branch serialization.  Because the other warps
are convergent, they execute the barrier once, and warp 0 deadlocks due to no
participation from other warps.

#### Workarounds

This problem requires a compiler fix, and otherwise can only be worked around
with various levels of friction.

Putting a nop explicitly at the else-clause sometimes helps:

```
if (tid_in_threadblock == 0) {
    // do something
} else {
    asm volatile ("nop");
}
mu_barrier(...);
```

Placing the barrier further away from divergent branches also helps, e.g. by
wrapping the branch inside a function.

Using the `-Os` compiler flag also seems to keep the compiler from doing aggressive
branch-duplication, albeit with a performance impact.

We put `__attribute__((convergent, noinline))` directive to `mu_barrier`, but
that doens't seem to fix this.
