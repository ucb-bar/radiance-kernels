# Radiance Kernels

This WIP repo is based off of [virgo-kernels](https://github.com/ucb-bar/virgo-kernels).

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

Similar to the ISA tests, go to `kernels/<kernel_name>` and run `make`. The
output `kernel.radiance.elf` is what you want.
