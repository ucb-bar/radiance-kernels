# llama_e2e: a whole Llama-family model, one prefill chunk over a KV cache, every stage on the device
# (gen_data.py describes the phases).  One ELF per model and cluster count:
#   make MODEL=cputest                     host only: read the loaded GPU blobs back through the address hash
#   make MODEL=muontest                    Muon only: each SIMT row op once, byte-exact vs numpy
#   make MODEL=silufuse                    gateup-shaped GEMM with SILU fused (C and H vs mx_golden / numpy)
#   make MODEL=gemmact                     GEMMs bit-exact vs mx_golden, then GELU on Muon (whole-word stores)
#   make MODEL=smoke                       small random model (RTL simulation), checked against an emulation
#   make MODEL=tinyllama PY=<python with torch + transformers>
#                                          TinyLlama-1.1B, real weights and text (~1.1 GB ELF)
#   make MODEL=llama7b                     Llama-2-7B shapes, synthetic weights filled on the device
#   TAG=l1 ARGS="--layers 1"               a variant (own data dir and ELF name: llama_e2e_<MODEL>_<TAG>)
#   make ladder                            the FireSim test ladder (see the ladder target)
# e.g. Llama-7B, 1 layer + LM head on 2 or 4 clusters (FireSimE4M3MxGemmini{Two,Four}ClusterHBMRadianceConfig):
#   make MODEL=llama7b TAG=l1 ARGS="--layers 1" SMS="2 4"  ->  build/{2,4}sm/llama_e2e_llama7b_l1.soc.elf
# The host prints per-phase cycles, mesh_util_model and PASS/FAIL, then exits through HTIF.
# Toolchain: L=<radiance-kernels>/llvm/llvm-muon; make ... RISCV_SYSROOT=$L RISCV_TOOLCHAIN_PATH=$L
# -> build/{1,2,4}sm/llama_e2e_<MODEL>.soc.elf.  Built for the hashed HBM configs (MU_ADDR_HASH=1, load with
# +loadmem / FireSim LoadMem): RadianceE4M3MxGemmini{SingleCluster,TwoCluster,FourCluster}HBMConfig and the
# FireSimE4M3MxGemmini*HBMRadianceConfig targets.
MODEL ?= smoke
TAG   ?=
PY    ?= python3
ARGS  ?=
NAME      = $(MODEL)$(if $(TAG),_$(TAG))
DATA      = data_$(NAME)
PROJECT   = llama_e2e_$(NAME)
MU_SRCS   = kernel.cpp
HOST_SRCS = host.cpp
DEPS      = $(DATA)/llama_data.h
BLOBS     = $(DATA)/llama.region1.bin $(DATA)/llama.region2.bin $(DATA)/llama.region3.bin
OCC       = 4
SMS      ?= 1 2 4
KERNEL_MU_CFLAGS   = -ffp-contract=off -I$(DATA) -nostdlibinc -isystem $(LLVM_MUON)/include/c++/v1
KERNEL_HOST_CFLAGS = -I$(DATA)
GEN_ARGS := --model $(MODEL) $(ARGS)
MU_ADDR_HASH ?= 1
include ../nightly.mk
# per-variant generator stamp (nightly.mk's GEN_STAMP is shared by every variant of this directory)
VAR_STAMP := $(DATA)/gen_args.stamp
$(shell mkdir -p $(DATA); printf '%s\n' "$(GEN_ARGS)" | cmp -s - $(VAR_STAMP) || printf '%s\n' "$(GEN_ARGS)" > $(VAR_STAMP))
$(DATA)/llama_data.h $(DATA)/llama.region1.bin $(DATA)/llama.region2.bin $(DATA)/llama.region3.bin &: gen_data.py gen_unit.py $(VAR_STAMP)
	$(PY) $(if $(filter cputest muontest gemmact m2g silufuse,$(MODEL)),gen_unit.py,gen_data.py) $(GEN_ARGS) --out $(DATA) --header $(DATA)/llama_data.h

# FireSim test ladder, cheapest first: smoke (random, every stage checked), TinyLlama 1 and 4 layers (real
# weights), Llama-7B 2 layers (synthetic), then the full models
ladder:
	$(MAKE) MODEL=smoke
	$(MAKE) MODEL=tinyllama TAG=l1 ARGS="--layers 1"
	$(MAKE) MODEL=tinyllama TAG=l4 ARGS="--layers 4"
	$(MAKE) MODEL=llama7b TAG=l2 ARGS="--layers 2"
	$(MAKE) MODEL=tinyllama
	$(MAKE) MODEL=llama7b
.PHONY: ladder
