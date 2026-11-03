# memstress: GPU memory bandwidth stress kernel with per-warp cycle records (see the kernel.cpp header).
# The rv64 host prints one AGG line per core; memstress_bw.py converts the AGG lines of a run log into
# load-phase bandwidth:  python3 memstress_bw.py <MS_WS> <log>
#
#   make                    read bandwidth, full size: 256 KiB and 4096 warp loads per core
#   make SIZE=short         128 KiB and 2048 warp loads per core
#   make TEST=store         store, fence, then load and check every word (data_errors in the host report)
#   make HBM4=0             a config without the GPU address hash and with a coherent cluster L1
#
# HBM4=1 (default) matches RadianceHBMConfig: the GPU address hash (64 B unit, 16 channels, 2 channels per
# L2 slice, channel set stride 64) and the incoherent 4-bank cluster L1, whose flush registers the kernel
# writes to publish its records. The ELF is pre-scrambled for the hash: load it with SimDRAM +loadmem,
# never through TSI. Run `make clean` after changing SIZE, TEST or HBM4 (they are not file dependencies).
#
# In the read test (MS_STORE=0) the loads verify nothing; use TEST=store to check data.

MU_SRCS = kernel.cpp
HOST_SRCS = host.cpp

SIZE ?= full
TEST ?= read
HBM4 ?= 1

# 8 warps on each of the 8 cores (4 clusters x 2), each warp streaming its own chunk with 8-register
# assembly load batches; stream starts skewed by 2 KiB so the streams do not move through the channels in step
MS_DEFS := -DMS_ASM=2 -DMS_ASM_REGS=8 -DMS_STREAM=1 -DMS_NWARPS=8 -DMS_CORES=0xFF -DMS_SKEW=2048
ifeq ($(SIZE),short)
MS_DEFS += -DMS_WS=0x20000 -DMS_INSTR=2048
else
MS_DEFS += -DMS_WS=0x40000 -DMS_INSTR=4096
endif
ifeq ($(TEST),store)
MS_DEFS += -DMS_STORE=1 -DMS_FENCE=1
else
MS_DEFS += -DMS_STORE=0
endif

ifeq ($(HBM4),1)
MS_DEFS += -DMS_L1_FLUSH=1 -DMS_L1_BANKS=4
MU_ADDR_HASH ?= 1
MU_ADDR_HASH_UNIT ?= 64
MU_ADDR_HASH_SLICES ?= 16
endif

EXTRA_MU_CFLAGS += $(MS_DEFS)

include ../common.mk

HOST_CXXFLAGS += $(filter -DMS_CORES=% -DMS_NWARPS=% -DMS_DEADLINE=%,$(MS_DEFS))
ifeq ($(HBM4),1)
MU_ADDR_HASH_ARGS += --channels-per-slice 2 --chan-set-stride 64
endif
