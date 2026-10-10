# fa_mxfp8_sr on the HBM memory system: radiance RadianceE4M3MxGemmini{SingleCluster,TwoCluster,FourCluster}HBMConfig
# (split host / GPU L2, one DRAM channel per GPU slice, GPU memory hashed over the slices).  Source and data are
# fa_mxfp8_sr's.  The GPU load segments are scrambled into the hashed layout for SimDRAM +loadmem (MU_ADDR_HASH=1;
# MU_ADDR_HASH=0 for TSI loading).  build/{1,2,4}sm: 1, 2 or 4 clusters (the 4 query tiles split evenly).
PROJECT   = fa_mxfp8_sr_hbm
SRC       = ../fa_mxfp8_sr
MU_SRCS   = $(SRC)/kernel.cpp
HOST_SRCS = $(SRC)/host.cpp
DEPS      = $(SRC)/fa_data.h
BLOBS     = $(SRC)/data/fa.region1.bin $(SRC)/data/fa.region2.bin $(SRC)/data/fa.region3.bin
OCC       = 4
SMS      ?= 1 2 4
KERNEL_MU_CFLAGS = -ffp-contract=off -DNIGHTLY_MX_WAIT_READY -DNIGHTLY_MX_RETIRE
MU_ADDR_HASH ?= 1
include ../nightly.mk
$(SRC)/fa_data.h $(SRC)/data/fa.region1.bin $(SRC)/data/fa.region2.bin $(SRC)/data/fa.region3.bin &:
	$(MAKE) -C $(SRC) fa_data.h
