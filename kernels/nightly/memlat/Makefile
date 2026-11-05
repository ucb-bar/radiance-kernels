# memlat: GPU load-to-use DRAM latency, idle and under Muon / DMA load (kernel.cpp).  Data: gen_data.py.
#   make   -> build/{1,2}sm/memlat.soc.elf
PROJECT   = memlat
MU_SRCS   = kernel.cpp
HOST_SRCS = host.cpp
DEPS      = memlat_data.h
BLOBS     = data/memlat.region1.bin
PY       ?= python3
include ../nightly.mk

memlat_data.h data/memlat.region1.bin &: gen_data.py
	$(PY) gen_data.py
