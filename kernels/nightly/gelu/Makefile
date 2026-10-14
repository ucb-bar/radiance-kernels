PROJECT   = gelu
MU_SRCS   = kernel.cpp
HOST_SRCS = host.cpp
DEPS      = gelu_data.h
BLOBS     = data/gelu.region1.bin data/gelu.region2.bin data/gelu.region3.bin
OCC       = 4
include ../nightly.mk
gelu_data.h data/gelu.region1.bin data/gelu.region2.bin data/gelu.region3.bin &: gen_data.py $(GEN_STAMP)
	python3 gen_data.py $(GEN_ARGS)
