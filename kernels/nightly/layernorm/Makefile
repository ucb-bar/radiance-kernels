PROJECT   = layernorm
MU_SRCS   = kernel.cpp
HOST_SRCS = host.cpp
DEPS      = layernorm_data.h
BLOBS     = data/layernorm.region1.bin data/layernorm.region2.bin data/layernorm.region3.bin
include ../nightly.mk
layernorm_data.h data/layernorm.region1.bin data/layernorm.region2.bin data/layernorm.region3.bin &: gen_data.py $(GEN_STAMP)
	python3 gen_data.py $(GEN_ARGS)
