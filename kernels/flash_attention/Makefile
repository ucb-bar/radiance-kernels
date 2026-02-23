PROJECT = flash_attention

MU_SRCS = flash_attention.cpp
CPU_SRCS = host.cpp
MU_BIN_OBJS = numpy.a.o numpy.b.o

numpy.%.o: numpy.%.bin
	./numpy_bin_to_o.sh $<

flash_attention.mu.o: flash_attention.cpp flash_impl.hpp

include ../common.mk
