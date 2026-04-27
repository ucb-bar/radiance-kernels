PROJECT = bfs

BFS_LEVELS = 0 1 2

MU_SRCS = \
	$(foreach l,$(BFS_LEVELS),bfs1_l$(l).cpp bfs2_l$(l).cpp)

HOST_SRCS = host.cpp

ifneq ($(strip $(OCCUPANCIES)),)
define add_bfs1_variant
MU_VARIANTS += bfs1_l$(1)_w$(2)
bfs1_l$(1)_w$(2).mu.o: bfs1_l$(1).cpp data kernel_impl.hpp
bfs1_l$(1)_w$(2).mu.o: MU_CFLAGS += -DBFS_NUM_WARPS=$(2)
endef

define add_bfs2_variant
MU_VARIANTS += bfs2_l$(1)_w$(2)
bfs2_l$(1)_w$(2).mu.o: bfs2_l$(1).cpp data kernel_impl.hpp
bfs2_l$(1)_w$(2).mu.o: MU_CFLAGS += -DBFS_NUM_WARPS=$(2)
endef

$(foreach l,$(BFS_LEVELS),$(foreach w,$(OCCUPANCIES),$(eval $(call add_bfs1_variant,$(l),$(w)))))
$(foreach l,$(BFS_LEVELS),$(foreach w,$(OCCUPANCIES),$(eval $(call add_bfs2_variant,$(l),$(w)))))
endif

include ../common.mk

data: bfs.py kernel_impl.hpp verify_trace.py host.cpp Makefile
	python3 bfs.py

$(addsuffix .mu.o,$(basename $(MU_SRCS))): %.mu.o: data %.cpp kernel_impl.hpp
	$(MU_CXX) $(MU_CFLAGS) -c $*.cpp -o $@
