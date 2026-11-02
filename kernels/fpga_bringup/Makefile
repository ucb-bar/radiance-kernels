PROJECT = fpga_bringup

MU_SRCS = kernel.cpp
HOST_SRCS = host.cpp

include ../common.mk

# host-side define hook (append AFTER the include: common.mk uses ?= so an env var would REPLACE
# HOST_CFLAGS and silently drop -march/-mabi).
HOST_CXXFLAGS += $(FB_HOST_DEFS)
