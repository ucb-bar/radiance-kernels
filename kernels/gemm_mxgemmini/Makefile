PROJECT = mxgemm

# Each source builds to its own .soc.elf (common.mk builds one elf per MU_SRCS entry).
# The list below is the set that compiles. Excluded (present in the tree but not built):
#   - tk256/tk512 fullout+requant configs: infeasible, C does not fit in the scratchpad
#     alongside the double-buffered A/B tiles (mxgemm_lib.hpp C_FITS_IN_SPAD static_assert;
#     a 128x128 output tile needs TILE_K <= 128). The feasible tk128 replacements are listed.
#   - fp6 tk128 singletile/m128n128k256 configs: no data header (fp6 gen is not yet wired up).
#   - mxgemm.flash_contention.cpp: references a removed SPAD_DEST symbol.
MU_SRCS = mxgemm.cpp mxgemm_singletile.cpp \
	  mxgemm.fp8.singletile.tm128tn128tk128.requant.cpp \
	  mxgemm.fp8.singletile.tm64tn64tk64.fullout.cpp \
	  mxgemm.fp8.singletile.tm64tn64tk64.requant.cpp \
	  mxgemm.fp8.m64n64k128.tm64tn64tk64.fullout.cpp \
	  mxgemm.fp8.m128n128k512.tm128tn128tk128.fullout.cpp \
	  mxgemm.fp8.m128n128k256.tm128tn128tk128.fullout.cpp \
	  mxgemm.fp8.m128n128k128.tm128tn128tk128.fullout.cpp \
	  mxgemm.fp4.singletile.tm64tn64tk64.fullout.cpp \
	  mxgemm.fp4.m64n64k128.tm64tn64tk64.fullout.cpp \
	  mxgemm.fp4.singletile.tm64tn64tk64.requant.cpp \
	  mxgemm.fp4.singletile.tm128tn128tk128.fullout.cpp \
	  mxgemm.fp4.m128n128k256.tm128tn128tk128.fullout.cpp \
	  mxgemm.fp4.singletile.tm128tn128tk128.requant.cpp \
	  mxgemm.fp8.m128n128k2048.tm128tn128tk128.fullout.cpp \
	  mxgemm.fp8.m128n128k5632.tm128tn128tk128.fullout.cpp \
	  mxgemm.fp6.m128n128k2048.tm128tn128tk128.fullout.cpp \
	  mxgemm.fp4.m128n128k2048.tm128tn128tk128.fullout.cpp \
	  mxgemm.fp4.m128n128k5632.tm128tn128tk128.fullout.cpp \
	  mxgemm.simt_contention.cpp \
	  requant.cpp
HOST_SRCS = host.cpp

MXGEMM_MU_OBJS := $(addsuffix .mu.o,$(basename $(MU_SRCS)))
$(MXGEMM_MU_OBJS): ../../lib/mxgemm/mxgemm_lib.hpp

include ../common.mk
