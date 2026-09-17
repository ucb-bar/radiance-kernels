SUBDIRS := address_space \
	special_fn \
	ubench \
	spin \
	launch

.PHONY: all $(SUBDIRS) clean clean-all data

# Default target: run make in all subdirectories
all: $(SUBDIRS)

# Deterministically regenerate every kernel's committed data (see regen_data.sh).
# Needs python3 with numpy (+ torch for the flash/fused generators) and the
# lib/mxgemmini submodule checked out. Override the interpreter with PYTHON=...
data:
	PYTHON=$(if $(PYTHON),$(PYTHON),python3) ./regen_data.sh

$(SUBDIRS):
	$(MAKE) -C $@

# Clean target: run make clean in all subdirectories
clean:
	for dir in $(SUBDIRS); do $(MAKE) -C $$dir clean; done

clean-all:
	for dir in $(SUBDIRS); do $(MAKE) -C $$dir clean-all; done
