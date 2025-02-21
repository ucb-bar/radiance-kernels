# Find all subdirectories containing a Makefile
SUBDIRS := $(shell find . -mindepth 1 -maxdepth 1 -type d -exec test -e {}/Makefile \; -print)

.PHONY: all $(SUBDIRS) clean clean-all

# Default target: run make in all subdirectories
all: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@

# Clean target: run make clean in all subdirectories
clean:
	for dir in $(SUBDIRS); do $(MAKE) -C $$dir clean; done

clean-all:
	for dir in $(SUBDIRS); do $(MAKE) -C $$dir clean-all; done
