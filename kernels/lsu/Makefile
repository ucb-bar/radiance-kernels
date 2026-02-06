# Auto-generated group Makefile
SUBDIRS := $(shell find . -mindepth 1 -maxdepth 1 -type d -exec test -e {}/Makefile \; -print)

.PHONY: all $(SUBDIRS) clean clean-all

all: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@

clean:
	for dir in $(SUBDIRS); do $(MAKE) -C $$dir clean; done

clean-all:
	for dir in $(SUBDIRS); do $(MAKE) -C $$dir clean-all; done
