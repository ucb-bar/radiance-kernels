SUBDIRS := address_space \
	special_fn \
	ubench \
	spin \
	launch

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
