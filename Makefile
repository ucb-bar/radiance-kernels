all:
	$(MAKE) -C lib
	$(MAKE) -C kernels

clean:
	$(MAKE) -C lib clean
	$(MAKE) -C kernels clean

clean-all:
	$(MAKE) -C lib clean
	$(MAKE) -C kernels clean-all

