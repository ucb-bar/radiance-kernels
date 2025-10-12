# minimal make: build fused.elf from rv64 stub + muon.elf (rv32)
CROSS64 ?= riscv64-unknown-elf
CC      ?= $(CROSS64)-gcc
AS      ?= $(CROSS64)-as
LD      ?= $(CROSS64)-ld

all: fused.elf

fused.elf: main.c start.S fuse_rv32_into_rv64.sh muon.elf
	./fuse_rv32_into_rv64.sh

clean:
	rm -f *.o fused.elf

.PHONY: all clean

