# minimal make: build fused.elf from rv64 stub + muon.elf (rv32)
CROSS64 ?= riscv64-unknown-elf
CC      ?= $(CROSS64)-gcc
AS      ?= $(CROSS64)-as
LD      ?= $(CROSS64)-ld
MUON_ELF ?= muon.elf
FUSED_DIR ?= ./tmp

all: fused.elf

fused.elf: main.c start.S fuse_rv32_into_rv64.sh muon.elf
	./fuse_rv32_into_rv64.sh

fuse-temp:
	@mkdir -p $(FUSED_DIR)
	@tmp="$(abspath $(FUSED_DIR))/fused_$$(basename $(MUON_ELF))"; \
	RV32_ELF="$(MUON_ELF)" OUT="$${tmp}" ./fuse_rv32_into_rv64.sh >/dev/null; \
	echo "$${tmp}"

clean:
	rm -f *.o fused.elf

.PHONY: all clean fuse-temp
