This directory contains scripts to fuse an RV64 host binary with an RV32 muon
binary into a single ELF.

**Instructions:**
1. Compile muon binary and copy it under this directory as `muon.elf`
2. Edit `main.c` to be the host binary source (TODO: modify the build process to
   have it take object files instead)
3. Run `make`
4. Profit (output is `fused.elf`)

Internally, this looks at all the sections in `muon.elf`, generates a linker
script that has all of these sections but offset their addresses by
`0x1_0000_0000`, and then links each section in the muon ELF by treating them as
raw binary data.

Disclaimer: this whole directory except this README is all agent generated 🤖✨

