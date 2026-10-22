# llama_e2e: whole models on Radiance (FireSim)

One prefill chunk of a whole model runs on the device: every GEMM on the MX Gemmini (E4M3 MX operands), every row
op (norms, rope, softmax, activations, quantization) on the Muon SIMT cores, attention as one flash-attention phase.
A Python generator lays the model out in GPU memory and emits a *phase program*; `kernel.cpp` interprets it and
`host.cpp` (Rocket) steps the clusters, prints per-phase cycles / mesh utilization and checks the outputs against a
numpy emulation of the quantized pipeline and against fp32 references.

| `MODEL=` | model | weights | checks |
|---|---|---|---|
| `tinyllama` | TinyLlama-1.1B-Chat-v1.0, 22 layers, hd 64 | real (HF) | hidden states, logits, top-1 vs HF >= 85 % |
| `deepseek` | DeepSeek-R1-Distill-Qwen-1.5B, 28 layers, hd 128, q/k/v biases | real (HF) | same |
| `llama7b` | Llama-2-7B shapes, 32 layers | synthetic (filled on device) | finite logits only |
| `gemma2` | Gemma-2-2B shapes and math (hd 256, GeGLU, post-norms, soft-caps) | synthetic | finite logits only |
| `smolvla` | lerobot/smolvla_base: SigLIP vision, SmolLM2 prefix, action expert, 10 flow-matching steps | real (HF) | vision features, prefix, actions vs emulation and fp32 |
| `smoke`, `fasmoke`, `qwensmoke`, `dsmoke`, `gemmasmoke` | small random models for the device paths | random | every stage vs emulation |

Prefill shapes: M = 128 query tokens over a KV cache of 896 (SK = 1024) for the LLMs; SmolVLA as in lerobot
(3 images x 1024 patches, 205-token prefix, 50-step action chunk).

## Prerequisites

- RISC-V toolchain with `riscv64-unknown-elf-*` (chipyard's `.conda-env/riscv-tools`): `export RISCV=<it>` and put
  `$RISCV/bin` on `PATH`.
- `llvm/llvm-muon` (repository root), built from the `llvm/llvm-src` submodule with `scripts/llvm.sh`; it must
  support `-riscv-stack-word-stride`.  `lib/libmuonrt.a` is built from it automatically by `nightly.mk`.
- Python for the generators (`PY=`):
  - LLMs: numpy, torch, transformers 5.x, pyarrow (e.g. torch 2.14 + transformers 5.20).
  - SmolVLA: a separate env with `lerobot[smolvla]` 0.6.1 (it pins its own torch / transformers 5.5; keep it out of
    the LLM env).
- Hugging Face weights and the text sample, downloaded on first use (`HF_HOME`, ~10 GB for all models).
- GPU memory: the ELFs are 0.05 - 2 GB; the 4SM / 2SM / 1SM FireSim configs are the hashed-HBM Radiance configs
  (`MU_ADDR_HASH=1`, loaded with FireSim LoadMem).

## Build

```
cd kernels/nightly/llama_e2e
make MODEL=tinyllama SMS="1 2 4" PY=<llm python>            # build/{1,2,4}sm/llama_e2e_tinyllama.soc.elf
make MODEL=deepseek  SMS="1 2 4" PY=<llm python>
make MODEL=smolvla   SMS="1 2 4" PY=<smolvla python>
make MODEL=deepseek TAG=l1 ARGS="--layers 1" SMS=1 PY=...    # variant: llama_e2e_deepseek_l1.soc.elf
```

Useful generator options (`ARGS=`): `--layers N`, `--check-l0` (report-only layer-0 stage checks),
`--fa-exact {0,1}`, `--lm-chunks N`, `--spread S` (random models: MX scales that vary like real data),
`--fa-dump` with `DEFS=-DLL_FA_DUMP=1` (dump and check one attention tile stage by stage).
`DEFS=-DLL_FA_PROF=1` adds per-role cycle totals of the attention phase (`fa_prof` lines).

The host prints `PASS` / `FAIL`; tolerances are relative Frobenius errors from FireSim runs plus margin (the fp8
format alone puts the emulation 5 - 10 % from fp32 on the LLM hidden states and 38 / 23 / 7.7 % on SmolVLA's vision /
prefix / actions; the MX Gemmini's reduced-precision accumulation adds a few percent).

## Device pitfalls found while bringing these up

- **Runtime-indexed local arrays alias across lanes.**  Muon's stack interleaves lanes word by word
  (`-riscv-stack-word-stride=16`), but a computed element address does not get the lane interleave.  Keep per-lane
  state in registers or explicit per-lane DRAM addresses.
- **At most 8 integer arguments for non-inlined functions.**  Outgoing stack arguments are stored at unscaled offsets
  while the callee reads them at scaled ones: the 10th argument and later arrive stale.  Use `always_inline`.
- **`fexp.h` wraps for x below about -355** (no underflow check in the fixed-point path): clamp its argument.
- **Per-cluster state must not be a shared global** (each cluster's GEMM thread kept its own accumulator half).
- **The Rocket host has no FPU enabled** (`soc/start.S`): host code must stay integer-only.
- **Random-weight tests miss magnitude bugs:** they pass even with a broken softmax max (shift invariance) or wrong
  scales; use `--spread` or real weights.
