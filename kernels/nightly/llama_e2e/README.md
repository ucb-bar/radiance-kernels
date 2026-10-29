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
`--fa-exact {0,1}`, `--fa-impl {standalone,port}` (attention: `fa_sa.h` / `fa_sa256.h`, the standalone flash-attention
kernel with O fused into oproj's operand, the default for every model),
`--lm-chunks N`, `--spread S` (random models: MX scales that vary like real data),
`--fa-dump` with `DEFS=-DLL_FA_DUMP=1` (dump and check one attention tile stage by stage).
`DEFS=-DLL_FA_PROF=1` adds per-role cycle totals of the attention phase (`fa_prof` lines).

The host prints `PASS` / `FAIL`; tolerances are relative Frobenius errors from FireSim runs plus margin (the fp8
format alone puts the emulation 5 - 10 % from fp32 on the LLM hidden states and 38 / 23 / 7.7 % on SmolVLA's vision /
prefix / actions; the MX Gemmini's reduced-precision accumulation adds a few percent).  The gates are calibrated on
the full models; a truncated SmolVLA (`--layers`) lands farther from the emulation on the actions (20 % at 2 layers), so
its actions gates are 30 % instead of 18 %.

## Decode (`DECODE=1`)

Batch-16 decode after the prefill's context: `make DECODE=1 MODEL=<model> SMS="1 2 4" PY=...` builds
`llama_e2e_dec_<model>.soc.elf`.

- LLMs (`gen_decode.py`, `tinyllama`, `deepseek`, `llama7b`, `gemma2`, `smoke`): 16 sequences (16 text windows; window 0
  is the prefill test's text), each with its own KV cache of the first 1024 tokens (from the HF model), decode
  `--steps` (4) tokens: per step every layer runs on the 16 rows, the new key / value is appended to each sequence's
  cache (DROPE phase: V's 32-key group is requantized from this layer's earlier steps), attention runs as batched GEMMs
  over (sequence, KV head) with the query heads of a KV head as rows.  Checks: next-token logits of every step vs the HF
  model (top-1 >= 85 %, teacher-forced with the text) and vs a numpy emulation of the pipeline.
- SmolVLA (`gen_vla_dec.py`, `MODEL=smolvla`): the action expert alone for 16 observations (seeds 0..15: own images,
  state, noise), over each sample's prefix caches and cross-attention K' / V' (the prefill's outputs, from the
  quantized emulation): the 10 flow-matching steps with every dense GEMM over all 1024 action rows.  Checks: actions vs
  the emulation and vs lerobot (fp32).

The host prints `decode: batch= steps= cycles_per_step=` next to the usual per-phase table.

## Offline checks (`LL_OFFLINE`, default off)

The Rocket host has no FPU and reads GPU DRAM slowly: checking a 152k-word vocabulary on it took longer than the model.
So after the per-phase report the host writes every checked device buffer to files on the simulation host
(`lldump.txt` + `lldump_<i>.bin` in the FireSim sim slot, through fesvr's `openat` / `write` proxy) and prints
`OFFLINE_DUMP ok`; `check_offline.py <dump dir> <soc.elf>` runs the same checks on x86 (bit-identical arithmetic to the
host's `nh_compare_bf16`, golden data from the `.radiance.elf` next to the `.soc.elf`).  `run_all_profiling.py` moves
the dump to `<log>.dump/`, runs the checker and records its PASS / FAIL.  If the host cannot create a file it checks on
the host as before.

Off by default (`DEFS=-DLL_OFFLINE=1` enables it): on FireSim fesvr reads target memory over serial TSI, 1 KiB per
round trip and one round trip per tsibridge step (`+fesvr-step-size`, default 2,004,765 target cycles), so the dump runs
at well under 2 KB/s; the first 1 MiB write of the l1 suite did not finish in 10 minutes.  It works under spike.

## Regression suite (`make regress`, `regress.py`)

Every model shortened to 1 layer (SmolVLA 2: one self- and one cross-attention expert layer) for prefill, decode and
speculative verify (Llama-7B, T = 4), at 1 / 2 / 4 SM; minutes per run instead of hours.

```
make regress PY=<llm python> REGRESS_VLA_PY=<smolvla python>   # build every variant
python3 regress.py jobs                                        # -> /scratch/nicolas/firesim-rundir/regress-jobs.yaml
python3 regress.py report [--since RUN_ID]                     # latest result per variant, scaled to the full model
```

`scale_report.py` scales a short run's per-op cycles by layer count (per stack for SmolVLA) to the full model; against
the full runs it is within ~1 % (TinyLlama, Llama-7B, Gemma-2 at 1 / 2 / 4 SM: `scale_report.py --validate SHORT:FULL`).
The real-weight variants add `--check-l0` (report-only layer-0 stage checks) and `--exact-emu`.

## Exact emulation (`--exact-emu`, `emu_exact.py`, `muon_fp.py`)

`emulate()` approximates the device (float GEMMs, numpy reductions, an ideal exp, softmax normalized before PV), so the
checks only gate gross failures.  `--exact-emu` (prefill, `fa_sa.h` models, not Gemma-2) follows the device instead:
GEMMs through `mx_golden` (parallel over output columns), the PRENORM sums in the kernel's lane order (2 / 4 SM; at 1 SM
the norm after oproj / down is fused into the GEMM and sums in another order), and `fa_sa.h`'s softmax with
`muon_fp.fexp_bf16`, a bit-level model of the fpEX unit (it reproduces the wrap below x ~ -355).  The goldens it gives
match the device bit for bit: TinyLlama and DeepSeek at 1 layer, every `--check-l0` stage of layer 0, the hidden state and
the logits, at 1 / 2 / 4 SM (runs 689 - 698).  The gates stay at their old values, so they pass either way.

## Cyclotron bench (`smbench/`)

`smbench/` runs the SOFTMAX phases (SmolVLA's in kernel_vla.cpp, the LLM one in kernel.cpp) on cyclotron, the
Radiance performance model, without FireSim: it extracts the phases from the sources, compares their output bytes
with the previous versions and times them (`smbench/README.md`).  Cyclotron ranked the variants of the 2026-10-29
softmax rewrite and proved it bit-identical; its speed-ups ran high (SmolVLA 2.2 - 2.5x there, 1.4 - 1.65x on FireSim).

## Known issues

- **SmolVLA prefill at 2 SM was unstable.**  The plain build (`TAG=vsa`, fa_sa_vla.h) hangs deterministically on
  FireSim at cycle 10,308,501,632 (runs 649 / 676): phase 1025, the action expert's SOFTMAX in denoise step 2, layer 5,
  with cluster 0 not finishing.  Builds with per-warp progress probes in that phase complete and pass (runs 679 / 684,
  2.71 G model cycles), so it is a timing race, not a logic error.  1 and 4 SM pass.  The same signature (a cluster
  never posts DONE, at one exact cycle; probe builds pass) hung Gemma-2 4 SM decode (570 / 595) and Llama-7B T = 4 at
  1 SM (628).  Suspected: the phase epilogue, where all 8 warps issued `fence` while their siblings were still storing
  (the pattern behind the RTL's SFU / L0d flush fixes).  Fix candidate, unconfirmed: `LL_PH_FENCE_PER_CORE` (default
  1): barrier, one fence per core, barrier (as `nightly_kernel_end`); `=0` restores the old epilogue for an A/B.  The
  A/B (2026-10-28) was inconclusive: both passed (old epilogue run 701, fix 703, with the new expert softmax), so the
  race stayed hidden; the fix is kept as the default.

## Device pitfalls found while bringing these up

- **Runtime-indexed local arrays alias across lanes.**  Muon's stack interleaves lanes word by word
  (`-riscv-stack-word-stride=16`), but a computed element address does not get the lane interleave.  Keep per-lane
  state in registers or explicit per-lane DRAM addresses.
- **At most 8 integer arguments for non-inlined functions.**  Outgoing stack arguments are stored at unscaled offsets
  while the callee reads them at scaled ones: the 10th argument and later arrive stale.  Use `always_inline`.
- **Sub-word stores to DRAM are unreliable.**  A lane's byte store into a shared word lost bytes (the K^T cache append),
  and the Q scale bytes of Gemma-2's 4-cluster decode hung the DROPE phase.  Stage bytes in SMEM and store whole words.
- **clang types `_Float16` as IEEE half; only the hardware runs it as bf16.**  A bf16 constant like `bf(0xFF80)` (-inf)
  is a NaN to the optimizer, so with a fully unrolled loop it folded every `x > m` to false: fa_sa.h's exact-max
  pre-pass stored -inf as each row's max (DeepSeek yattn 100 % off, runs 664 / 668).  Pass such constants through
  `opaque()`.
- **`fexp.h` wraps for x below about -355** (no underflow check in the fixed-point path): clamp its argument.
- **Per-cluster state must not be a shared global** (each cluster's GEMM thread kept its own accumulator half).
- **The Rocket host has no FPU enabled** (`soc/start.S`): host code must stay integer-only.
- **Random-weight tests miss magnitude bugs:** they pass even with a broken softmax max (shift invariance) or wrong
  scales; use `--spread` or real weights.
