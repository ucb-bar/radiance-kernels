# simt_quant

A SIMT **MX-fp8 (e4m3) block quantizer** — a software replacement for the broken
tapeout-330 hardware requantizer.

For each contiguous 32-element block along `N`:

```
amax  = max |x| in block
e     = ceil(log2(amax / 448))              (FP8_MAX = 448)
scale = 2^e,  e8m0 code = e + 127  (clamped [0, 254])
out_fp8[i] = quantize_e4m3(x[i] * 2^-e)     (round-to-nearest-even)
```

Input is a bf16 activation tile `X[64][512]`; output is fp8 codes packed 4/uint32
plus one e8m0 scale per block. One thread == one block, and a block is exactly 8
word-aligned output words, so no two threads share an output word (race-free, no
sub-word RMW). Bit-exact vs a numpy golden using identical arithmetic.
