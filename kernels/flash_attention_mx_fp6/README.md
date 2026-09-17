# flash_attention_mx_fp6

The **live fp6 activation-quantizer chain**: a runtime SIMT fp6-e3m2-LUT quantizer
that feeds the MX mesh.

```
X (bf16 activation)
  --[SIMT: bf16 -> fp6 code -> nearest-LUT-entry finder -> 4-bit index]-->
A_in (fp6 indices) + A_lut (16-entry fp6 LUT)
  --[fp6 mxgemm, static fp6 weight B]--> C (bf16)
```

This is the piece that was missing on-device: there was a runtime `quantize_fp4`
but no runtime **fp6-LUT activation quantizer**. The two device functions
(`dev_bf16_to_fp6_code` + `dev_fp6_nearest_finder`) mirror the RTL
`BF16ScaleRoundToTiny` (fp6 path) and `FP6E3M2NearestFinder`, and are validated
bit-exact against the Python golden over all 65280 finite bf16 patterns
(`tohost 0`).
