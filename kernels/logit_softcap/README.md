# logit_softcap

Gemma-2 **final-logit soft-capping** on the LM-head output.

```
out[i] = cap * tanh(logits[i] / cap),   cap = 30.0
```

Confirmed from `google/gemma-2-2b-it` config (`final_logit_softcapping: 30.0`). A
plain SIMT elementwise epilogue — no reductions or cross-lane comms. It also
validates the `tanh` primitive reused by the attention-score soft-cap (`cap = 50`)
in `flash_attention_mx_gemma`. Verdict via `tohost` (`0 = pass`).
