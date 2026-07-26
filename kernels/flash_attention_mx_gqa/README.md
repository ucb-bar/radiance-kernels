# flash_attention_mx_gqa

MXFP8 **flash attention** with grouped-query attention and a causal mask —
TinyLlama prefill shape (`Sq = 64`, `Sk = 256`, `d = 64`, 8 query heads : 2 KV
heads).

Per query head it runs a full streaming online-softmax pass over its KV head's key
blocks, reusing the same K/V across the query heads that share it (GQA). Each
key block does two MX-Gemmini GEMMs — `S = Q@K^T` then `O = P@V` — with the
softmax and the P-tile requantization to fp8 in between; causal blocks fully above
the diagonal are skipped. At these dims (Sq=Bk=d=64) both GEMMs are square 64x64; the mesh core's non-square
path (needed only for the Gemma d=256 variant) is inherited but not exercised here. O is
checked offline via the RTL/SQLite-dump host flow -- the mesh-PV FP path is platform-blocked
for on-device verify on the functional model.
