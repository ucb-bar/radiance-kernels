"""SmolVLA vision encoder, image 0: per-layer error of the quantized emulation vs fp32, each layer starting from the
fp32 input (so errors do not compound), with the largest activations of every GEMM input."""
import sys
import numpy as np
import gen_vla as v
from gen_data import mx_quant, mx_dequant

R = v.load_reference()
sd, P = R["sd"], R["P"] + "vlm.model.vision_model."
w = lambda n: sd[P + n]
f, q = v.Ops(False), v.Ops(True)
img = R["images"][0]
patches = img.reshape(3, 32, 16, 32, 16).transpose(1, 3, 0, 2, 4).reshape(1024, 768)
h = f.gemm(patches, w("embeddings.patch_embedding.weight").reshape(768, 768)) + \
    w("embeddings.patch_embedding.bias")[None, :] + w("embeddings.position_embedding.weight")
mask = np.ones((1024, 1024), bool)


def mxerr(x):
    xd = mx_dequant(*mx_quant(x))
    return np.linalg.norm(xd - x) / np.linalg.norm(x)


def rel(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


for l in range(R["n_vision"]):
    p = f"encoder.layers.{l}."
    res = {}
    for name, o in (("f", f), ("q", q)):
        x = v.lnorm(o, o.bf(h), w(p + "layer_norm1.weight"), w(p + "layer_norm1.bias"), R["eps_v"])
        qkv = o.gemm(x, np.concatenate([w(p + f"self_attn.{n}_proj.weight") for n in "qkv"], 0))
        qkv = o.bf(qkv + o.bf(np.concatenate([w(p + f"self_attn.{n}_proj.bias") for n in "qkv"]))[None, :])
        Q, K, V = (qkv[:, i * 768:(i + 1) * 768].reshape(1024, 12, 64) for i in range(3))
        att = v.attention(o, Q, K, V, mask, 1 / 8, fa=True)
        y = o.gemm(att, w(p + "self_attn.out_proj.weight"))
        hmid = o.bf(o.bf(h) + (y + o.bf(w(p + "self_attn.out_proj.bias"))))
        x2 = v.lnorm(o, hmid, w(p + "layer_norm2.weight"), w(p + "layer_norm2.bias"), R["eps_v"])
        f1 = o.gemm(x2, w(p + "mlp.fc1.weight"))
        a = o.gelu((f1 + o.bf(w(p + "mlp.fc1.bias"))).astype(np.float32))
        f2 = o.gemm(a, w(p + "mlp.fc2.weight"))
        res[name] = dict(x=x, qkv=qkv, att=att, y=y, x2=x2, a=a, f2=f2, out=hmid + f2 + w(p + "mlp.fc2.bias"))
    F, Qd = res["f"], res["q"]
    print(f"layer {l:2d}: |h| max {np.abs(h).max():8.1f}  err " + "  ".join(
        f"{k}={rel(Qd[k], F[k]) * 100:5.1f}%" for k in ("qkv", "att", "y", "f2", "out")) +
        f"   MX err of inputs: x={mxerr(F['x']) * 100:.1f}% att={mxerr(F['att']) * 100:.1f}% x2={mxerr(F['x2']) * 100:.1f}% "
        f"a={mxerr(F['a']) * 100:.1f}%  |a| max {np.abs(F['a']).max():.0f}", flush=True)
    h = F["out"]
