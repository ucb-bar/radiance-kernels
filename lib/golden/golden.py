#!/usr/bin/env python3
"""golden.py -- the single reference-golden tool for radiance-kernels.

One module that every kernel's data generator uses, for any engine mix:

  * quantization primitives (bf16 / fp8 e4m3 / fp6 / fp4 e2m1, block e8m0 scales),
  * reference implementations of the SIMT ops (RMSNorm, RoPE, GeGLU/gelu-tanh,
    LayerNorm, soft-cap, block-quantize),
  * mx_matmul(), the MX-Gemmini matmul in HARDWARE semantics (drives the
    mx_golden binary sitting next to this file; builds it on demand),
  * emit helpers for writing the `data` / `*.h` operand+golden headers.

A SIMT-only kernel builds its golden from the op references. A matmul kernel uses
mx_matmul(). A fused kernel composes the two, e.g.
    C = golden.mx_matmul(golden.fp8_codes(golden.rmsnorm(X, gamma, eps)), W, ...)

Why the matmul is a compiled backend and not numpy: the MX-Gemmini mesh accumulates
through a 16-deep systolic array with a per-column accumulate-precision schedule and
one e8m0 scale per 32-element K group. A plain numpy/torch matmul disagrees with the
silicon on ~90% of elements, so mx_golden (verified bit-exact against real spike
libgemmini) is the reference. Everything the SIMT cores do is ordinary IEEE float
math, reproduced directly here.

The fp8 e4m3 encoder exists in two byte-equivalent forms because two device kernels
compute it by different instruction paths: fp8_code() (exponent-field form) and
fp8_code_rshift() (round-shift form). Each generator uses the one its kernel matches.
"""
import math
import pathlib
import subprocess

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
MX_GOLDEN = _HERE / "mx_golden"
GROUP = 32
FMT_CODE = {"fp8": 0, "fp6": 1, "fp4": 2}


# ============================ bf16 ============================
def bf16_trunc(x):
    """float32 -> bf16(truncate) -> float32."""
    u = x.astype(np.float32).view(np.uint32)
    return ((u >> 16) << 16).view(np.float32)


def f_to_bf16(x):
    """float32 -> uint16 bf16 (truncate)."""
    return (x.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def bf16_to_f32(u16):
    """uint16 bf16 -> float32."""
    return (u16.astype(np.uint32) << 16).view(np.float32)


def f32_bits(x):
    return int(np.float32(x).view(np.uint32))


def bits_f32(u):
    return float(np.uint32(u & 0xFFFFFFFF).view(np.float32))


def round_bf16_bits(u):
    """round-to-nearest-even of fp32 bits -> bf16-truncated fp32 bits."""
    lsb = (u >> 16) & 1
    return (u + 0x7FFF + lsb) & 0xFFFF0000 & 0xFFFFFFFF


def round_half_to_even(x):
    fl = math.floor(x)
    frac = x - fl
    if frac < 0.5:
        return int(fl)
    if frac > 0.5:
        return int(fl) + 1
    return int(fl) + 1 if (int(fl) & 1) else int(fl)


# ============================ fp8 e4m3 ============================
def fp8_code(v):
    """fp8 e4m3 encode, RNE, subnormals flush to 0. Exponent taken from the float32
    exponent field (lets the device kernel avoid libm log2f). Matches the fused
    prologue/epilogue kernels."""
    vf = np.float32(v)
    if float(vf) == 0.0 or not math.isfinite(float(vf)):
        return 0
    ubits = int(vf.view(np.uint32))
    s = (ubits >> 31) & 1
    av = np.float32(abs(float(vf)))
    expf = (int(av.view(np.uint32)) >> 23) & 0xFF
    bias, emin, emax = 7, -6, 8
    if expf == 0:
        return 0
    E = expf - 127
    if E < emin:
        return 0
    if E > emax:
        E_used, mant = emax, 6
    else:
        E_used = E
        base = np.float32(np.uint32((E_used + 127) << 23).view(np.float32))
        delta = np.float32(base / np.float32(8.0))
        k = round_half_to_even(float(np.float32((av - base) / delta)))
        if k >= 8:
            E_used += 1
            k = 0
            if E_used > emax:
                E_used, k = emax, 6
        else:
            hi = 6 if E_used == emax else 7
            k = min(max(k, 0), hi)
        mant = k
    return ((s << 7) | (((E_used + bias) & 0xF) << 3) | (mant & 0x7)) & 0xFF


def _round_shift(v, sh):
    """round-half-to-even of v >> sh (sh > 0)."""
    mask = (1 << sh) - 1
    low = v & mask
    half = 1 << (sh - 1)
    q = v >> sh
    if low > half or (low == half and (q & 1)):
        q += 1
    return q


def fp8_code_rshift(xf):
    """fp8 e4m3 encode via integer round-shift, RNE, saturate to +/-448. Matches the
    block-quantize kernel (simt_quant)."""
    x = f32_bits(xf)
    sign = (x >> 31) & 1
    absx = x & 0x7FFFFFFF
    s = (sign << 7) & 0xFF
    if absx == 0:
        return s
    axf = bits_f32(absx)
    if axf >= 448.0:
        return s | 0x7E
    e = ((absx >> 23) & 0xFF) - 127
    man = absx & 0x7FFFFF
    sig = (1 << 23) | man
    if e < -6:
        q = _round_shift(sig, 14 - e)
        out = ((1 << 3) | (q - 8)) if q >= 8 else q
    else:
        r = _round_shift(sig, 20)
        E = e
        if r == 16:
            r = 8
            E += 1
        expf = E + 7
        mant = r - 8
        if expf > 15 or (expf == 15 and mant == 7):
            return s | 0x7E
        out = ((expf << 3) | mant) & 0xFF
    return s | out


def block_exp(amax_bits):
    """e = ceil(log2(amax / 448)) via exact integer exponent extraction (e8m0 block scale)."""
    E = ((amax_bits >> 23) & 0xFF) - 127
    man = amax_bits & 0x7FFFFF
    return (E - 8) + (1 if man > 0x600000 else 0)


def pow2_neg_e(e):
    """2^-e as an exact fp32 from its bit pattern."""
    return bits_f32(((127 - e) & 0xFFFFFFFF) << 23)


# ============================ random operands / packing ============================
def rand_fp8(rng, n):
    """Random fp8 e4m3 codes with the exponent field in [0, 7] (<= bias 7, i.e. exponent
    <= 0), so |value| < 2 (max ~1.875). Bounded on purpose: unrestricted fp8 (|v| up to 448)
    saturates the block accumulators to +/-inf, giving a golden any broken kernel matches."""
    exp = rng.integers(0, 8, size=n, dtype=np.uint8)
    mant = rng.integers(0, 8, size=n, dtype=np.uint8)
    sign = rng.integers(0, 2, size=n, dtype=np.uint8)
    return ((sign << 7) | (exp << 3) | mant).astype(np.uint8)


def rand_fp4(rng, n):
    """Random fp4 e2m1 nibbles, |value| <= 1.5 (exp field <= 1)."""
    exp = rng.integers(0, 2, size=n, dtype=np.uint8)
    mant = rng.integers(0, 2, size=n, dtype=np.uint8)
    sign = rng.integers(0, 2, size=n, dtype=np.uint8)
    return ((sign << 3) | (exp << 1) | mant).astype(np.uint8)


def pack_axis0(x):
    """[R,C] nibbles -> [R/2,C] bytes; even row low nibble."""
    return ((x[1::2, :].astype(np.uint8) << 4) | (x[0::2, :].astype(np.uint8) & 0xF)).astype(np.uint8)


def pack_axis1(x):
    """[R,C] nibbles -> [R,C/2] bytes; even col low nibble."""
    return ((x[:, 1::2].astype(np.uint8) << 4) | (x[:, 0::2].astype(np.uint8) & 0xF)).astype(np.uint8)


# ============================ SIMT op references (fp32) ============================
def rmsnorm(X, gamma, eps):
    """Row RMSNorm matching the device float32 path (scalar accumulation order)."""
    M, K = X.shape
    out = np.empty((M, K), dtype=np.float32)
    for m in range(M):
        row = X[m].astype(np.float32)
        ss = np.float32(0.0)
        for k in range(K):
            ss = np.float32(ss + np.float32(row[k] * row[k]))
        mean = np.float32(ss / np.float32(K))
        rms = np.float32(1.0) / np.float32(math.sqrt(float(np.float32(mean + np.float32(eps)))))
        for k in range(K):
            out[m, k] = np.float32(np.float32(row[k] * rms) * gamma[k])
    return out


def rope_caches(M, N, theta=10000.0):
    """HF Llama-style cos/sin caches, duplicated across the two halves. Returns
    (cos_half[M,N/2], sin_half[M,N/2], cos_full[M,N], sin_full[M,N])."""
    half = N // 2
    inv_freq = theta ** (-(np.arange(0, half, dtype=np.float64) * 2.0 / N))
    ang = np.arange(M, dtype=np.float64)[:, None] * inv_freq[None, :]
    cos_h = np.cos(ang).astype(np.float32)
    sin_h = np.sin(ang).astype(np.float32)
    cos_full = np.concatenate([cos_h, cos_h], axis=1).astype(np.float32)
    sin_full = np.concatenate([sin_h, sin_h], axis=1).astype(np.float32)
    return cos_h, sin_h, cos_full, sin_full


def rope_apply(Cf, cos_h, sin_h):
    """Apply RoPE (rotate-half) to Cf[M,N] float32, returns rotated float32 [M,N]."""
    half = Cf.shape[1] // 2
    c0, c1 = Cf[:, :half], Cf[:, half:]
    o = np.empty_like(Cf)
    o[:, :half] = (c0 * cos_h - c1 * sin_h).astype(np.float32)
    o[:, half:] = (c1 * cos_h + c0 * sin_h).astype(np.float32)
    return o


def gelu_tanh(x):
    """gelu_pytorch_tanh (the Gemma-2 gate activation), computed in float64."""
    xf = x.astype(np.float64)
    inner = 0.7978845608028654 * (xf + 0.044715 * xf ** 3)
    return 0.5 * xf * (1.0 + np.tanh(inner))


def layernorm(x, gamma, beta, eps):
    """torch LayerNorm (biased variance), float32."""
    xf = x.astype(np.float32)
    mean = xf.mean(axis=-1, keepdims=True, dtype=np.float32)
    var = ((xf - mean) ** 2).mean(axis=-1, keepdims=True, dtype=np.float32)
    xhat = (xf - mean) / np.sqrt(var + np.float32(eps))
    return (xhat * gamma + beta).astype(np.float32)


def softcap(x, cap):
    """c * tanh(x / c) logit soft-cap, float32."""
    xf = x.astype(np.float64)
    return (cap * np.tanh(xf / cap)).astype(np.float32)


# ============================ MX-Gemmini matmul (hardware semantics) ============================
def ensure_mx_golden():
    if not MX_GOLDEN.exists():
        subprocess.run(["make", "-C", str(_HERE)], check=True)


def mx_matmul(A_bytes, B_bytes, SA, SB, M, N, K, fmt="fp8",
              out_fmt=None, tmpdir=None):
    """Run the MX-Gemmini matmul golden. A_bytes/B_bytes are the operand byte arrays
    (fp8: [M,K]/[K,N] uint8; sub-byte: nibble-packed A along M, B along N). SA/SB are
    e8m0 scale codes [K/32,M] / [K/32,N]. Returns C bf16 uint16 [M,N] when out_fmt is
    None; when out_fmt is set (requant), returns (C_codes[M,N] uint8, C_scales[M,N/32] uint8)."""
    ensure_mx_golden()
    tmp = pathlib.Path(tmpdir) if tmpdir else (_HERE / "_gen" / f"{fmt}_{M}_{N}_{K}")
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "A.bin").write_bytes(np.ascontiguousarray(A_bytes).tobytes())
    (tmp / "B.bin").write_bytes(np.ascontiguousarray(B_bytes).tobytes())
    (tmp / "SA.bin").write_bytes(np.ascontiguousarray(SA).tobytes())
    (tmp / "SB.bin").write_bytes(np.ascontiguousarray(SB).tobytes())
    env = {"PATH": "/usr/bin:/bin"}
    cbin = tmp / "C.bin"
    scbin = tmp / "C_scales.bin"
    if out_fmt is not None:
        env["MX_OUT_FMT"] = str(out_fmt)
        env["MX_SCALES_OUT"] = str(scbin)
    subprocess.run(
        [str(MX_GOLDEN), str(M), str(N), str(K), str(tmp / "A.bin"), str(tmp / "B.bin"),
         str(tmp / "SA.bin"), str(tmp / "SB.bin"), str(cbin), str(FMT_CODE[fmt])],
        check=True, env=env)
    if out_fmt is None:
        return np.frombuffer(cbin.read_bytes(), dtype="<u2")[: M * N].reshape(M, N)
    codes = np.frombuffer(cbin.read_bytes(), dtype=np.uint8)[: M * N].reshape(M, N)
    scales = np.frombuffer(scbin.read_bytes(), dtype=np.uint8)[: M * (N // GROUP)]
    return codes, scales


# ============================ emit helpers ============================
def emit(f, ctype, name, dims, arr, braces=False):
    """Emit a C array of hex values. braces=True nests per-row {..} (matmul headers)."""
    w = {"uint8_t": 2, "uint16_t": 4, "uint32_t": 8}[ctype]
    if braces:
        rows = ["    { " + ", ".join(f"0x{v:0{w}x}" for v in row) + " }" for row in arr]
    else:
        rows = ["    " + ", ".join(f"0x{v:0{w}x}" for v in row) for row in np.atleast_2d(arr)]
    f.write(f"static const {ctype} {name}{dims} = {{\n")
    f.write(",\n".join(rows))
    f.write("\n};\n")


def emit_f32(f, name, arr, dims=None, ncol=1, trailing_comma=False, storage="__global float"):
    """Emit a C float array. dims defaults to [size]; ncol>1 packs that many values per
    indented row; trailing_comma appends a comma after the last value."""
    flat = np.asarray(arr).astype(np.float32).reshape(-1)
    d = dims if dims is not None else f"[{flat.size}]"
    if ncol > 1:
        body = ",\n".join("    " + ", ".join(f"{v:.9e}f" for v in flat[i:i + ncol])
                          for i in range(0, flat.size, ncol))
    else:
        body = ",\n".join(f"{v:.9e}f" for v in flat)
    if trailing_comma:
        body += ","
    f.write(f"{storage} {name}{d} = {{\n{body}\n}};\n")


def emit_u32(f, name, arr, dims=None, storage="__global uint32_t"):
    flat = np.asarray(arr).reshape(-1)
    d = dims if dims is not None else f"[{flat.size}]"
    f.write(f"{storage} {name}{d} = {{\n")
    f.write(",\n".join(f"0x{int(v) & 0xFFFFFFFF:08x}u" for v in flat))
    f.write("\n};\n")
