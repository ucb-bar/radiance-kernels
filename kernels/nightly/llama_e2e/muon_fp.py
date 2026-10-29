"""Bit-exact numpy models of Muon floating-point units the emulation needs (llama_e2e gen_*.py).

fexp_bf16: Muon's fexp.h on bf16, the fpEX unit in radiance (muon/backend/fp/fpEX/fpex/fpex.scala, FPType.BF16T), stage
by stage:
  0  specials: NaN -> NaN, zero / subnormal -> 1, -inf -> 0, +inf or e^x > max -> +inf (no underflow check)
  1  x -> Q9.12 fixed point (21-bit signed, truncating: |x| >= 256 wraps; with stage 6's 10-bit exponent this is why
     fexp(x) for x below about -355 returns garbage instead of 0)
  2  x / ln 2 = x * 5909 (1/ln2 in Q2.12) >> 12, split into k (integer part) and r (12 fraction bits)
  3-5  2^r: 32-entry table round(2^(i/32) 2^16), linear interpolation on r's low 7 bits
  6  2^k * 2^r -> bf16, round to nearest even (HardFloat RoundRawFNToRecFN), the exponent k + 256 held in 10 bits
"""
import numpy as np

_LUT = np.array([int(np.floor(2.0 ** (i / 32) * 65536 + 0.5)) for i in range(32)], np.int64)   # Scala math.round
_LUT_TOP = (1 << 17) - 1                                 # lutTopEndpoint: the interpolation end past the last entry


def _wrap(v, bits):
    """two's-complement truncation of an int64 array to `bits` bits"""
    m = (1 << bits) - 1
    v = v & m
    return np.where(v >= (1 << (bits - 1)), v - (1 << bits), v)


def fexp_bf16(x, neg=False):
    """Muon fexp.h: e^x (e^-x with neg) of bf16 bit patterns -> bf16 bit patterns (uint16), bit-exact with fpEX."""
    b = np.asarray(x, np.uint16).astype(np.int64)
    s_in, e, f = b >> 15, (b >> 7) & 0xFF, b & 0x7F
    s = s_in ^ int(bool(neg))
    # ---- stage 0: specials ----
    is_nan = (e == 0xFF) & (f != 0)
    is_inf = (e == 0xFF) & (f == 0)
    is_zs = e == 0                                       # zero or subnormal (flushed): e^0 = 1
    of = (s == 0) & ((e > 0x85) | ((e == 0x85) & (f > 0x31)))   # expFPIsInf: ln(max bf16) ~ 88.7
    early = np.select([is_nan, is_zs, is_inf & (s == 1), (is_inf & (s == 0)) | of],
                      [(s << 15) | (0xFF << 7) | (((f >> 6) & 1) ^ 1) << 6 | 0x3F, 0x3F80, 0x0000, 0x7F80], 0)
    term = is_nan | is_inf | is_zs | of
    # ---- stage 1: Q9.12 ----
    sig = 0x80 | f
    sh = e - 122                                         # qmnN + (e - 127) - (sigWidth - 1)
    mag = np.where(sh >= 21, 0, np.where(sh >= 0, (sig << np.clip(sh, 0, 20)) & ((1 << 21) - 1),
                                         sig >> np.clip(-sh, 0, 62)))
    v = _wrap(np.where(s == 1, -mag, mag), 21)
    # ---- stage 2: * 1/ln 2, k and r ----
    prod = (v * 5909) >> 12                              # 23-bit result, arithmetic (floor) shift
    k, r = prod >> 12, prod & 0xFFF
    # ---- stages 3-5: 2^r ----
    addr, low = r >> 7, r & 0x7F
    y0 = _LUT[addr]
    y1 = np.where(addr == 31, _LUT_TOP, _LUT[np.minimum(addr + 1, 31)])
    p = (y0 + (((y1 - y0) * low) >> 7)) & 0x1FFFF       # in [2^16, 2^17): 1.xxx with 16 fraction bits
    # ---- stage 6: 2^k 2^r -> bf16 (RNE); sExp = k + 256 in 10 bits ----
    E = _wrap(_wrap(k + 256, 11), 10) - 256              # p 2^(E - 16)
    be = E + 127
    q, rem = p >> 9, p & 0x1FF                           # 8 significant bits, 9 dropped
    up = (rem > 256) | ((rem == 256) & ((q & 1) == 1))
    q = q + up
    carry = q >= 256
    q, be_n = np.where(carry, q >> 1, q), be + carry
    normal = np.where(be_n >= 255, 0x7F80, (np.clip(be_n, 0, 254) << 7) | (q & 0x7F))
    # subnormal: value / 2^-133 = p 2^(E + 117); round p >> (-(E + 117)) to nearest even
    d = np.clip(-(E + 117), 1, 62)
    qs, rs, half = p >> d, p & ((np.int64(1) << d) - 1), np.int64(1) << (d - 1)
    sub = qs + ((rs > half) | ((rs == half) & ((qs & 1) == 1)))
    res = np.where(be >= 1, normal, np.where(E + 117 >= 0, 0, sub))
    return np.where(term, early, res).astype(np.uint16)


def _self_test():
    import math
    xs = np.arange(65536, dtype=np.uint32).astype(np.uint16)
    y = fexp_bf16(xs)
    xf = (xs.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    yf = (y.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    fin = np.isfinite(xf) & (np.abs(xf) < 80) & (np.abs(xf) > 2 ** -20)
    ref = np.exp(xf[fin])
    rel = np.abs(yf[fin] - ref) / ref
    print(f"fexp_bf16 vs exp over {fin.sum()} finite inputs |x| < 80: max rel err {rel.max():.4f} "
          f"(bf16 ulp {2 ** -8:.4f}), mean {rel.mean():.5f}")
    for xv in (0.0, 1.0, -1.0, 0.5, -64.0, -100.0, -300.0, -355.0, -356.0, -400.0, 88.0, 89.0):
        bits = np.array([np.float32(xv).view(np.uint32) >> 16], np.uint16)
        ybits = int(fexp_bf16(bits)[0])
        yv = np.array([ybits << 16], np.uint32).view(np.float32)[0]
        print(f"  fexp({xv:8.1f}) = {yv:.6g}   (exp {math.exp(xv):.6g})")


if __name__ == "__main__":
    _self_test()
