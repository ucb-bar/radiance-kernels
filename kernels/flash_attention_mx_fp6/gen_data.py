#!/usr/bin/env python3
"""Generate `data` for the LIVE fp6 chain kernel.

Live chain:  X (bf16 activation, op1 output stand-in)  --[SIMT runtime fp6-LUT
quantizer on device]-->  A_in indices + A_lut  -->  fp6 mxgemm  -->  C.

This gen produces ONLY the static operands (B weight, scales, canonical LUT) and the
GOLDEN C.  A_in and A_lut are produced ON DEVICE at runtime; the golden mirrors the
device quantizer bit-exactly (validated scalar bf16->fp6-code + fixed-point finder).
"""
import os, sys, math, struct
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib", "mxgemmini"))
import torch
from fp8_matmul_model import (tiled_matmul_hwlike, make_fp_quantizer,
                              tensor_to_custom_fp_codes)
from lut_golden_model import make_lut, quantize_lut_indices

M, K, N, GROUP = 64, 64, 64, 32
Gk = K // GROUP
INPUT_SPEC, SCALE_SPEC = "fp6:e3m2", "fpe8m0"
DEV = torch.device("cpu")
A_TILE_M, K_TILE = 32, 16
G = 1  # QUANT_LUT_UPDATE_GRANULARITY: every 2 rows/cols share a LUT

# ---------------------------------------------------------------- canonical LUT
torch.manual_seed(1234)
CANON = make_lut(INPUT_SPEC, 4, DEV)                       # 16 fp6-grid floats
def fp6_value_to_code(v):
    if v == 0.0: return 0
    s = 1 if v < 0 else 0; av = abs(v); emin = 1 - 3
    if av < 2.0**emin:
        mant = int(round(av / (2.0**(emin-2)))); return (s<<5)|max(0,min(mant,3))
    E = int(math.floor(math.log2(av))); base = 2.0**E
    mant = int(round((av-base)/(base/4)))
    if mant >= 4: mant = 0; E += 1
    return (s<<5)|(min(E+3,7)<<2)|min(mant,3)
def decode_fp6(code):
    s=(code>>5)&1; e=(code>>2)&7; m=code&3
    v = (m/4.0)*(2.0**(1-3)) if e==0 else (1.0+m/4.0)*(2.0**(e-3))
    return -v if s else v
CANON_CODES = [fp6_value_to_code(float(v)) for v in CANON.tolist()]

# ------------------------------- device-mirrored bf16->fp6 code (scalar, validated)
def bf16_to_fp6_code(bits):
    sign=(bits>>15)&1; E=(bits>>7)&0xFF; Mm=bits&0x7F
    if E==0: return 0
    if E==255: return (sign<<5)|0x1F
    e=E-127
    q=(Mm>>5)&3; r=(Mm>>4)&1; sticky=1 if (Mm&0xF) else 0; lsb=(Mm>>5)&1
    sig=q+(r&(sticky|lsb)); carry=1 if sig>=4 else 0
    mant_e=0 if carry else sig; exp_e=(e+1) if carry else e; quantum=2.0**-8
    if e>=-6:      av=float('inf') if exp_e>7 else (1.0+mant_e*0.25)*(2.0**exp_e)
    elif e==-7:    av=(2 if Mm<=32 else (3 if Mm<=95 else 4))*quantum
    elif e==-8:    av=(1 if Mm<64 else 2)*quantum
    elif e==-9:    av=(0 if Mm==0 else 1)*quantum
    else:          av=0.0
    if av==0.0 or av<=0.0546875: return 0
    if av>=32.0 or av==float('inf'): fp6=28.0
    elif 0.0625<=av<=0.21875: fp6=0.0625 if av<=0.078125 else (0.125 if av<=0.15625 else 0.1875)
    else: fp6=av
    if fp6<0.25:
        code=max(0,min(int(round(fp6/0.0625)),3))
    else:
        Ef=int(math.floor(math.log2(fp6))); base=2.0**Ef
        m=int(round((fp6-base)/(base/4)))
        if m>=4: m=0; Ef+=1
        code=(min(Ef+3,7)<<2)|min(m,3)
    return (sign<<5)|code

# ------------------------------------- fixed-point nearest finder (mirrors RTL)
def fp6_to_fixed(code):
    code&=0x3F; sign=(code>>5)&1; exp=(code>>2)&7; mant=code&3
    is_zero=(exp==0 and mant==0)
    s_exp=-2 if exp==0 else exp-3; imp=0 if exp==0 else 1; sig=(imp<<2)|mant
    shift=(s_exp+2)&7; shifted=(sig<<shift)&0x1FF; mag=shifted&0xFF
    return 0 if is_zero else (-mag if sign else mag)
CANON_FIXED = [fp6_to_fixed(c) for c in CANON_CODES]
def finder(in_code):
    fin=fp6_to_fixed(in_code); best=0; bd=abs(fin-CANON_FIXED[0])&0x1FF
    for i in range(1,16):
        d=abs(fin-CANON_FIXED[i])&0x1FF
        if d<bd: bd=d; best=i
    return best

# ---------------------------------------------------------------- runtime X (activation)
torch.manual_seed(7)
X = torch.randn(M, K, device=DEV, dtype=torch.float32)
X_bits = X.to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF     # (M,K) bf16 bits
A_indices = torch.zeros(M, K, dtype=torch.int64)
for m in range(M):
    for k in range(K):
        A_indices[m, k] = finder(bf16_to_fp6_code(int(X_bits[m, k])))
A_fp = torch.tensor([[decode_fp6(CANON_CODES[int(A_indices[m,k])]) for k in range(K)]
                     for m in range(M)], dtype=torch.float32)

# ---------------------------------------------------------------- static B weight
torch.manual_seed(0)
B = torch.randn(K, N, device=DEV, dtype=torch.float32)
B_luts = [make_lut(INPUT_SPEC, 4, DEV) for _ in range(N >> G)]
B_luts_t = torch.stack(B_luts)
B_indices = torch.stack([quantize_lut_indices(B_luts[j>>G], B[:, j]) for j in range(N)], dim=1)
ni = (torch.arange(N)>>G).unsqueeze(0).expand(K, -1)
B_fp = B_luts_t[ni, B_indices]

# ------------------------------------------------------------------------- scales
# A: unit e8m0 scale (activation quantized directly to fp6 grid).  B: fixed random.
A_scales_row_q = torch.ones(M, Gk, dtype=torch.float32)
torch.manual_seed(123)
B_scale_exp = torch.randint(-4, 4, (Gk, N))
B_scales_col_q = make_fp_quantizer(SCALE_SPEC, "nearest")(torch.pow(2.0, B_scale_exp.float()))

# --------------------------------------------------------------------- golden C
prod_list = [(4,3)]*16
acc_list  = [(4,4)]*8 + [(4,5)]*2 + [(4,6)]*5 + [(8,7)]*1
in_q = make_fp_quantizer(INPUT_SPEC, rounding="zero")
C_golden = tiled_matmul_hwlike(in_q(A_fp), in_q(B_fp), A_scales_row_q, B_scales_col_q,
                               verbose=False, prod_precision_list=prod_list, acc_precision_list=acc_list)
C_bits = C_golden.to(torch.bfloat16).view(torch.int16).numpy().astype("uint16").reshape(M, N) & 0xFFFF

# ---------------------------------------------- HW-layout sanity (device must match)
def a_hw_layout(idx):
    data = idx.tolist(); out=[[0]*K for _ in range(M//2)]
    for mi in range(0, M, A_TILE_M):
        for ki in range(0, K, K_TILE):
            for r in range(A_TILE_M//2):
                hw=mi//2+r
                for c in range(K_TILE):
                    col=ki+c
                    out[hw][col]=((data[mi+2*r+1][col]&0xF)<<4)|(data[mi+2*r][col]&0xF)
    return out
simple=[[((int(A_indices[2*i+1][k])&0xF)<<4)|(int(A_indices[2*i][k])&0xF) for k in range(K)] for i in range(M//2)]
assert simple==a_hw_layout(A_indices), "device simple row-pair packing != _a_indices_to_hw_layout"

# ------------------------------------------------------------- e8m0 scale codes
As_codes,_ = tensor_to_custom_fp_codes(A_scales_row_q.transpose(0,1), SCALE_SPEC)  # [Gk][M]
Bs_codes,_ = tensor_to_custom_fp_codes(B_scales_col_q, SCALE_SPEC)                 # [Gk][N]
# B_in nibble-packed [K][N/2]
B_idx = B_indices.tolist()
B_in = [[((B_idx[k][2*j+1]&0xF)<<4)|(B_idx[k][2*j]&0xF) for j in range(N//2)] for k in range(K)]
# B_lut / C_lut HW-packed words
def pack_lut(codes16):
    l=codes16
    w0=(l[0]|(l[1]<<6)|(l[2]<<12)|(l[3]<<18)|(l[4]<<24)|(l[5]<<30))&0xFFFFFFFF
    w1=((l[5]>>2)|(l[6]<<4)|(l[7]<<10)|(l[8]<<16)|(l[9]<<22)|(l[10]<<28))&0xFFFFFFFF
    w2=((l[10]>>4)|(l[11]<<2)|(l[12]<<8)|(l[13]<<14)|(l[14]<<20)|(l[15]<<26))&0xFFFFFFFF
    return (w0,w1,w2)
B_lut_codes,_ = tensor_to_custom_fp_codes(B_luts_t, INPUT_SPEC)   # [N/2][16]
B_lut_words = [pack_lut(row) for row in B_lut_codes]
C_lut_words = [pack_lut(CANON_CODES) for _ in range(M//2)]

# --------------------------------------------------------------------- emit header
def arr2d(name, ctype, rows, cols, data, fmt="0x{:02x}"):
    body=",\n".join("    { "+", ".join(fmt.format(data[r][c]) for c in range(cols))+" }" for r in range(rows))
    return f"static const {ctype} {name}[{rows}][{cols}] = {{\n{body}\n}};\n\n"

out=[]
out.append("#ifndef _FP6_LIVE_DATA_H\n#define _FP6_LIVE_DATA_H\n#include <stdint.h>\n\n")
out.append(f"#define MATMUL_M   {M}\n#define MATMUL_K   {K}\n#define MATMUL_N   {N}\n")
out.append(f"#define MATMUL_GK  {Gk}\n#define MATMUL_GN  {N//GROUP}\n")
out.append(f"#define A_TILE_M   {A_TILE_M}\n#define K_TILE     {K_TILE}\n\n")
# runtime activation X (bf16 bits)
out.append(arr2d("X_bf16","uint16_t",M,K,X_bits.tolist(),"0x{:04x}"))
# canonical LUT: 16 raw 6-bit codes (device stages + runs finder)
out.append("// Canonical 16-entry fp6:e3m2 LUT (from lut_golden_model.make_lut, seed 1234)\n")
out.append("static const uint8_t CANON_LUT[16] = { "+", ".join(f"0x{c:02x}" for c in CANON_CODES)+" };\n\n")
# static B operand
out.append(arr2d("B_in","uint8_t",K,N//2,B_in))
out.append("static const uint32_t B_lut[%d][3] = {\n"%(N//2)+",\n".join("    { "+", ".join(f"0x{w:08x}" for w in ws)+" }" for ws in B_lut_words)+"\n};\n\n")
out.append("static const uint32_t C_lut[%d][3] = {\n"%(M//2)+",\n".join("    { "+", ".join(f"0x{w:08x}" for w in ws)+" }" for ws in C_lut_words)+"\n};\n\n")
# A_lut_baked: canonical palette, HW-packed, one per A row-pair (const; used when the mesh
# loads a fixed palette).  The RUNTIME_LUT kernel variant instead stages this on-device.
A_lut_words = [pack_lut(CANON_CODES) for _ in range(M//2)]
out.append("static const uint32_t A_lut_baked[%d][3] = {\n"%(M//2)+",\n".join("    { "+", ".join(f"0x{w:08x}" for w in ws)+" }" for ws in A_lut_words)+"\n};\n\n")
# scales
out.append(arr2d("A_scales_row","uint8_t",Gk,M,As_codes))
out.append(arr2d("B_scales_col","uint8_t",Gk,N,Bs_codes))
# golden
out.append(arr2d("C_out_bf16","uint16_t",M,N,C_bits.tolist(),"0x{:04x}"))
# golden A_in (HW layout) + A_lut words, for quantizer-only debug verification
out.append(arr2d("A_in_gold","uint8_t",M//2,K,simple))
CANON_WORDS = pack_lut(CANON_CODES)
out.append("static const uint32_t A_lut_gold[3] = { "+", ".join(f"0x{w:08x}" for w in CANON_WORDS)+" };\n\n")
out.append("__global uint16_t C_raw[MATMUL_M * MATMUL_N] = {0};\n")
out.append("static const uint16_t* gold_raw = &C_out_bf16[0][0];\n")
out.append("#define VERIFY_COUNT (MATMUL_M * MATMUL_N)\n\n")
out.append("#endif\n")
open("data","w").write("".join(out))
print("wrote data:  M,K,N=%d,%d,%d  CANON codes=%s"%(M,K,N,[f'{c:02x}' for c in CANON_CODES]))
print("index histogram:", torch.bincount(A_indices.flatten(),minlength=16).tolist())
print("C_golden[0,:6]:", C_golden[0,:6].tolist())
