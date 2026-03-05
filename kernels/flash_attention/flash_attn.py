import sys
from ml_dtypes import bfloat16
import numpy as np

if __name__ == "__main__":
    seqlen, headdim = 1024, 64

    np.random.seed(0)
    # A = np.random.rand(seqlen, headdim) - 0.5
    # B = np.random.rand(headdim, seqlen) - 0.5
    # C = np.random.rand(seqlen, headdim) - 0.5
    A = np.arange(seqlen * headdim).astype('bfloat16').reshape([seqlen, headdim])
    B = np.arange(headdim * seqlen).astype('bfloat16').reshape([headdim, seqlen])
    C = np.arange(seqlen * seqlen).astype('bfloat16').reshape([seqlen, seqlen])
    # C = np.zeros([M, N])

    A.tofile("numpy.a.bin")
    B.tofile("numpy.b.bin")

    rowmax = np.max(A, axis=1)
    rowmax.tofile("rowmax.a.bin")
