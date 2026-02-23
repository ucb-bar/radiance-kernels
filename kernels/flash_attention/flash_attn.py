import sys
from ml_dtypes import bfloat16
import numpy as np

if __name__ == "__main__":
    seqlen, headdim = 1024, 64

    np.random.seed(0)
    # A = np.random.rand(seqlen, headdim) - 0.5
    # B = np.random.rand(headdim, seqlen) - 0.5
    # C = np.random.rand(seqlen, headdim) - 0.5
    A = np.arange(seqlen * headdim).reshape([seqlen, headdim])
    B = np.arange(headdim * seqlen).reshape([headdim, seqlen])
    C = np.arange(seqlen * seqlen).reshape([seqlen, seqlen])
    # C = np.zeros([M, N])

    A.astype('bfloat16').tofile("numpy.a.bin")
    B.astype('bfloat16').tofile("numpy.b.bin")
