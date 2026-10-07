"""Metal kernel (torch.mps.compile_shader) that replaces the 63-step sequential
row-substitution loop inside transformers' torch_chunk_gated_delta_rule with a
single dispatch per layer.

The loop computes T = (I - L)^-1 for each 64x64 chunk matrix (L strictly lower,
from the decayed chunk attention), by forward substitution:
    for i in 1..63:  A[i,:i] = A[i,:i] + A[i,:i] @ A[:i,:i]
This kernel assigns one 64x64 matrix to one threadgroup; threads 0..63 each own
one column and iterate rows in lockstep using threadgroup memory. The identity
is folded into the write-back, so the output is T directly.
"""

import torch

_SHADER = """
#include <metal_stdlib>
using namespace metal;

kernel void row_substitution(device float* attn [[buffer(0)]],
                             constant uint& count [[buffer(1)]],
                             uint tpgid [[threadgroup_position_in_grid]],
                             uint tid [[thread_position_in_threadgroup]]) {
    threadgroup float M[64][64];
    threadgroup float rowbuf[64];
    if (tpgid >= count) { return; }
    device float* A = attn + (size_t)tpgid * 4096;
    if (tid < 64) {
        for (uint idx = tid; idx < 4096; idx += 64) {
            M[idx >> 6][idx & 63] = A[idx];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 1; i < 64; i++) {
        if (tid < 64) { rowbuf[tid] = M[i][tid]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < i) {
            float sum = 0.0f;
            for (uint k = 0; k < i; k++) {
                sum += rowbuf[k] * M[k][tid];
            }
            M[i][tid] = rowbuf[tid] + sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 64) {
        for (uint idx = tid; idx < 4096; idx += 64) {
            uint r = idx >> 6, c = idx & 63;
            A[idx] = M[r][c] + (r == c ? 1.0f : 0.0f);
        }
    }
}
"""

_lib = None


def lib():
    global _lib
    if _lib is None:
        _lib = torch.mps.compile_shader(_SHADER)
    return _lib


def row_substitution_inplace(attn, num_matrices):
    """attn: contiguous fp32 (..., NC, 64, 64) on MPS; replaced in place with (I-L)^-1.

    The first tensor argument fixes the dispatch size (count*4096 threads,
    1024-thread threadgroups): threadgroup tpgid < count processes matrix tpgid,
    the remaining threadgroups exit early.
    """
    count = num_matrices
    flat = attn.reshape(-1, 64, 64)
    lib().row_substitution(flat, count)
    return attn
