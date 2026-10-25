# SPDX-License-Identifier: GPL-3.0-only
"""Forward-only GQA with the attention gate folded into the output store.

One program owns a query tile and a Q head; four Q heads can directly address
one K/V head without materializing repeat_interleave copies. Online softmax
uses exp2. The low precision attention-output cast precedes the gate multiply.
"""
import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


@triton.jit
def _gqa_gated(Q, K, V, Gate, Out,
               L, HQ: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
               SQB, SQH, SQL, SKB, SKH, SKL, SVB, SVH, SVL,
               BM: tl.constexpr, BN: tl.constexpr):
    block, head, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    kvhead = head // (HQ // HK)
    rows = block * BM + tl.arange(0, BM)
    cols = tl.arange(0, BN)
    channels = tl.arange(0, D)
    q = tl.load(Q + batch * SQB + head * SQH + rows[:, None] * SQL + channels[None, :],
                rows[:, None] < L, 0)
    maximum = tl.full((BM,), float('-inf'), tl.float32)
    denominator = tl.full((BM,), 0., tl.float32)
    accumulator = tl.full((BM, D), 0., tl.float32)
    factor: tl.constexpr = 1.4426950408889634 * (D ** -0.5)
    for start in range(tl.cdiv(L, BN)):
        tokens = start * BN + cols
        k = tl.load(K + batch * SKB + kvhead * SKH + tokens[None, :] * SKL + channels[:, None],
                    tokens[None, :] < L, 0)
        score = tl.dot(q, k) * factor
        score = tl.where(tokens[None, :] < L, score, float('-inf'))
        updated = tl.maximum(maximum, tl.max(score, 1))
        alpha = tl.exp2(maximum - updated)
        p = tl.exp2(score - updated[:, None])
        denominator = denominator * alpha + tl.sum(p, 1)
        accumulator *= alpha[:, None]
        v = tl.load(V + batch * SVB + kvhead * SVH + tokens[:, None] * SVL + channels[None, :],
                    tokens[:, None] < L, 0)
        accumulator = tl.dot(p.to(q.dtype), v, accumulator)
        maximum = updated
    rounded = (accumulator / denominator[:, None]).to(Out.dtype.element_ty)
    offsets = (batch * L + rows[:, None]) * (HQ * D) + head * D + channels[None, :]
    gate = tl.load(Gate + offsets, rows[:, None] < L, 0)
    output = (rounded.to(tl.float32) * gate.to(tl.float32)).to(Out.dtype.element_ty)
    tl.store(Out + offsets, output, rows[:, None] < L)


@triton_op('k2_fullgraph::gqa_gated', mutates_args=())
def gated_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    sigmoid_gate: torch.Tensor) -> torch.Tensor:
    batch, heads, length, dim = q.shape
    output = torch.empty((batch, length, heads * dim), device=q.device, dtype=q.dtype)
    wrap_triton(_gqa_gated)[(triton.cdiv(length, 32), heads, batch)](
        q, k, v, sigmoid_gate, output, length, heads, k.shape[1], dim,
        *q.stride()[:3], *k.stride()[:3], *v.stride()[:3],
        BM=32, BN=64, num_warps=4, num_stages=1, enable_fp_fusion=False)
    return output
