# SPDX-License-Identifier: LicenseRef-KohakuMoss-1.0
# Copyright 2026 Shih-Ying Yeh (KohakuBlueLeaf) and contributors.
# Modified for K2: native GQA head addressing, uniform batched/strided tensors,
# optional masks and gated output, inference-only (no saved M/L or backward).
# Adapted from KohakuMossTrainer 0536e62b6062039104d8f9575635b5d07bf573d3,
# src/kohakumoss_kernels/backends/triton/attention/{_kernels,varlen}.py.
"""Kohaku online-softmax FA; no repeated KV heads or full attention matrix."""
import math

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

LOG2E = 1.0 / math.log(2.0)
SAFE = {'BM': 64, 'BN': 32, 'warps': 4, 'stages': 1}
# Kohaku sm89 inherits sm86's seeds; shared seeds follow the arch seeds.
SEEDS = [
    {'BM': 128, 'BN': 32, 'warps': 4, 'stages': 2},
    {'BM': 128, 'BN': 64, 'warps': 8, 'stages': 2},
    {'BM': 64, 'BN': 64, 'warps': 4, 'stages': 2},
    SAFE,
]


@triton.jit
def _mma(a, b, acc):
    return tl.dot(a.to(b.dtype), b, acc)


@triton.jit
def _fwd_tile(acc, m_i, l_i, q, k_ptrs, v_ptrs, cols, rows, c,
              mask_ptrs, INV_SCALE: tl.constexpr, MASK_KIND: tl.constexpr,
              MASKED: tl.constexpr):
    if MASKED:
        k = tl.load(k_ptrs, mask=cols[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=cols[:, None], other=0.0)
    else:
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
    s = tl.dot(q, tl.trans(k))
    if MASKED:
        s = tl.where(cols[None, :], s, float('-inf'))
    if MASK_KIND == 1:
        allowed = tl.load(mask_ptrs, rows[:, None] & cols[None, :], other=False)
        s = tl.where(allowed, s, float('-inf'))
    elif MASK_KIND == 2:
        bias = tl.load(mask_ptrs, rows[:, None] & cols[None, :], other=0.0).to(tl.float32)
        s += bias * INV_SCALE
    m_new = tl.maximum(m_i, tl.max(s, 1))
    if MASK_KIND:
        base = tl.where(m_new == float('-inf'), 0.0, m_new)
        alpha = tl.where(m_new == float('-inf'), 0.0, tl.exp2((m_i - base) * c))
    else:
        base = m_new
        alpha = tl.exp2((m_i - m_new) * c)
    p = tl.exp2((s - base[:, None]) * c)
    acc = _mma(p, v, acc * alpha[:, None])
    l_i = l_i * alpha + tl.sum(p, 1)
    return acc, m_new, l_i


@triton.jit
def fwd_kernel(Q, K, V, Gate, Mask, Out, c, LQ, LK,
               sq_b, sq_h, sq_n, sq_d, sk_b, sk_h, sk_n, sk_d,
               sv_b, sv_h, sv_n, sv_d, sm_b, sm_h, sm_q, sm_k,
               HQ: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr,
               MASK_KIND: tl.constexpr, GATE_KIND: tl.constexpr):
    pid, head, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    kv_head = head // (HQ // HK)
    c = tl.full((), 1.0, tl.float32) * c
    offs_m = pid.to(tl.int64) * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    rows = offs_m < LQ
    q = tl.load(Q + batch.to(tl.int64) * sq_b + head * sq_h
                + offs_m[:, None] * sq_n + offs_d[None, :] * sq_d, rows[:, None], other=0.0)
    acc = tl.zeros([BM, D], tl.float32)
    m_i = tl.full([BM], float('-inf'), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    k_base = K + batch.to(tl.int64) * sk_b + kv_head * sk_h
    v_base = V + batch.to(tl.int64) * sv_b + kv_head * sv_h
    mask_base = Mask + batch.to(tl.int64) * sm_b + head * sm_h
    full = LK // BN * BN
    for n0 in range(0, full, BN):
        offs_n = n0 + tl.arange(0, BN).to(tl.int64)
        acc, m_i, l_i = _fwd_tile(
            acc, m_i, l_i, q,
            k_base + offs_n[:, None] * sk_n + offs_d[None, :] * sk_d,
            v_base + offs_n[:, None] * sv_n + offs_d[None, :] * sv_d,
            offs_n < LK, rows, c,
            mask_base + offs_m[:, None] * sm_q + offs_n[None, :] * sm_k,
            D ** 0.5, MASK_KIND, False)
    if full < LK:
        offs_n = full + tl.arange(0, BN).to(tl.int64)
        acc, m_i, l_i = _fwd_tile(
            acc, m_i, l_i, q,
            k_base + offs_n[:, None] * sk_n + offs_d[None, :] * sk_d,
            v_base + offs_n[:, None] * sv_n + offs_d[None, :] * sv_d,
            offs_n < LK, rows, c,
            mask_base + offs_m[:, None] * sm_q + offs_n[None, :] * sm_k,
            D ** 0.5, MASK_KIND, True)
    out = ((batch.to(tl.int64) * LQ + offs_m[:, None]) * HQ + head) * D + offs_d[None, :]
    o = tl.where(l_i[:, None] == 0, 0.0, acc / l_i[:, None]).to(Out.dtype.element_ty)
    if GATE_KIND:
        gate = tl.load(Gate + out, rows[:, None], other=0.0)
        if GATE_KIND == 1:
            gate = (1.0 / (1.0 + tl.exp(-gate.to(tl.float32)))).to(Out.dtype.element_ty)
        o = (o.to(tl.float32) * gate.to(tl.float32)).to(Out.dtype.element_ty)
    tl.store(Out + out, o, rows[:, None])


def candidates(q, device):
    length, dim = int(q.shape[2]), int(q.shape[3])
    if length <= 16:
        return [{'BM': 16, 'BN': 16, 'warps': 4, 'stages': 1}, SAFE]
    props = torch.cuda.get_device_properties(device)
    limit = getattr(props, 'shared_memory_per_block_optin', props.shared_memory_per_block)
    return [cfg for cfg in SEEDS
            if 2 * dim * (cfg['BM'] + cfg['stages'] * 2 * cfg['BN']) <= limit
            and (cfg['BM'] * dim + 2 * cfg['BM'] * cfg['BN']) // (32 * cfg['warps']) <= 192]


def _mask_layout(mask, q, k):
    if mask is None:
        return (0, 0, 0, 0), 0
    if mask.ndim == 1:
        sizes, strides = (1, 1, 1, mask.shape[0]), (0, 0, 0, mask.stride(0))
    elif mask.ndim == 2:
        if mask.shape[0] == q.shape[0] and mask.shape[0] != q.shape[2]:
            sizes, strides = (mask.shape[0], 1, 1, mask.shape[1]), (mask.stride(0), 0, 0, mask.stride(1))
        else:
            sizes, strides = (1, 1, *mask.shape), (0, 0, *mask.stride())
    elif mask.ndim == 3:
        sizes, strides = (mask.shape[0], 1, mask.shape[1], mask.shape[2]), (mask.stride(0), 0, mask.stride(1), mask.stride(2))
    elif mask.ndim == 4:
        sizes, strides = tuple(mask.shape), mask.stride()
    else:
        raise ValueError('Kohaku FA mask must have 1 to 4 dimensions')
    expected = (q.shape[0], q.shape[1], q.shape[2], k.shape[2])
    if any(size != 1 and size != target for size, target in zip(sizes, expected)):
        raise ValueError('Kohaku FA mask is not broadcastable to attention scores')
    return tuple(0 if size == 1 else stride for size, stride in zip(sizes, strides)), 1 if mask.dtype == torch.bool else 2


def launch(q, k, v, mask, gate, gate_is_sigmoid, cfg, wrapped=False):
    if not q.is_cuda or k.device != q.device or v.device != q.device:
        raise ValueError('Kohaku FA requires Q/K/V on the same CUDA device')
    if mask is not None and (mask.device != q.device or mask.dtype not in (
            torch.bool, torch.float16, torch.bfloat16, torch.float32, torch.float64)):
        raise ValueError('Kohaku FA requires a boolean or floating-point mask on the Q/K/V device')
    if q.dtype not in (torch.bfloat16, torch.float16) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError('Kohaku FA expects matching BF16/FP16 Q/K/V')
    if (q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or q.shape[0] != k.shape[0]
            or k.shape[:3] != v.shape[:3] or k.shape[1] <= 0 or q.shape[-1] != k.shape[-1]
            or q.shape[-1] != v.shape[-1] or q.shape[1] % k.shape[1]):
        raise ValueError('Kohaku FA expects Q[B,Hq,Lq,D], K/V[B,Hkv,Lk,D] with Hq divisible by Hkv')
    if q.shape[-1] not in (32, 64, 128, 256):
        raise ValueError('Kohaku FA supports head dimensions 32/64/128/256')
    batch, heads, length, dim = q.shape
    out = torch.empty((batch, length, heads * dim), device=q.device, dtype=q.dtype)
    if length == 0:
        return out
    strides, mask_kind = _mask_layout(mask, q, k)
    gate_kind = 0
    if gate is not None:
        if gate.shape != out.shape or gate.dtype != q.dtype or gate.device != q.device:
            raise ValueError('Kohaku FA gate must match output shape, precision and device')
        gate = gate.contiguous()
        gate_kind = 2 if gate_is_sigmoid else 1
    runner = wrap_triton(fwd_kernel) if wrapped else fwd_kernel
    runner[(triton.cdiv(length, cfg['BM']), heads, batch)](
        q, k, v, gate if gate is not None else q, mask if mask is not None else q, out,
        LOG2E * dim ** -0.5, length, k.shape[2], *q.stride(), *k.stride(), *v.stride(), *strides,
        HQ=heads, HK=k.shape[1], D=dim, BM=cfg['BM'], BN=cfg['BN'],
        MASK_KIND=mask_kind, GATE_KIND=gate_kind,
        num_warps=cfg['warps'], num_stages=cfg['stages'])
    return out


@triton_op('k2_fullgraph::kohaku_fa_launch', mutates_args=())
def compiled_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                       mask: torch.Tensor | None, gate: torch.Tensor | None,
                       gate_is_sigmoid: bool, bm: int, bn: int, warps: int, stages: int) -> torch.Tensor:
    return launch(q, k, v, mask, gate, gate_is_sigmoid,
                  {'BM': bm, 'BN': bn, 'warps': warps, 'stages': stages}, wrapped=True)


@torch.library.custom_op('k2_fullgraph::kohaku_fa', mutates_args=())
def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              mask: torch.Tensor | None, gate: torch.Tensor | None,
              gate_is_sigmoid: bool) -> torch.Tensor:
    return launch(q, k, v, mask, gate, gate_is_sigmoid, SAFE)


@attention.register_fake
def _fake(q, k, v, mask, gate, gate_is_sigmoid):
    return torch.empty((q.shape[0], q.shape[2], q.shape[1] * q.shape[3]), device=q.device, dtype=q.dtype)
