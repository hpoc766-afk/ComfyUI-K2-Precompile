# SPDX-License-Identifier: GPL-3.0-only
"""Inference fusion kernels. Intermediate low-precision casts are intentional.

No torch.compile calls live here; these kernels execute inside the model graph's
operators. Weight formats are handled by the projection backend independently.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _norm_mod(X, W, Scale, Shift, Y, D: tl.constexpr, L: tl.constexpr,
              CUT: tl.constexpr, BATCH: tl.constexpr, MOD: tl.constexpr,
              EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    dtype: tl.constexpr = X.dtype.element_ty
    x = tl.load(X + row * D + col, col < D, 0).to(tl.float32)
    w = tl.load(W + col, col < D, 0).to(tl.float32) + 1.0
    # ATen 2.9's vectorized RMSNorm: 128 threads, four consecutive values
    # per thread, then warp and inter-warp reductions. Match its accumulation
    # order instead of reducing the whole row in an arbitrary tree.
    thread = tl.arange(0, 128)
    partial = tl.full((128,), 0, tl.float32)
    for base in range(tl.cdiv(D, 512)):
        for item in tl.static_range(4):
            j = base * 512 + thread * 4 + item
            v = tl.load(X + row * D + j, j < D, 0).to(tl.float32)
            partial = tl.fma(v, v, partial)
    warp_sums = tl.sum(tl.reshape(partial, (4, 32)), 1)
    total = tl.sum(warp_sums, 0)
    inv = tl.rsqrt(libdevice.div_rn(total, float(D)) + EPS)
    y = ((x * inv) * w).to(dtype)
    if MOD:
        batch = row // L
        if CUT >= 0:
            batch += tl.where(row % L >= CUT, BATCH, 0)
        scale = tl.load(Scale + batch * D + col, col < D, 0)
        shift = tl.load(Shift + batch * D + col, col < D, 0)
        factor = (scale.to(tl.float32) + 1.0).to(dtype)
        y = (y.to(tl.float32) * factor.to(tl.float32)).to(dtype)
        y = (y.to(tl.float32) + shift.to(tl.float32)).to(dtype)
    tl.store(Y + row * D + col, y, col < D)


@triton.jit
def _activate_mul(A, B, Y, N: tl.constexpr, KIND: tl.constexpr,
                  BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    dtype: tl.constexpr = A.dtype.element_ty
    a = tl.load(A + i, i < N, 0).to(tl.float32)
    b = tl.load(B + i, i < N, 0).to(tl.float32)
    # libdevice exp matches ATen's float activation; round before multiplication.
    denom = 1.0 + libdevice.exp(-a)
    if KIND == 0:
        activated = libdevice.div_rn(a, denom).to(dtype)
    else:
        activated = libdevice.div_rn(1.0, denom).to(dtype)
    tl.store(Y + i, activated.to(tl.float32) * b, i < N)


@triton.jit
def _qk_post(Q, K, QW, KW, F, QO, KO, L: tl.constexpr, QH: tl.constexpr,
             KH: tl.constexpr, D: tl.constexpr, FB: tl.constexpr,
             QE: tl.constexpr, KE: tl.constexpr, QDONE: tl.constexpr,
             KDONE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    h = row % (QH + KH)
    token = row // (QH + KH) % L
    batch = row // ((QH + KH) * L)
    col = tl.arange(0, BLOCK)
    if h < QH:
        src, weight, dst = Q, QW, QO
        head, heads, eps = h, QH, QE
    else:
        src, weight, dst = K, KW, KO
        head, heads, eps = h - QH, KH, KE
    if ((h < QH) & (not QDONE)) | ((h >= QH) & (not KDONE)):
        dtype: tl.constexpr = src.dtype.element_ty
        start = ((batch * L + token) * heads + head) * D
        x = tl.load(src + start + col, col < D, 0).to(tl.float32)
        lane = tl.arange(0, BLOCK // 4)
        partial = tl.full((BLOCK // 4,), 0, tl.float32)
        for item in tl.static_range(4):
            v = tl.load(src + start + lane * 4 + item, lane * 4 + item < D, 0).to(tl.float32)
            partial = tl.fma(v, v, partial)
        inv = tl.rsqrt(libdevice.div_rn(tl.sum(partial, 0), float(D)) + eps)
        w = tl.load(weight + col, col < D, 0).to(tl.float32) + 1.0
        y = ((x * inv) * w).to(dtype).to(tl.float32)
        even = tl.gather(y, col // 2 * 2, 0)
        odd = tl.gather(y, col // 2 * 2 + 1, 0)
        fbase = ((tl.where(FB == 1, 0, batch) * L + token) * (D // 2) + col // 2) * 4 + col % 2 * 2
        fa = tl.load(F + fbase, col < D, 0)
        fb = tl.load(F + fbase + 1, col < D, 0)
        out = tl.fma(fa, even, fb * odd)
        tl.store(dst + start + col, out, col < D)


def qk_post(q, k, qw, kw, freqs, qeps, keps, heads, kvheads, qdone=False, kdone=False):
    if qdone and kdone:
        return q, k
    dim = q.shape[-1] // heads
    q, k = q.contiguous(), k.contiguous()
    if (not q.is_cuda or not q.is_contiguous() or not k.is_contiguous()
            or dim not in (32, 64, 128, 256) or freqs is None
            or freqs.dtype != torch.float32 or not freqs.is_contiguous()
            or freqs.shape[1:] != (1, q.shape[1], dim // 2, 2, 2)):
        return None
    from comfy import model_management as mm
    qw = mm.cast_to(qw, device=q.device, dtype=qw.dtype)
    kw = mm.cast_to(kw, device=q.device, dtype=kw.dtype)
    qo = q if qdone else torch.empty_like(q)
    ko = k if kdone else torch.empty_like(k)
    b, length, _ = q.shape
    _qk_post[(b * length * (heads + kvheads),)](
        q, k, qw, kw, freqs, qo, ko, length, heads, kvheads, dim, freqs.shape[0],
        qeps, keps, qdone, kdone, triton.next_power_of_2(dim),
        num_warps=4, enable_fp_fusion=False)
    return qo, ko


def norm_mod(x, weight, eps, scale=None, shift=None, cut=-1):
    if not x.is_cuda or x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return None
    if x.ndim != 3 or not x.is_contiguous() or weight.numel() != x.shape[-1]:
        return None
    if scale is not None and (not scale.is_contiguous() or not shift.is_contiguous()
                              or scale.dtype != x.dtype or shift.dtype != x.dtype):
        return None
    from comfy import model_management as mm
    weight = mm.cast_to(weight, device=x.device, dtype=weight.dtype)
    out = torch.empty_like(x)
    b, length, dim = x.shape
    _norm_mod[(b * length,)](x, weight, scale if scale is not None else weight,
                            shift if shift is not None else weight, out,
                            dim, length, cut, b, scale is not None, eps,
                            triton.next_power_of_2(dim), num_warps=4 if dim <= 2048 else 8,
                            enable_fp_fusion=False)
    return out


def activate_mul(a, b, kind):
    if (not a.is_cuda or a.shape != b.shape or a.dtype != b.dtype
            or not a.is_contiguous() or not b.is_contiguous()):
        return None
    out = torch.empty_like(a)
    _activate_mul[(triton.cdiv(a.numel(), 256),)](a, b, out, a.numel(),
                                                0 if kind == 'silu' else 1, 256,
                                                enable_fp_fusion=False)
    return out
