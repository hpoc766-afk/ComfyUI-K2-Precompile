# SPDX-License-Identifier: GPL-3.0-only
"""Explicit packed normalized activations; buffers live only for this forward."""
import math
import torch


def key(module):
    value = getattr(module.forward, 'native_key', None)
    if value is None:
        raise RuntimeError('Normalized packed inputs require the native fullgraph adapter')
    return value


def norm_quant(module, target, x, scale, shift, cut=-1):
    return torch.ops.k2_native.norm_quant(x, scale, shift, key(module), key(target), cut)


def gated_norm_quant(module, norm, target, x, gate, residual, scale, shift):
    return torch.ops.k2_native.gated_norm_quant(x, gate, residual, scale, shift,
                                               key(module), key(norm), key(target), -1)


def attention_inputs(module, x_metadata, packed, freqs):
    return torch.ops.k2_native.qkv_prepared(x_metadata, packed[0], packed[1], freqs,
        [key(m) for m in (module.wq, module.wk, module.wv, module.gate)],
        [key(m) for m in (module.qknorm.qnorm, module.qknorm.knorm)], module.heads, module.kvheads)


def swiglu(module, x_metadata, packed):
    return torch.ops.k2_native.swiglu_prepared(x_metadata, packed[0], packed[1],
                                              [key(module.gate), key(module.up)], module.gate.out_features)


def fake_norm_quant(x, scale, shift, norm_key, target_key, cut):
    rows, width = math.prod(x.shape[:-1]), x.shape[-1]
    return (torch.empty((rows, width // 2), device=x.device, dtype=torch.int8),
            torch.empty((rows, 1), device=x.device, dtype=torch.float32))


def fake_gated_norm_quant(x, gate, residual, scale, shift, key, norm_key, target_key, cut):
    packed = fake_norm_quant(residual, scale, shift, norm_key, target_key, cut)
    return (torch.empty_like(residual, memory_format=torch.contiguous_format), *packed)
