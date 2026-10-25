# SPDX-License-Identifier: GPL-3.0-only
"""Joint packed-INT4 projection launchers and immutable binding metadata."""
from contextlib import ExitStack
import ctypes
import functools
import hashlib
import os
from pathlib import Path

import torch


@functools.lru_cache(maxsize=1)
def library_path():
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256((root / 'kernels/joint_projection.cu').read_bytes()).hexdigest()[:16]
    path = root / '.kernel-build' / ('joint_projection_' + digest + ('.dll' if os.name == 'nt' else '.so'))
    return str(path) if path.is_file() else ''


@functools.lru_cache(maxsize=1)
def library():
    lib = ctypes.CDLL(library_path())
    lib.k2_int4_swiglu.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
    lib.k2_int4_swiglu.restype = ctypes.c_int
    lib.k2_int4_qkv.argtypes = ([ctypes.c_void_p] * 17 + [ctypes.c_int] * 5
                              + [ctypes.c_float] * 2 + [ctypes.c_void_p])
    lib.k2_int4_qkv.restype = ctypes.c_int
    lib.k2_int4_residual_norm.argtypes = ([ctypes.c_void_p] * 12 + [ctypes.c_int] * 4
                                       + [ctypes.c_float, ctypes.c_void_p])
    lib.k2_int4_residual_norm.restype = ctypes.c_int
    return lib


def metadata(binding):
    """Run outside graph capture; record a joint projection's storage contract."""
    if binding.kind != 0 or not binding.fuse_epilogue or torch.is_grad_enabled():
        return None
    from comfy import model_management as mm
    from comfy_kitchen.tensor import QuantizedTensor
    from comfy_kitchen.tensor.convrot_w4a4 import TensorCoreConvRotW4A4Layout
    from comfy_kitchen.backends.eager.svdquant import _INT4_GROUP_SIZE
    from .int4_epilogue import is_native_projection, has_hooks
    module = binding.module
    if (mm.in_training or has_hooks(module) or module.bias is not None
            or any(getattr(module, name, None) for name in (
                'comfy_force_forward', '_full_precision_mm', 'comfy_force_cast_weights',
                'weight_function', 'bias_function', 'weight_lowvram_function', 'bias_lowvram_function'))
            or getattr(module, 'pre_quant_scale', None) is not None):
        return None
    with binding.native_forward():
        if not is_native_projection(module):
            return None
    weight = module.weight
    if not isinstance(weight, QuantizedTensor) or weight.layout_cls is not TensorCoreConvRotW4A4Layout:
        return None
    p = weight.params
    if (p.transposed or p.linear_dtype != 'int4' or p.quant_group_size != _INT4_GROUP_SIZE
            or p.convrot_groupsize not in (16, 64, 256)):
        return None
    return p.convrot_groupsize


def input_supported(x):
    return (x.is_cuda and x.ndim == 3 and x.dtype in (torch.bfloat16, torch.float16)
            and x.shape[-1] % 256 == 0 and torch.cuda.get_device_capability(x.device) == (8, 9))


def _weights(stack, bindings, x):
    from comfy import ops
    from comfy_kitchen.tensor import QuantizedTensor
    from comfy_kitchen.tensor.convrot_w4a4 import TensorCoreConvRotW4A4Layout
    weights = []
    for binding in bindings:
        module = binding.module
        weight, bias = stack.enter_context(ops.CastBiasWeightContext(
            module, input=None, dtype=module.weight.dtype, device=x.device,
            bias_dtype=x.dtype, offloadable=True, compute_dtype=x.dtype, want_requant=True))
        if (bias is not None or not isinstance(weight, QuantizedTensor)
                or weight.layout_cls is not TensorCoreConvRotW4A4Layout
                or weight.params.transposed or weight.params.linear_dtype != 'int4'
                or weight.params.convrot_groupsize != binding.joint_info):
            raise RuntimeError('Joint projection weight representation changed during preparation')
        weights.append(weight.layout_cls.get_plain_tensors(weight))
    return weights


def residual_norm(x, gate, residual, scale, shift, binding, norm):
    from comfy import model_management as mm
    from comfy_kitchen.backends import cuda as kitchen
    m, k, n = x.numel() // x.shape[-1], x.shape[-1], binding.width
    with ExitStack() as stack:
        (weight, ws), = _weights(stack, [binding], x)
        qa, xs = kitchen.quantize_int4_rowwise_convrot64(x.reshape(m, k).contiguous(), binding.joint_info)
        gamma = mm.cast_to(norm.module.scale, dtype=torch.float32, device=x.device).contiguous()
        gate, residual, scale, shift = [v.contiguous() for v in (gate, residual, scale, shift)]
        output = torch.empty_like(residual)
        normalized = torch.empty_like(residual)
        counters = torch.empty(((m + 127) // 128,), device=x.device, dtype=torch.int32)
        with torch.cuda.device(x.device):
            status = library().k2_int4_residual_norm(
                *(v.data_ptr() for v in (qa, weight, xs, ws, gate, residual, gamma, scale, shift,
                                         output, normalized, counters)),
                m, n, k, int(x.dtype == torch.float16), norm.module.eps,
                torch.cuda.current_stream(x.device).cuda_stream)
        if status:
            raise RuntimeError(f'Joint residual/RMSNorm kernel launch failed: {status}')
        binding.forward_stats['residual_norm'] = binding.forward_stats.get('residual_norm', 0) + 1
        return output, normalized


def swiglu(x, bindings):
    from comfy_kitchen.backends import cuda as kitchen
    group = bindings[0].joint_info
    m, k, n = x.numel() // x.shape[-1], x.shape[-1], bindings[0].width
    with ExitStack() as stack:
        (wg, sg), (wu, su) = _weights(stack, bindings, x)
        qa, xs = kitchen.quantize_int4_rowwise_convrot64(x.reshape(m, k).contiguous(), group)
        output = torch.empty((*x.shape[:-1], n), device=x.device, dtype=x.dtype)
        with torch.cuda.device(x.device):
            status = library().k2_int4_swiglu(
                *(t.data_ptr() for t in (qa, wg, wu, xs, sg, su, output)),
                m, n, k, int(x.dtype == torch.float16), torch.cuda.current_stream(x.device).cuda_stream)
        if status:
            raise RuntimeError(f'Joint SwiGLU kernel launch failed: {status}')
        bindings[0].forward_stats['joint_swiglu'] = bindings[0].forward_stats.get('joint_swiglu', 0) + 1
        return output


def qkv(x, freqs, bindings, norms, heads, kvheads):
    from comfy import model_management as mm
    from comfy_kitchen.backends import cuda as kitchen
    m, k = x.shape[1], x.shape[-1]
    widths = [b.width for b in bindings]
    with ExitStack() as stack:
        weights = _weights(stack, bindings, x)
        qa, xs = kitchen.quantize_int4_rowwise_convrot64(x.reshape(m, k).contiguous(), bindings[0].joint_info)
        output = [torch.empty((*x.shape[:-1], n), device=x.device, dtype=x.dtype) for n in widths]
        nq, nk = [mm.cast_to(b.module.scale, dtype=torch.float32, device=x.device).contiguous() for b in norms]
        inputs = [qa, *(pair[0] for pair in weights), xs, *(pair[1] for pair in weights), nq, nk, freqs, *output]
        with torch.cuda.device(x.device):
            status = library().k2_int4_qkv(
                *(t.data_ptr() for t in inputs), m, widths[0], widths[1], k,
                int(x.dtype == torch.float16), norms[0].module.eps, norms[1].module.eps,
                torch.cuda.current_stream(x.device).cuda_stream)
        if status:
            raise RuntimeError(f'Joint QKV kernel launch failed: {status}')
        bindings[0].forward_stats['joint_qkv'] = bindings[0].forward_stats.get('joint_qkv', 0) + 1
        return (output[0].view(1, m, heads, 128).transpose(1, 2),
                output[1].view(1, m, kvheads, 128).transpose(1, 2), output[2], output[3])
