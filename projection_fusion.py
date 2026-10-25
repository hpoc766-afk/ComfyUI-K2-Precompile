# SPDX-License-Identifier: GPL-3.0-only
"""Grouped projections with a call-local activation quantization buffer.

Each weight retains its original ComfyUI cast/offload context. No quantized
activation or weight is cached between forwards, models, or CUDA streams.
"""
import ctypes
import functools
import hashlib
import os
from pathlib import Path

import torch


@functools.lru_cache(maxsize=1)
def _qk_kernel():
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256((root / 'kernels/qk_norm_rope.cu').read_bytes()).hexdigest()[:16]
    path = root / '.kernel-build' / (f'qk_norm_rope_{digest}' + ('.dll' if os.name == 'nt' else '.so'))
    lib = ctypes.CDLL(str(path))
    fn = lib.k2_int4_qk_norm_rope
    fn.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int] * 4 + [ctypes.c_float, ctypes.c_void_p]
    fn.restype = ctypes.c_int
    return lib


def _count(binding, name):
    binding.forward_stats[name] = binding.forward_stats.get(name, 0) + 1


def project(binding, x, quantized, norm=None, freqs=None):
    """Return (projection, norm_and_rope_done), preserving module extensions."""
    from comfy import ops, model_management as mm
    from comfy_kitchen.tensor import QuantizedTensor
    from comfy_kitchen.tensor.convrot_w4a4 import TensorCoreConvRotW4A4Layout
    from comfy_kitchen.backends.eager.svdquant import _INT4_GROUP_SIZE
    from .int4_epilogue import is_native_projection, has_hooks
    linear = binding.module
    with binding.native_forward():
        native = (not torch.is_grad_enabled() and not mm.in_training
                  and x.is_cuda and x.dtype in (torch.float16, torch.bfloat16)
                  and x.ndim == 3 and not has_hooks(linear) and is_native_projection(linear)
                  and not getattr(linear, 'comfy_force_forward', False)
                  and not getattr(linear, '_full_precision_mm', False)
                  and not getattr(linear, 'comfy_force_cast_weights', False)
                  and getattr(linear, 'pre_quant_scale', None) is None
                  and not getattr(linear, 'weight_function', ())
                  and not getattr(linear, 'bias_function', ())
                  and getattr(linear, 'layout_type', None) is not None
                  and getattr(linear, 'quant_format', None) == 'convrot_w4a4')
        weight = linear.weight
        if native and isinstance(weight, QuantizedTensor) and weight.layout_cls is TensorCoreConvRotW4A4Layout:
            p = weight.params
            native = (not p.transposed and p.linear_dtype == 'int4'
                      and p.quant_group_size == _INT4_GROUP_SIZE
                      and p.convrot_groupsize in (16, 64, 256)
                      and torch.cuda.get_device_capability(x.device) == (8, 9))
            if native:
                from comfy_kitchen.backends import cuda as kitchen
                native = kitchen._convrot_int4_fused_shared_memory_fits(
                    x, x.shape[-1], p.convrot_groupsize)
        else:
            native = False
        if not native:
            out = linear(x)
            # Module hooks may modify x in place. Later projections must
            # quantize the current input, rather than reuse an earlier buffer.
            quantized.clear()
            if torch._C._is_alias_of(out, x):
                out = out.clone()
            return out, False
        ops.run_every_op()
        with ops.CastBiasWeightContext(linear, input=None, dtype=linear.weight.dtype,
                device=x.device, bias_dtype=x.dtype, offloadable=True,
                compute_dtype=x.dtype, want_requant=True) as (weight, bias):
            weight = weight.to(dtype=x.dtype)
            # Cast hooks may change layout: use this cast weight for native math.
            if not isinstance(weight, QuantizedTensor) or weight.layout_cls is not TensorCoreConvRotW4A4Layout:
                quantized.clear()
                return linear._forward(x, weight, bias), False
            p = weight.params
            if (p.transposed or p.linear_dtype != 'int4'
                    or p.quant_group_size != _INT4_GROUP_SIZE
                    or p.convrot_groupsize not in (16, 64, 256)
                    or not kitchen._convrot_int4_fused_shared_memory_fits(
                        x, x.shape[-1], p.convrot_groupsize)):
                quantized.clear()
                return linear._forward(x, weight, bias), False
            cache_key = (p.convrot_groupsize, p.quant_group_size, p.linear_dtype)
            if cache_key not in quantized:
                quantized[cache_key] = kitchen.quantize_int4_rowwise_convrot64(
                    x.reshape(-1, x.shape[-1]).contiguous(), p.convrot_groupsize)
                _count(binding, 'input_quantizations')
            else:
                _count(binding, 'shared_quantizations')
            qa, xs = quantized[cache_key]
            qw, ws = weight.layout_cls.get_plain_tensors(weight)
            fused = (norm is not None and freqs is not None and bias is None
                     and x.shape[0] == 1 and norm.module.scale.numel() == 128
                     and freqs.dtype == torch.float32 and freqs.is_contiguous()
                     and freqs.shape == (1, 1, x.shape[1], 64, 2, 2)
                     and linear.out_features % 128 == 0 and x.shape[-1] % 256 == 0)
            if fused:
                try:
                    kernel = _qk_kernel()
                except (OSError, FileNotFoundError):
                    kernel = None
                if kernel is not None:
                    nw = mm.cast_to(norm.module.scale, dtype=torch.float32, device=x.device).contiguous()
                    out = torch.empty((*x.shape[:-1], linear.out_features), device=x.device, dtype=x.dtype)
                    with torch.cuda.device(x.device):
                        status = kernel.k2_int4_qk_norm_rope(
                            *(t.data_ptr() for t in (qa, qw, xs, ws, nw, freqs, out)),
                            x.shape[1], linear.out_features, x.shape[-1],
                            int(x.dtype == torch.float16), norm.module.eps,
                            torch.cuda.current_stream(x.device).cuda_stream)
                    if status:
                        raise RuntimeError(f'K2 fused Q/K launch failed: {status}')
                    _count(binding, 'qk_gemm_norm_rope')
                    return out, True
            out = kitchen.int4_linear(qa, qw, xs, ws, bias=bias, out_dtype=x.dtype)
            _count(binding, 'shared_int4_gemm')
            return out.reshape(*x.shape[:-1], linear.out_features), False
