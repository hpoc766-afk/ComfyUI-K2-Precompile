"""Optional SM89 ConvRot W4A4 GEMM + rounded gate + residual (CUTLASS epilogue).

The native kitchen quantizer is reused. No dense weight materialization,
weight caching, global monkey patches or LoRA folding occur here.
try_project() returns None when the fused path does not apply.
"""
# SPDX-License-Identifier: GPL-3.0-only
import ctypes
import functools
import hashlib
import logging
import os
from pathlib import Path
from types import FunctionType, MethodType

import torch


@functools.lru_cache(maxsize=1)
def load_kernel():
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256((root / "kernels/int4_epilogue.cu").read_bytes()).hexdigest()[:16]
    path = root / ".kernel-build" / (f"int4_epilogue_{digest}" + (".dll" if os.name == "nt" else ".so"))
    if not path.is_file():
        raise RuntimeError("K2 INT4 融合内核尚未构建。请运行 scripts/build_int4_epilogue.py。")
    lib = ctypes.CDLL(str(path))
    fn = lib.k2_int4_gated_residual
    fn.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int] * 5 + [ctypes.c_void_p]
    fn.restype = ctypes.c_int
    return lib


def packed_residual(qact, qweight, xscale, wscale, gate, residual, *, config=0):
    """Launch on the current PyTorch stream. Caller supplies contiguous tensors."""
    m, n = residual.shape
    k = qact.shape[1] * 2
    if not (qact.shape == (m, k // 2) and qweight.shape == (n, k // 2)
            and xscale.numel() == m and wscale.numel() == n and gate.numel() == n
            and residual.dtype in (torch.bfloat16, torch.float16) and gate.dtype == residual.dtype
            and qact.dtype == qweight.dtype == torch.int8
            and xscale.dtype == wscale.dtype == torch.float32 and config in (0, 1)
            and all(t.is_cuda and t.device == residual.device and t.is_contiguous()
                    for t in (qact, qweight, xscale, wscale, gate, residual))):
        raise ValueError("Unsupported K2 INT4 residual tensor layout")
    if torch.cuda.get_device_capability(residual.device) != (8, 9):
        raise ValueError("K2 INT4 residual kernel currently supports SM89 only")
    result = torch.empty_like(residual)
    with torch.cuda.device(residual.device):
        status = load_kernel().k2_int4_gated_residual(
            *(t.data_ptr() for t in (qact, qweight, xscale, wscale, gate, residual, result)),
            m, n, k, int(residual.dtype == torch.float16), config,
            torch.cuda.current_stream(residual.device).cuda_stream,
        )
    if status:
        raise RuntimeError(f"K2 INT4 residual CUTLASS launch failed: status={status}")
    return result


def has_hooks(module):
    return bool(module._forward_hooks or module._forward_pre_hooks)


@functools.lru_cache(maxsize=1)
def _native_projection_methods():
    from comfy import ops

    # The factory creates fresh classes/functions, but their code objects are shared.
    # Capture the native implementations, not the class of a candidate projection.
    cls = ops.mixed_precision_ops().Linear
    return tuple(
        (name, fn.__code__, fn.__globals__, fn.__defaults__, fn.__kwdefaults__)
        for name in ("forward", "forward_comfy_cast_weights", "_forward")
        for fn in (getattr(cls, name),)
    )


def is_native_projection(linear):
    """Check resolved methods on every call so extension injection/ejection is visible."""
    for name, code, globals_, defaults, kwdefaults in _native_projection_methods():
        method = getattr(linear, name, None)
        if not isinstance(method, MethodType) or method.__self__ is not linear:
            return False
        fn = method.__func__
        if (not isinstance(fn, FunctionType) or fn.__code__ is not code
                or fn.__globals__ is not globals_ or fn.__closure__ is not None
                or fn.__defaults__ != defaults or fn.__kwdefaults__ != kwdefaults):
            return False
    return True


class EpilogueFusion:
    def __init__(self):
        load_kernel()
        self.fused = 0
        self.fallbacks = {}

    def fallback(self, reason):
        self.fallbacks[reason] = self.fallbacks.get(reason, 0) + 1

    def project(self, linear, x, gate, residual):
        """Return the fused result, or None to keep the native computation."""
        from comfy import ops, model_management
        from comfy_kitchen.tensor import QuantizedTensor
        from comfy_kitchen.tensor.convrot_w4a4 import TensorCoreConvRotW4A4Layout

        def original(reason):
            self.fallback(reason)
            return None

        if torch.is_grad_enabled() or model_management.in_training:
            return original("training/grad enabled")
        if (getattr(linear, "comfy_force_forward", False)
                or getattr(linear, "_full_precision_mm", False)
                or getattr(linear, "comfy_force_cast_weights", False)
                or getattr(linear, "pre_quant_scale", None) is not None
                or not is_native_projection(linear)
                or has_hooks(linear)
                or getattr(linear, "weight_function", ()) or getattr(linear, "bias_function", ())):
            return original("projection hooks/LoRA/custom weights")
        if (not x.is_cuda or x.ndim != 3 or x.shape[0] != 1
                or x.dtype not in (torch.bfloat16, torch.float16)
                or residual.dtype != x.dtype or gate.dtype != x.dtype
                or residual.shape != (1, x.shape[1], linear.out_features)
                or gate.shape != (1, 1, linear.out_features)
                or residual.device != x.device or gate.device != x.device
                or torch.cuda.get_device_capability(x.device) != (8, 9)):
            return original("device/dtype/batch/gate layout")
        weight = linear.weight
        if not isinstance(weight, QuantizedTensor) or weight.layout_cls is not TensorCoreConvRotW4A4Layout:
            return original("non-ConvRot projection")
        p = weight.params
        from comfy_kitchen.backends.eager.svdquant import _INT4_GROUP_SIZE
        from comfy_kitchen.backends import cuda as kitchen
        if (p.transposed or p.linear_dtype != "int4" or p.quant_group_size != _INT4_GROUP_SIZE
                or p.convrot_groupsize not in (16, 64, 256) or linear.bias is not None
                or x.shape[-1] % 256 or linear.out_features % 8 or x.shape[1] == 0
                or not kitchen._convrot_int4_fused_shared_memory_fits(x, x.shape[-1], p.convrot_groupsize)):
            return original("unsupported ConvRot configuration")
        # Matches ComfyUI's weight-only quantized path, including VBAR/offload lifetime.
        ops.run_every_op()
        with ops.CastBiasWeightContext(
            linear, input=None, dtype=linear.weight.dtype, device=x.device,
            bias_dtype=x.dtype, offloadable=True, compute_dtype=x.dtype, want_requant=True,
        ) as (weight, bias):
            weight = weight.to(dtype=x.dtype)
            if not isinstance(weight, QuantizedTensor) or bias is not None:
                raise RuntimeError("K2 residual fusion: cast changed ConvRot weight layout")
            qweight, scales = weight.layout_cls.get_plain_tensors(weight)
            qact, xscale = kitchen.quantize_int4_rowwise_convrot64(
                x.reshape(-1, x.shape[-1]).contiguous(), p.convrot_groupsize)
            result = packed_residual(qact, qweight.contiguous(), xscale,
                                     scales.float().contiguous(), gate.reshape(-1).contiguous(),
                                     residual.reshape(-1, residual.shape[-1]).contiguous())
        self.fused += 1
        return result.view_as(residual)
