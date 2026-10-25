# SPDX-License-Identifier: GPL-3.0-only
"""Graph-visible native operators for one complete Krea2 compilation.

There is no torch.compile in this module. A managed linear operator includes its
original cast/offload lifetime; its weight storage/scale tensors are explicit graph
inputs. This keeps quantization and VBAR in the native backend without tracing
Python pointer access. Bindings live only as long as their owning compiled model.
"""
from contextlib import contextmanager
import itertools
import threading
import weakref

import torch
import torch.nn.functional as F


_BINDINGS = weakref.WeakValueDictionary()
_KEYS = itertools.count(1)
_LOCK = threading.RLock()
_LIB = torch.library.Library('k2_fullgraph', 'FRAGMENT')


def _managed(x, state, key, kind, width):
    binding = _BINDINGS.get(key)
    if binding is None:
        raise RuntimeError('K2 full graph: model binding has been released.')
    result = binding.normalized(x) if kind == 1 else None
    if result is None:
        result = binding.original(x)
    if kind == 3:
        return torch.cat(result, dim=1)
    if kind == 4:
        return torch.cat(result, dim=-1)
    return result


def _managed_fake(x, state, key, kind, width):
    if kind == 0:  # Linear
        shape, dtype = (*x.shape[:-1], width), x.dtype
    elif kind == 2:  # EmbedND
        shape, dtype = (x.shape[0], 1, x.shape[1], width // 2, 2, 2), torch.float32
    elif kind == 3:  # SimpleModulation
        shape, dtype = (x.shape[0], 2, x.shape[-1]), x.dtype
    else:
        shape, dtype = x.shape, x.dtype
    return torch.empty(shape, device=x.device, dtype=dtype)


# Preserve the native F.linear dispatch, including non-contiguous inputs. A
# custom_op below-autograd kernel can choose a different quantized handler.
_LIB.define('managed(Tensor x, Tensor[] state, int key, int kind, int width) -> Tensor')
_LIB.impl('managed', _managed, 'CompositeExplicitAutograd')
torch.library.register_fake('k2_fullgraph::managed')(_managed_fake)


@torch.library.custom_op('k2_fullgraph::activation', mutates_args=())
def activation(x: torch.Tensor, kind: str) -> torch.Tensor:
    return F.silu(x) if kind == 'silu' else torch.sigmoid(x)


@activation.register_fake
def _activation_fake(x, kind):
    return torch.empty_like(x)


@torch.library.custom_op('k2_fullgraph::modulation', mutates_args=())
def modulation(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    return (1 + scale) * x + shift


@modulation.register_fake
def _modulation_fake(x, scale, shift):
    return torch.empty_like(x, memory_format=torch.contiguous_format)


@torch.library.custom_op('k2_fullgraph::residual', mutates_args=())
def residual(x: torch.Tensor, gate: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return x + gate * value


@residual.register_fake
def _residual_fake(x, gate, value):
    return torch.empty_like(x, memory_format=torch.contiguous_format)


@torch.library.custom_op('k2_fullgraph::gated', mutates_args=())
def gated(x: torch.Tensor, gate: torch.Tensor, residual: torch.Tensor, key: int) -> torch.Tensor:
    binding = _BINDINGS.get(key)
    if binding is None:
        raise RuntimeError('K2 full graph: model binding has been released.')
    # This operator replaces the entire module call. Restore its real forward
    # while dispatching so native-method checks see the original implementation
    # and fallback executes Module.__call__ (including pre/post hooks) exactly once.
    with binding.native_forward():
        fused = binding.gated(x, gate, residual)
        return fused if fused is not None else residual + gate * binding.module(x)


@gated.register_fake
def _gated_fake(x, gate, residual, key):
    return torch.empty_like(residual, memory_format=torch.contiguous_format)


@torch.library.custom_op('k2_fullgraph::sdpa', mutates_args=())
def _sdpa_graph(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                enable_gqa: bool) -> torch.Tensor:
    out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, enable_gqa=enable_gqa)
    return out.transpose(1, 2).contiguous().view(q.shape[0], q.shape[2], q.shape[1] * q.shape[3])


@_sdpa_graph.register_fake
def _sdpa_fake(q, k, v, enable_gqa):
    return torch.empty((q.shape[0], q.shape[2], q.shape[1] * q.shape[3]), device=q.device, dtype=q.dtype)


def sdpa(module, q, k, v, enable_gqa):
    binding = module.forward
    if isinstance(binding, ManagedCall) and binding.native_key is not None:
        return torch.ops.k2_native.sdpa(q, k, v, enable_gqa)
    return _sdpa_graph(q, k, v, enable_gqa)


@torch.library.custom_op('k2_fullgraph::rope', mutates_args=())
def rope(q: torch.Tensor, k: torch.Tensor, freqs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from comfy.ldm.flux.math import apply_rope
    return apply_rope(q, k, freqs)


@rope.register_fake
def _rope_fake(q, k, freqs):
    return torch.empty_like(q), torch.empty_like(k)


@torch.library.custom_op('k2_fullgraph::timestep', mutates_args=())
def timestep(t: torch.Tensor, dim: int) -> torch.Tensor:
    from comfy.ldm.flux.layers import timestep_embedding
    return timestep_embedding(t, dim)


@timestep.register_fake
def _timestep_fake(t, dim):
    return torch.empty((t.shape[0], dim), device=t.device,
                       dtype=t.dtype if t.is_floating_point() else torch.float32)


def _storage(tensor):
    """Expose current backing tensors, without copying or dequantizing weights."""
    if hasattr(tensor, '_qdata'):
        import dataclasses
        values = [tensor._qdata]
        values.extend(getattr(tensor.params, field.name) for field in dataclasses.fields(tensor.params)
                      if isinstance(getattr(tensor.params, field.name), torch.Tensor))
    else:
        values = [tensor]
    return [value for value in values if value.device.type != 'meta']


class ManagedCall:
    def __init__(self, module, original, kind, width=0, *, fuse_epilogue=False, joint_kernels=True, optimize_io=True):
        self.module, self.original = module, original
        self.kind, self.width = kind, width
        self.fuse_epilogue = bool(fuse_epilogue)
        self.joint_kernels = bool(joint_kernels)
        self.optimize_io = bool(optimize_io)
        self.state = []
        self.key = next(_KEYS)
        _BINDINGS[self.key] = self
        self.fusion = None
        self.forward_stats = {}
        self.native_key = None
        self.native_loras = ()
        self.native_pre_scale = None
        self.native_trace_hooks = False
        self.joint_info = None
        self.joint_available = False
        self.joint_norm = False
        self.norm_info = False
        self.norm_width = 0

    def can_fuse_norm(self):
        from comfy.ldm.krea2.model import RMSNorm
        from comfy import model_management as mm
        from .int4_epilogue import has_hooks
        return (self.fuse_epilogue and not mm.in_training and not torch.is_grad_enabled()
                and getattr(self.original, '__func__', None) is RMSNorm.forward
                and not has_hooks(self.module))

    def normalized(self, x, scale=None, shift=None, cut=-1):
        if not x.is_cuda or not self.can_fuse_norm():
            return None
        from .forward_kernels import norm_mod
        out = norm_mod(x, self.module.scale, self.module.eps, scale, shift, cut)
        if out is not None:
            self.forward_stats['norm_mod'] = self.forward_stats.get('norm_mod', 0) + 1
        return out

    @contextmanager
    def native_forward(self):
        """Expose the saved forward only during the opaque operator's execution."""
        with _LOCK:
            existed = 'forward' in self.module.__dict__
            current = self.module.forward
            self.module.forward = self.original
            try:
                yield
            finally:
                if existed:
                    self.module.forward = current
                else:
                    delattr(self.module, 'forward')

    def gated(self, x, gate, residual):
        """GEMM with gate+residual in the epilogue; None keeps the native path."""
        if not self.fuse_epilogue or self.fusion is False:
            return None
        if self.fusion is None:
            from .int4_epilogue import EpilogueFusion
            try:
                self.fusion = EpilogueFusion()
            except (OSError, RuntimeError, ValueError) as error:
                import logging
                logging.warning('[K2 epilogue] fused kernel unavailable: %s', error)
                self.fusion = False
                return None
        return self.fusion.project(self.module, x, gate, residual)

    def prepare(self):
        from .joint_projection import metadata, library_path
        self.joint_info = metadata(self)
        self.joint_available = self.joint_kernels and bool(library_path())
        self.norm_info = self.kind == 1 and self.can_fuse_norm()
        self.norm_width = self.module.scale.numel() if self.kind == 1 else 0
        self.joint_norm = self.norm_info and self.norm_width == 128
        self.state = [value for param in self.module.parameters(recurse=False)
                      for value in _storage(param)]
        # Bypass LoRA factors are owned by its hook, rather than the base module.
        adapter = getattr(getattr(self.original, '__self__', None), 'adapter', None)
        if adapter is not None:
            self.state.extend(value for value in getattr(adapter, 'weights', ())
                              if isinstance(value, torch.Tensor))

    def __call__(self, x):
        if self.native_key is not None:
            base_input = x if self.native_pre_scale is None else x * self.native_pre_scale.to(dtype=x.dtype)
            value = torch.ops.k2_native.managed(base_input, self.native_key, self.kind, self.width)
            for up, down, strength in self.native_loras:
                delta = F.linear(F.linear(x, down.to(dtype=x.dtype)), up.to(dtype=x.dtype))
                value = value + delta * strength
        else:
            value = torch.ops.k2_fullgraph.managed(x, self.state, self.key, self.kind, self.width)
        if self.kind == 3:
            return value.chunk(2, dim=1)
        if self.kind == 4:
            return value.chunk(6, dim=-1)
        return value


@torch.library.custom_op('k2_fullgraph::norm_mod', mutates_args=())
def _norm_mod_graph(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor,
                    state: list[torch.Tensor], key: int, cut: int) -> torch.Tensor:
    binding = _BINDINGS[key]
    out = binding.normalized(x, scale, shift, cut)
    if out is not None:
        return out
    with binding.native_forward():
        y = binding.module(x)
    if cut < 0:
        return (1 + scale) * y + shift
    batch = x.shape[0]
    return torch.cat(((1 + scale[:batch]) * y[:, :cut] + shift[:batch],
                      (1 + scale[batch:]) * y[:, cut:] + shift[batch:]), dim=1)


@_norm_mod_graph.register_fake
def _norm_mod_graph_fake(x, scale, shift, state, key, cut):
    return torch.empty_like(x, memory_format=torch.contiguous_format)


def norm_mod(module, x, scale, shift, cut=-1):
    binding = module.forward
    if (isinstance(binding, ManagedCall) and binding.native_key is not None
            and binding.fuse_epilogue and not binding.native_trace_hooks):
        return torch.ops.k2_native.norm_mod(x, scale, shift, binding.native_key, cut)
    if isinstance(binding, ManagedCall) and binding.native_key is None:
        return _norm_mod_graph(x, scale, shift, binding.state, binding.key, cut)
    if cut < 0:
        return modulation(module(x), scale, shift)
    y, batch = module(x), x.shape[0]
    return torch.cat((modulation(y[:, :cut], scale[:batch], shift[:batch]),
                      modulation(y[:, cut:], scale[batch:], shift[batch:])), dim=1)


@torch.library.custom_op('k2_fullgraph::activate_mul', mutates_args=())
def _activate_mul_graph(a: torch.Tensor, b: torch.Tensor, kind: str) -> torch.Tensor:
    out = None
    if a.is_cuda:
        from .forward_kernels import activate_mul
        out = activate_mul(a, b, kind)
    return out if out is not None else (F.silu(a) if kind == 'silu' else torch.sigmoid(a)) * b


@_activate_mul_graph.register_fake
def _activate_mul_graph_fake(a, b, kind):
    return torch.empty_like(a)


def activation_product(module, a, b, kind):
    binding = module.forward
    if isinstance(binding, ManagedCall) and binding.native_key is not None and binding.fuse_epilogue:
        return torch.ops.k2_native.activate_mul(a, b, kind)
    if isinstance(binding, ManagedCall) and binding.fuse_epilogue:
        return _activate_mul_graph(a, b, kind)
    return activation(a, kind) * b


@torch.library.custom_op('k2_fullgraph::project_many', mutates_args=())
def _project_many_graph(x: torch.Tensor, state: list[torch.Tensor],
                        keys: list[int], widths: list[int]) -> list[torch.Tensor]:
    from .projection_fusion import project
    quantized = {}
    return [project(_BINDINGS[key], x, quantized)[0] for key in keys]


@_project_many_graph.register_fake
def _project_many_graph_fake(x, state, keys, widths):
    return [torch.empty((*x.shape[:-1], width), device=x.device, dtype=x.dtype) for width in widths]


@torch.library.custom_op('k2_fullgraph::joint_swiglu', mutates_args=())
def _joint_swiglu_graph(x: torch.Tensor, state: list[torch.Tensor],
                       keys: list[int], width: int) -> torch.Tensor:
    from .joint_projection import swiglu
    return swiglu(x, [_BINDINGS[key] for key in keys])


@_joint_swiglu_graph.register_fake
def _joint_swiglu_fake(x, state, keys, width):
    return torch.empty((*x.shape[:-1], width), device=x.device, dtype=x.dtype)


@torch.library.custom_op('k2_fullgraph::joint_qkv', mutates_args=())
def _joint_qkv_graph(x: torch.Tensor, freqs: torch.Tensor, state: list[torch.Tensor],
                     keys: list[int], norms: list[int], heads: int,
                     kvheads: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from .joint_projection import qkv
    return qkv(x, freqs, [_BINDINGS[key] for key in keys],
               [_BINDINGS[key] for key in norms], heads, kvheads)


@_joint_qkv_graph.register_fake
def _joint_qkv_fake(x, freqs, state, keys, norms, heads, kvheads):
    return _attention_inputs_graph_fake(x, freqs, state, keys,
                                       [heads * 128, kvheads * 128, kvheads * 128, heads * 128],
                                       norms, heads, kvheads)


def _joint_bindings(x, modules):
    from .joint_projection import input_supported
    bindings = [module.forward for module in modules]
    if not all(isinstance(b, ManagedCall) and b.joint_available and b.joint_info is not None
               for b in bindings):
        return None
    if not input_supported(x) or any(b.joint_info != bindings[0].joint_info for b in bindings):
        return None
    return bindings


@torch.library.custom_op('k2_fullgraph::residual_norm', mutates_args=())
def _residual_norm_graph(x: torch.Tensor, gate: torch.Tensor, residual: torch.Tensor,
                         scale: torch.Tensor, shift: torch.Tensor, state: list[torch.Tensor],
                         key: int, norm_key: int) -> tuple[torch.Tensor, torch.Tensor]:
    from .joint_projection import residual_norm
    return residual_norm(x, gate, residual, scale, shift, _BINDINGS[key], _BINDINGS[norm_key])


@_residual_norm_graph.register_fake
def _residual_norm_fake(x, gate, residual, scale, shift, state, key, norm_key):
    return (torch.empty_like(residual, memory_format=torch.contiguous_format),
            torch.empty_like(residual, memory_format=torch.contiguous_format))


def linear_gated_norm(module, norm_module, x, gate, residual, scale, shift):
    bindings = _joint_bindings(x, (module,))
    norm = norm_module.forward
    if (bindings is None or not isinstance(norm, ManagedCall) or not norm.norm_info
            or x.shape[0] != 1 or bindings[0].width % 128
            or norm.norm_width != bindings[0].width or residual.shape != (*x.shape[:-1], bindings[0].width)
            or any(v.dtype != x.dtype or v.device != x.device for v in (gate, residual, scale, shift))
            or gate.numel() != bindings[0].width or scale.numel() != bindings[0].width
            or shift.numel() != bindings[0].width):
        return None
    binding = bindings[0]
    if binding.native_key is not None and norm.native_key is not None:
        return torch.ops.k2_native.residual_norm(x, gate, residual, scale, shift,
                                                 binding.native_key, norm.native_key)
    if binding.native_key is not None or norm.native_key is not None:
        return None
    return _residual_norm_graph(x, gate, residual, scale, shift, binding.state + norm.state,
                                binding.key, norm.key)


def joint_swiglu(module, x):
    bindings = _joint_bindings(x, (module.gate, module.up))
    if bindings is None or bindings[0].width != bindings[1].width:
        return None
    if all(b.native_key is not None for b in bindings):
        return torch.ops.k2_native.joint_swiglu(x, [b.native_key for b in bindings], bindings[0].width)
    if any(b.native_key is not None for b in bindings):
        return None
    return _joint_swiglu_graph(x, [t for b in bindings for t in b.state],
                              [b.key for b in bindings], bindings[0].width)


def joint_attention_inputs(module, x, freqs):
    bindings = _joint_bindings(x, (module.wq, module.wk, module.wv, module.gate))
    norms = [m.forward for m in (module.qknorm.qnorm, module.qknorm.knorm)]
    if (bindings is None or x.shape[0] != 1 or freqs is None
            or module.kvheads <= 0 or module.heads < module.kvheads or module.heads % module.kvheads != 0
            or freqs.dtype != torch.float32 or not freqs.is_contiguous()
            or freqs.shape != (1, 1, x.shape[1], 64, 2, 2)
            or not all(isinstance(n, ManagedCall) and n.joint_norm for n in norms)
            or bindings[0].width != module.heads * 128
            or bindings[1].width != module.kvheads * 128
            or bindings[2].width != bindings[1].width or bindings[3].width != bindings[0].width):
        return None
    if all(b.native_key is not None for b in bindings + norms):
        return torch.ops.k2_native.joint_qkv(x, freqs, [b.native_key for b in bindings],
                                            [n.native_key for n in norms], module.heads, module.kvheads)
    if any(b.native_key is not None for b in bindings + norms):
        return None
    return _joint_qkv_graph(x, freqs, [t for b in bindings + norms for t in b.state],
                            [b.key for b in bindings], [n.key for n in norms], module.heads, module.kvheads)


def linears(x, modules):
    bindings = [module.forward for module in modules]
    if all(isinstance(b, ManagedCall) and b.native_key is not None for b in bindings):
        if any(b.native_trace_hooks or b.native_loras or b.native_pre_scale is not None for b in bindings):
            return [module(x) for module in modules]
        return torch.ops.k2_native.project_many(x, [b.native_key for b in bindings],
                                                [b.width for b in bindings])
    if all(isinstance(b, ManagedCall) and b.fuse_epilogue for b in bindings):
        return _project_many_graph(x, [t for b in bindings for t in b.state],
                                   [b.key for b in bindings], [b.width for b in bindings])
    return [module(x) for module in modules]


@torch.library.custom_op('k2_fullgraph::attention_inputs', mutates_args=())
def _attention_inputs_graph(x: torch.Tensor, freqs: torch.Tensor | None,
        state: list[torch.Tensor], keys: list[int], widths: list[int], norm_keys: list[int], heads: int,
        kvheads: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from .projection_fusion import project
    quantized = {}
    norms = [_BINDINGS[key] for key in norm_keys]
    native_norms = x.is_cuda and all(n.can_fuse_norm() for n in norms)
    if native_norms:
        from .forward_kernels import qk_post
    outputs, done = [], []
    for i, key in enumerate(keys):
        norm = norms[i] if native_norms and i < 2 else None
        value, processed = project(_BINDINGS[key], x, quantized, norm, freqs)
        outputs.append(value)
        done.append(processed)
    q, k, v, gate = outputs
    pair = qk_post(q, k, norms[0].module.scale, norms[1].module.scale, freqs,
                   norms[0].module.eps, norms[1].module.eps, heads, kvheads,
                   done[0], done[1]) if native_norms else None
    if pair is not None:
        q, k = pair
        return (q.view(*x.shape[:2], heads, -1).transpose(1, 2),
                k.view(*x.shape[:2], kvheads, -1).transpose(1, 2), v, gate)
    if any(done[:2]):
        raise RuntimeError('K2 Q/K epilogue produced an unsupported postprocessing layout')
    q, k = q.view(*x.shape[:2], heads, -1).transpose(1, 2), k.view(*x.shape[:2], kvheads, -1).transpose(1, 2)
    # Preserve extensions through the normal module call.
    for i, value in enumerate((q, k)):
        with norms[i].native_forward():
            outputs[i] = norms[i].module(value)
    q, k = outputs[:2]
    if freqs is not None:
        from comfy.ldm.flux.math import apply_rope
        q, k = apply_rope(q, k, freqs)
    return (q.transpose(1, 2).contiguous().transpose(1, 2),
            k.transpose(1, 2).contiguous().transpose(1, 2), v, gate)


@_attention_inputs_graph.register_fake
def _attention_inputs_graph_fake(x, freqs, state, keys, widths, norm_keys, heads, kvheads):
    b, length = x.shape[:2]
    return (torch.empty((b, length, heads, widths[0] // heads), device=x.device, dtype=x.dtype).transpose(1, 2),
            torch.empty((b, length, kvheads, widths[1] // kvheads), device=x.device, dtype=x.dtype).transpose(1, 2),
            torch.empty((b, length, widths[2]), device=x.device, dtype=x.dtype),
            torch.empty((b, length, widths[3]), device=x.device, dtype=x.dtype))


def attention_inputs(module, x, freqs):
    bindings = [m.forward for m in (module.wq, module.wk, module.wv, module.gate)]
    norms = [m.forward for m in (module.qknorm.qnorm, module.qknorm.knorm)]
    if all(isinstance(b, ManagedCall) and b.native_key is not None for b in bindings + norms):
        if any(b.native_trace_hooks or b.native_loras or b.native_pre_scale is not None for b in bindings + norms):
            return None
        return torch.ops.k2_native.attention_inputs(x, freqs, [b.native_key for b in bindings],
                                                   [b.width for b in bindings], [n.native_key for n in norms],
                                                   module.heads, module.kvheads)
    if all(isinstance(b, ManagedCall) and b.fuse_epilogue for b in bindings + norms):
        return _attention_inputs_graph(x, freqs, [t for b in bindings + norms for t in b.state],
                                       [b.key for b in bindings],
                                       [b.width for b in bindings], [n.key for n in norms],
                                       module.heads, module.kvheads)
    return None


def linear_gated(module, x, gate, residual):
    """Fused projection+gate+residual when enabled and the module is bound."""
    binding = module.forward
    if not isinstance(binding, ManagedCall):
        return graph_ops_native(module, x, gate, residual)
    if binding.native_key is not None:
        if binding.native_trace_hooks or binding.native_loras or binding.native_pre_scale is not None:
            return residual + gate * module(x)
        return torch.ops.k2_native.gated(x, gate, residual, binding.native_key)
    # Keep the module-call/hook boundary in both modes. The binding owns the
    # switch; disabled bindings never construct or launch the fused kernel.
    return torch.ops.k2_fullgraph.gated(x, gate, residual, binding.key)


def graph_ops_native(module, x, gate, residual):
    return residual + gate * module(x)


class ModelBindings:
    """Temporarily install graph adapters; restore every method after the call."""
    def __init__(self, owner, *, fuse_epilogue=False, joint_kernels=True, optimize_io=True,
                 fuse_input_quant=True, native_execution=True):
        self.owner = owner
        self.fuse_epilogue = bool(fuse_epilogue)
        self.joint_kernels = bool(joint_kernels)
        self.optimize_io = bool(optimize_io)
        self.fuse_input_quant = bool(fuse_input_quant)
        self.native_execution = bool(native_execution)
        self.adapters = {}

    @contextmanager
    def active(self):
        from comfy.ldm.krea2 import model as krea
        from comfy.ldm.flux.layers import EmbedND
        from .graph_model import attention, mlp, dit, block, last, _norm_chain_block, _norm_chain_last, _norm_quant_block
        from .norm_quant import library_path as norm_quant_library
        from functools import partial
        changes, current = [], {}
        with _LOCK:
            try:
                for module in self.owner.modules():
                    original = module.forward
                    kind, width = None, 0
                    if hasattr(module, 'in_features') and hasattr(module, 'out_features') and hasattr(module, 'weight'):
                        kind, width = 0, module.out_features
                    elif isinstance(module, krea.RMSNorm):
                        kind = 1
                    elif isinstance(module, EmbedND):
                        kind, width = 2, sum(module.axes_dim)
                    elif isinstance(module, krea.SimpleModulation):
                        kind = 3
                    elif isinstance(module, krea.DoubleSharedModulation):
                        kind = 4
                    elif isinstance(module, torch.nn.GELU):
                        kind = 5
                    replacement = None
                    if kind is not None:
                        previous = self.adapters.get(module)
                        if previous is None or previous.original != original:
                            previous = ManagedCall(module, original, kind, width,
                                                   fuse_epilogue=self.fuse_epilogue, joint_kernels=self.joint_kernels,
                                                   optimize_io=self.optimize_io)
                        previous.prepare()
                        current[module] = previous
                        replacement = previous
                    elif isinstance(module, krea.Attention) and getattr(original, '__func__', None) is krea.Attention.forward:
                        replacement = self.adapters.get(module)
                        fuse_qk_inputs = (getattr(module.qknorm.forward, '__func__', None) is krea.QKNorm.forward
                                          and not module.qknorm._forward_hooks
                                          and not module.qknorm._forward_pre_hooks)
                        if (replacement is None
                                or replacement.keywords.get('fuse_qk_inputs') != fuse_qk_inputs):
                            replacement = partial(attention, module, fuse_qk_inputs=fuse_qk_inputs)
                        current[module] = replacement
                    elif isinstance(module, krea.SwiGLU) and getattr(original, '__func__', None) is krea.SwiGLU.forward:
                        replacement = self.adapters.get(module)
                        if replacement is None:
                            replacement = partial(mlp, module)
                        current[module] = replacement
                    elif isinstance(module, krea.SingleStreamBlock) and getattr(original, '__func__', None) is krea.SingleStreamBlock.forward:
                        replacement = self.adapters.get(module)
                        if replacement is None:
                            replacement = partial(block, module)
                        current[module] = replacement
                    elif isinstance(module, krea.LastLayer) and getattr(original, '__func__', None) is krea.LastLayer.forward:
                        replacement = self.adapters.get(module)
                        if replacement is None:
                            replacement = partial(last, module)
                        current[module] = replacement
                    if replacement is not None:
                        changes.append((module, 'forward', original, 'forward' in module.__dict__))
                        module.forward = replacement
                original_body = self.owner._forward
                if getattr(original_body, '__func__', None) is krea.SingleStreamDiT._forward:
                    body_key = '_model_body'
                    replacement = self.adapters.get(body_key)
                    chain = bool(self.fuse_epilogue and self.joint_kernels and len(self.owner.blocks)
                                 and all(_norm_chain_block(b) for b in self.owner.blocks))
                    chain_last = chain and _norm_chain_last(self.owner.last)
                    chain_quant = bool(chain and self.fuse_input_quant and self.native_execution
                                       and norm_quant_library() and all(_norm_quant_block(b) for b in self.owner.blocks))
                    if (replacement is None or replacement.keywords.get('allow_norm_chain') != chain
                            or replacement.keywords.get('allow_last_norm') != chain_last
                            or replacement.keywords.get('allow_input_quant') != chain_quant):
                        replacement = partial(dit, self.owner, allow_norm_chain=chain,
                                              allow_last_norm=chain_last, allow_input_quant=chain_quant)
                    current[body_key] = replacement
                    changes.append((self.owner, '_forward', original_body, '_forward' in self.owner.__dict__))
                    self.owner._forward = replacement
                self.adapters = current
                yield
            finally:
                for module, name, original, existed in reversed(changes):
                    if existed:
                        setattr(module, name, original)
                    else:
                        delattr(module, name)
                for adapter in current.values():
                    if isinstance(adapter, ManagedCall):
                        adapter.state = []
