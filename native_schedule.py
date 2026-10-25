# SPDX-License-Identifier: GPL-3.0-only
"""Build-time registration for a native K2 execution plan.

The hot entry is a C++ wrapper. Python is used to capture/compile and when the
model configuration or input signature changes, not to dispatch its operators.
"""
from contextlib import contextmanager
import functools
import hashlib
import json
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent
_DLL_PATHS = []


def compiler_environment():
    if os.name == 'nt':
        import shutil
        if shutil.which('cl') is None:
            from setuptools._distutils._msvccompiler import _get_vc_env
            values = _get_vc_env('x64')
            for name in ('path', 'include', 'lib', 'libpath'):
                if name in values:
                    os.environ[name.upper()] = values[name]


@functools.lru_cache(maxsize=1)
def runtime_module():
    compiler_environment()
    from torch.utils.cpp_extension import load, CUDA_HOME
    if os.name == 'nt' and CUDA_HOME:
        import ctypes
        folder = Path(CUDA_HOME) / 'bin'
        _DLL_PATHS.append(os.add_dll_directory(str(folder)))
        for pattern in ('cudart64_*.dll', 'cublasLt64_*.dll', 'cublas64_*.dll'):
            for library in folder.glob(pattern):
                _DLL_PATHS.append(ctypes.WinDLL(str(library)))
    digest = hashlib.sha256((ROOT / 'native_runtime.cpp').read_bytes() + torch.__version__.encode()).hexdigest()[:12]
    build = ROOT / '.kernel-build' / ('runtime_' + digest)
    build.mkdir(parents=True, exist_ok=True)
    module = load(name='k2_native_' + digest, sources=[str(ROOT / 'native_runtime.cpp')],
                  build_directory=str(build), with_cuda=True,
                  extra_cflags=['/O2', '/bigobj'] if os.name == 'nt' else ['-O3'],
                  verbose=False)
    register_fakes()
    return module


@functools.lru_cache(maxsize=1)
def register_fakes():
    from . import graph_ops as g
    from . import input_quant_ops as iq
    def managed(x, key, kind, width):
        return g._managed_fake(x, [], key, kind, width)
    def norm(x, scale, shift, key, cut):
        return torch.empty_like(x, memory_format=torch.contiguous_format)
    def many(x, keys, widths):
        return g._project_many_graph_fake(x, [], keys, widths)
    def inputs(x, freqs, keys, widths, norms, heads, kvheads):
        return g._attention_inputs_graph_fake(x, freqs, [], keys, widths, norms, heads, kvheads)
    def qkv_prepared(x, qa, qs, freqs, keys, norms, heads, kvheads):
        return g._joint_qkv_fake(x, freqs, [], keys, norms, heads, kvheads)
    def swiglu_prepared(x, qa, qs, keys, width):
        return g._joint_swiglu_fake(x, [], keys, width)
    def activated(x, kind):
        return torch.empty_like(x)
    def residual_norm(x, gate, residual, scale, shift, key, norm_key):
        return g._residual_norm_fake(x, gate, residual, scale, shift, [], key, norm_key)
    def joint_swiglu(x, keys, width):
        return g._joint_swiglu_fake(x, [], keys, width)
    def joint_qkv(x, freqs, keys, norms, heads, kvheads):
        return g._joint_qkv_fake(x, freqs, [], keys, norms, heads, kvheads)
    def product(a, b, kind):
        return torch.empty_like(a)
    def mod(x, scale, shift):
        return torch.empty_like(x, memory_format=torch.contiguous_format)
    def residual(x, gate, value):
        return torch.empty_like(x, memory_format=torch.contiguous_format)
    def rope(q, k, freqs):
        return (torch.empty(q.shape, device=q.device, dtype=q.dtype),
                torch.empty(k.shape, device=k.device, dtype=k.dtype))
    def step(t, dim):
        return g._timestep_fake(t, dim)
    for name, fn in {'norm_quant': iq.fake_norm_quant, 'gated_norm_quant': iq.fake_gated_norm_quant,
                     'qkv_prepared': qkv_prepared, 'swiglu_prepared': swiglu_prepared,
                     'sdpa': g._sdpa_fake, 'managed': managed, 'norm_mod': norm, 'project_many': many,
                     'residual_norm': residual_norm,
                     'joint_swiglu': joint_swiglu, 'joint_qkv': joint_qkv,
                     'attention_inputs': inputs, 'activation': activated,
                     'activate_mul': product, 'modulation': mod, 'residual': residual,
                     'rope': rope, 'timestep': step,
                     'gated': lambda x, gate, residual, key: torch.empty_like(residual)}.items():
        torch.library.register_fake('k2_native::' + name)(fn)


def _library(name, source):
    digest = hashlib.sha256((ROOT / 'kernels' / source).read_bytes()).hexdigest()[:16]
    path = ROOT / '.kernel-build' / (name + '_' + digest + ('.dll' if os.name == 'nt' else '.so'))
    return str(path) if path.is_file() else ''


class NativePlan:
    def __init__(self, bindings, device):
        from . import graph_ops
        self.module = runtime_module()
        manifest = json.loads((ROOT / '.kernel-build/native_dispatch.json').read_text(encoding='utf-8'))
        expected = hashlib.sha256((ROOT / 'kernels/native_dispatch.cu').read_bytes()).hexdigest()
        if manifest['source_hash'] != expected:
            raise RuntimeError('Native CUDA primitives changed; rebuild scripts/build_native_schedule.py')
        import comfy_aimdo.control
        aimdo = getattr(comfy_aimdo.control.lib, '_name', '') or ''
        self.module.configure(str(ROOT / '.kernel-build' / manifest['library']),
                              _library('int4_epilogue', 'int4_epilogue.cu'),
                              _library('qk_norm_rope', 'qk_norm_rope.cu'), aimdo)
        if bindings.fuse_epilogue and bindings.joint_kernels:
            self.module.configure_joint(_library('joint_projection', 'joint_projection.cu'))
        if bindings.fuse_input_quant and bindings.native_execution:
            from .norm_quant import library_path
            self.module.configure_norm_quant(library_path())
        self.bindings = bindings
        self.device = torch.device(device)
        self.keys = []
        self.keepalive = []
        self.closed = False
        self.tuned_problems = set()
        self.tuning_decisions = []
        try:
            for adapter in bindings.adapters.values():
                if isinstance(adapter, graph_ops.ManagedCall):
                    self.register(adapter)
            self.lookahead()
        except Exception:
            self.close()
            raise

    def register(self, binding):
        from comfy import ops, memory_management
        from comfy_kitchen.tensor import QuantizedTensor
        from .int4_epilogue import is_native_projection
        module = binding.module
        from .int4_epilogue import has_hooks
        binding.native_trace_hooks = has_hooks(module)
        binding.native_loras = ()
        key = binding.key
        kind = binding.kind
        cpu, gpu, vbar, fmt, group = [], [], [], 0, 256
        if kind == 0:
            from comfy.weight_adapter.bypass import BypassForwardHook
            from comfy.weight_adapter.lora import LoRAAdapter
            original = binding.original
            factors = []
            while isinstance(getattr(original, '__self__', None), BypassForwardHook):
                hook = original.__self__
                adapter = hook.adapter
                if type(adapter) is not LoRAAdapter or any(v is not None for v in adapter.weights[3:]):
                    raise RuntimeError('Native scheduling supports ordinary linear LoRA bypass factors')
                up, down, alpha = adapter.weights[:3]
                strength = (alpha / down.shape[0] if alpha is not None else 1.) * adapter.multiplier
                factors.append((up.to(device=self.device), down.to(device=self.device), float(strength)))
                original = hook.original_forward
            binding.native_loras = tuple(reversed(factors))
            with binding.native_forward():
                saved = module.forward
                module.forward = original
                try:
                    valid = is_native_projection(module)
                    if not valid and not isinstance(module.weight, QuantizedTensor):
                        method = getattr(module.forward, '__func__', None)
                        simple = (torch.nn.Linear.forward, ops.disable_weight_init.Linear.forward)
                        cast = getattr(getattr(module, 'forward_comfy_cast_weights', None), '__func__', None)
                        valid = (method in simple and
                                 (method is torch.nn.Linear.forward or
                                  cast is ops.disable_weight_init.Linear.forward_comfy_cast_weights))
                finally:
                    module.forward = saved
                if not valid:
                    raise RuntimeError('Native scheduling needs a traceable native projection')
            if (isinstance(module.weight, QuantizedTensor) and
                    (getattr(module, '_full_precision_mm', False) or
                     getattr(module, 'comfy_force_cast_weights', False))):
                raise RuntimeError('Native scheduling requires the selected projection compute format')
            pre_scale = getattr(module, 'pre_quant_scale', None)
            binding.native_pre_scale = None if pre_scale is None else pre_scale.to(device=self.device)
            if any(getattr(module, name, None) for name in ('weight_function', 'bias_function',
                   'weight_lowvram_function', 'bias_lowvram_function')):
                raise RuntimeError('Native scheduling needs prepared static projection weights')
            ops.materialize_meta_param(module, ['weight', 'bias'])
            weight, bias = module.weight, module.bias
            if isinstance(weight, QuantizedTensor):
                raw, scale = weight.layout_cls.get_plain_tensors(weight)
                params = weight.params
                if getattr(params, 'transposed', False):
                    raise RuntimeError('Native scheduling does not accept transposed quantized storage')
                name = weight.layout_cls.__name__
                if 'ConvRotW4A4' in name:
                    if params.linear_dtype != 'int4':
                        raise RuntimeError('Native W4A4 scheduling requires INT4 compute')
                    fmt, group = 1, params.convrot_groupsize
                elif 'INT8' in name:
                    fmt, group = 2, params.convrot_groupsize if params.convrot else 0
                elif raw.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                    fmt = 3 if raw.dtype == torch.float8_e4m3fn else 4
                else:
                    raise RuntimeError(f'Unsupported native weight layout: {name}')
                cpu = [raw, scale]
            else:
                cpu = [weight]
            if bias is not None:
                cpu.append(bias)
            allocation = getattr(module, '_v', None)
            if allocation is not None and weight.device != self.device:
                import comfy_aimdo.torch
                vb, address, size = allocation
                self.keepalive.append(vb)
                buffer = comfy_aimdo.torch.aimdo_to_tensor(allocation, self.device)
                w, b = memory_management.interpret_gathered_like([weight, bias], buffer)
                gpu = list(w.layout_cls.get_plain_tensors(w)) if isinstance(w, QuantizedTensor) else [w]
                if b is not None:
                    gpu.append(b)
                vbar = [int(vb._devctx), int(vb._ptr), address, vb.base_addr, size]
                # Native copies retain the same payload/layout. Invalidate the
                # Python residency claim once when handing the plan ownership.
                module._v_signature = None
            else:
                gpu = [t.to(device=self.device, non_blocking=True) for t in cpu]
        elif kind == 1:
            from comfy.ldm.krea2.model import RMSNorm
            if getattr(binding.original, '__func__', None) is not RMSNorm.forward:
                raise RuntimeError('Native scheduling needs the native RMSNorm forward')
            cpu = [module.scale]
            gpu = [module.scale.to(device=self.device, dtype=torch.float32)]
        elif kind in (3, 4):
            cpu = [module.lin]
            gpu = [module.lin.to(device=self.device)]
        input_scale = getattr(module, 'input_scale', None)
        if input_scale is None:
            input_scale = torch.empty(0)
        elif not isinstance(input_scale, torch.Tensor):
            input_scale = torch.tensor(input_scale, device=self.device, dtype=torch.float32)
        else:
            input_scale = input_scale.to(device=self.device, dtype=torch.float32)
        self.keepalive.extend(cpu)
        self.module.add_binding(key, kind, fmt, cpu, gpu, vbar, group,
                                binding.fuse_epilogue, float(getattr(module, 'eps', 1e-5)),
                                list(getattr(module, 'axes_dim', ())),
                                float(getattr(module, 'theta', 1000)), input_scale, binding.optimize_io)
        self.keys.append(key)
        binding.native_key = key
        from .joint_projection import metadata, library_path
        binding.joint_info = metadata(binding)
        binding.joint_available = binding.joint_kernels and bool(library_path())

    def lookahead(self):
        adapters = self.bindings.adapters
        blocks = list(self.bindings.owner.blocks)
        def key(module):
            return adapters[module].native_key
        def attention_keys(block):
            a = block.attn
            return [key(m) for m in (a.wq, a.wk, a.wv, a.gate)]
        for i, block in enumerate(blocks):
            self.module.set_lookahead(key(block.prenorm), attention_keys(block))
            self.module.set_lookahead(key(block.attn.wo), [key(block.mlp.gate), key(block.mlp.up)])
            if i + 1 < len(blocks):
                self.module.set_lookahead(key(block.mlp.down), attention_keys(blocks[i + 1]))
        if blocks:
            self.module.set_lookahead(key(self.bindings.owner.first), attention_keys(blocks[0]))

    @contextmanager
    def capture(self, on_library):
        from . import graph_ops
        from torch._inductor.codecache import CppWrapperCodeCache
        names = ('activation', 'modulation', 'residual', 'rope', 'timestep')
        saved = {name: getattr(graph_ops, name) for name in names}
        original = CppWrapperCodeCache.load_pybinding
        existed = 'load_pybinding' in CppWrapperCodeCache.__dict__
        descriptor = CppWrapperCodeCache.__dict__.get('load_pybinding')
        def export_entry(*args, **kwargs):
            code = kwargs.get('main_code', args[1] if len(args) > 1 else '')
            if 'k2_native' not in code:
                return original(*args, **kwargs)
            suffix = '''\n#ifdef _WIN32\nextern "C" __declspec(dllexport)\n#else\nextern "C" __attribute__((visibility("default")))\n#endif\nvoid k2_compiled_entry(AtenTensorHandle* inputs, AtenTensorHandle* outputs) { inductor_entry_impl(inputs, outputs); }\n'''
            if 'main_code' in kwargs:
                kwargs['main_code'] += suffix
            else:
                args = list(args)
                args[1] += suffix
            result = original(*args, **kwargs)
            extension = sys.modules[result.__module__]
            on_library(extension.__file__, kwargs.get('num_outputs', 1))
            return result
        try:
            for name in names:
                setattr(graph_ops, name, getattr(torch.ops.k2_native, name).default)
            CppWrapperCodeCache.load_pybinding = staticmethod(export_entry)
            yield
        finally:
            for name, fn in saved.items():
                setattr(graph_ops, name, fn)
            if existed:
                CppWrapperCodeCache.load_pybinding = descriptor
            else:
                delattr(CppWrapperCodeCache, 'load_pybinding')

    def make_runner(self, library, flat, args, kwargs, outputs):
        if library is None or flat is None:
            raise RuntimeError('C++ wrapper did not expose the complete compiled entry')
        tensors, paths, guards, guard_paths, containers, lengths = [], [], [], [], [], []
        def visit(value, path):
            if isinstance(value, torch.Tensor):
                tensors.append(value)
                paths.append(path)
            elif isinstance(value, (tuple, list)):
                containers.append(path)
                lengths.append(len(value))
                for i, item in enumerate(value):
                    visit(item, path + ['i' + str(i)])
            elif isinstance(value, dict):
                if not all(isinstance(key, str) for key in value):
                    raise RuntimeError('Native input dictionaries require string keys')
                containers.append(path)
                lengths.append(len(value))
                for key, item in value.items():
                    visit(item, path + ['d' + key])
            else:
                guard_paths.append(path)
                guards.append(value)
        visit(args, ['A'])
        visit(kwargs, ['K'])
        identities = {id(t): i for i, t in enumerate(tensors)}
        constants, sources = [], []
        for value in flat:
            if isinstance(value, torch.Tensor):
                source = identities.get(id(value), -1)
                if source < 0:
                    for i, tensor in enumerate(tensors):
                        if (value.device == tensor.device and value.shape == tensor.shape
                                and value.stride() == tensor.stride()
                                and torch._C._is_alias_of(value, tensor)):
                            source = i
                            break
                constants.append(value)
                sources.append(source)
            else:
                constants.append(torch.tensor(value, device='cpu'))
                sources.append(-1)
        if not any(i >= 0 for i in sources):
            raise RuntimeError('Compiled input mapping did not retain dynamic model inputs')
        return self.module.NativeGraph(library, constants, sources, tensors, paths,
                                       guard_paths, guards, containers, lengths, outputs)

    def configuration(self):
        values = []
        for module in self.bindings.owner.modules():
            weight = getattr(module, 'weight', None)
            scale = getattr(module, 'scale', None)
            parameters = (weight, getattr(module, 'bias', None), scale,
                          getattr(module, 'input_scale', None),
                          getattr(module, 'pre_quant_scale', None), getattr(module, 'lin', None))
            hook = getattr(module.forward, '__self__', None)
            adapter = getattr(hook, 'adapter', None)
            factors = getattr(adapter, 'weights', ())
            versions = []
            for tensor in parameters:
                try:
                    versions.append(tensor._version if isinstance(tensor, torch.Tensor) else None)
                except RuntimeError:
                    versions.append(None)
            values.append((id(module), module.forward, tuple(id(p) for p in parameters), tuple(versions),
                           tuple(module._forward_hooks.items()), tuple(module._forward_pre_hooks.items()),
                           tuple(id(t) for t in factors), getattr(adapter, 'multiplier', None),
                           id(getattr(module, 'pre_quant_scale', None)),
                           bool(getattr(module, '_full_precision_mm', False)),
                           bool(getattr(module, 'comfy_force_cast_weights', False)),
                           tuple(getattr(module, 'weight_function', ())),
                           getattr(module, 'weight_lowvram_function', None),
                           getattr(module, 'bias_lowvram_function', None),
                           tuple(getattr(module, 'bias_function', ())),
                           getattr(module, 'heads', None), getattr(module, 'kvheads', None),
                           getattr(module, 'eps', None), getattr(module, 'theta', None),
                           tuple(getattr(module, 'axes_dim', ()))))
        return tuple(values)

    def save_configuration(self):
        self.saved_configuration = self.configuration()

    def configuration_changed(self):
        return self.configuration() != self.saved_configuration

    def close(self):
        if getattr(self, 'closed', True):
            return
        self.closed = True
        self.module.release_bindings(self.keys)
        from .graph_ops import ManagedCall
        for binding in self.bindings.adapters.values():
            if isinstance(binding, ManagedCall):
                binding.native_key = None
                binding.native_loras = ()
                binding.native_pre_scale = None
                binding.native_trace_hooks = False
        self.keepalive.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
