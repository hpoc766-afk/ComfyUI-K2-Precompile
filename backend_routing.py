# SPDX-License-Identifier: GPL-3.0-only
"""Bind Kohaku-selected native routes from FX tensor metadata before Inductor."""
import hashlib
import json
import math
import os
from pathlib import Path

import torch

from .KohakuKernelRouter import Router, pow2

ROOT = Path(__file__).resolve().parent
OPERATORS = {'joint_qkv': 0, 'joint_swiglu': 1, 'residual_norm': 2,
             'qkv_prepared': 0, 'swiglu_prepared': 1}
CONFIGS = [{'path': 'split'}, {'path': 'fused'}]


def make_router(device):
    props = torch.cuda.get_device_properties(device)
    fingerprint = hashlib.sha256()
    for name in ('native_runtime.cpp', 'kernels/joint_projection.cu', 'kernels/native_dispatch.cu',
                 'kernels/qk_norm_rope.cu', 'kernels/int4_epilogue.cu',
                 'KohakuKernelRouter/tuner.py', 'KohakuKernelRouter/attention.py',
                 'kernels/norm_quant.cu', 'norm_quant.py', 'input_quant_ops.py',
                 'fa_routing.py', 'backend_routing.py'):
        fingerprint.update((ROOT / name).read_bytes())
    for name in ('.kernel-build/native_dispatch.json', '.kernel-build/norm_quant.json'):
        path = ROOT / name
        if path.is_file():
            fingerprint.update(path.read_bytes())
    fingerprint.update(str((props.name, props.major, props.minor, props.multi_processor_count,
                            props.total_memory, torch.__version__, torch.version.cuda)).encode())
    cache = Path(os.environ.get('K2_KERNEL_CACHE', str(Path.home() / '.cache' / 'comfyui-k2' / 'KohakuKernelRouter')))
    return Router(cache / fingerprint.hexdigest()[:20], device,
                  mode=os.environ.get('K2_KERNEL_TUNE', 'table'))


def _norm_quant_route(plan, node, scalar, by_key):
    from torch._subclasses.fake_tensor import unset_fake_temporarily
    args = node.args
    bare = str(node.target).split('.')[1] == 'norm_quant'
    source = args[0] if bare else args[2]
    example = source.meta.get('example_value', source.meta.get('val'))
    shape = tuple(int(d) for d in example.shape)
    norm_key, target_key, cut = ((scalar(args[3]), scalar(args[4]), scalar(args[5])) if bare
                                else (scalar(args[6]), scalar(args[7]), scalar(args[8])))
    rows, width = math.prod(shape[:-1]), shape[-1]
    group = by_key[target_key].joint_info
    problem = json.dumps([pow2(rows), shape[0], width, group, str(example.dtype), cut], separators=(',', ':'))
    identity = ('norm_quant', norm_key, rows, problem)
    if identity in plan.tuned_problems:
        return
    buffers = {}
    def launch(config):
        if not buffers:
            buffers['x'] = torch.full(shape, 0.125, device=plan.device, dtype=example.dtype)
            buffers['scale'] = torch.zeros((shape[0], 1, width), device=plan.device, dtype=example.dtype)
            buffers['shift'] = torch.zeros_like(buffers['scale'])
        plan.module.set_route(3, norm_key, rows, int(config['path'] == 'fused'))
        return lambda: torch.ops.k2_native.norm_quant(buffers['x'], buffers['scale'], buffers['shift'], norm_key, target_key, cut)
    with unset_fake_temporarily(), torch.no_grad():
        try:
            choice = plan.kernel_router.select('norm_quant', problem, CONFIGS, launch)
        except Exception:
            plan.module.set_route(3, norm_key, rows, 0)
            raise
        finally:
            plan.module.finish_tuning(plan.keys)
            buffers.clear()
        plan.module.set_route(3, norm_key, rows, int(choice['path'] == 'fused'))
    plan.tuned_problems.add(identity)


def tune_graph(plan, gm, inputs):
    """Only run in the compile backend; no selector is emitted into the FX graph."""
    from torch._subclasses.fake_tensor import unset_fake_temporarily
    if not hasattr(plan, 'kernel_router'):
        plan.kernel_router = make_router(plan.device)
    from .fa_routing import lower_attention
    plan.fa_node_count = lower_attention(gm, plan.kernel_router, plan.device)
    plan.tuning_decisions = list(plan.kernel_router.decisions)
    if not plan.bindings.fuse_epilogue or not plan.bindings.joint_kernels:
        return
    placeholders = [node for node in gm.graph.nodes if node.op == 'placeholder']
    concrete = dict(zip(placeholders, inputs))

    def scalar(value):
        if isinstance(value, torch.fx.Node):
            if value in concrete:
                value = concrete[value]
            elif value.op == 'get_attr':
                parent = gm
                for part in value.target.split('.'):
                    parent = getattr(parent, part)
                value = parent
            else:
                value = value.meta.get('example_value', value.meta.get('val'))
        return int(value)

    by_key = {b.native_key: b for b in plan.bindings.adapters.values()
              if getattr(b, 'native_key', None) is not None}
    for node in gm.graph.nodes:
        if node.op != 'call_function':
            continue
        name = str(node.target).split('.')
        if len(name) < 2 or name[0] != 'k2_native':
            continue
        if name[1] in ('norm_quant', 'gated_norm_quant'):
            _norm_quant_route(plan, node, scalar, by_key)
            continue
        if name[1] not in OPERATORS:
            continue
        kind = name[1]
        args = node.args
        if kind == 'qkv_prepared':
            args, kind = (args[0], args[3], args[4], args[5], args[6], args[7]), 'joint_qkv'
        elif kind == 'swiglu_prepared':
            args, kind = (args[0], args[3], args[4]), 'joint_swiglu'
        example = args[0].meta.get('example_value', args[0].meta.get('val'))
        if example is None:
            raise RuntimeError(f'Kohaku routing requires tensor metadata for {kind}')
        shape = tuple(int(dim) for dim in example.shape)
        rows, inner = math.prod(shape[:-1]), shape[-1]
        if kind == 'joint_qkv':
            keys, norms, heads, kvheads = [scalar(k) for k in args[2]], [scalar(k) for k in args[3]], scalar(args[4]), scalar(args[5])
            scalar_args = (norms, heads, kvheads)
            widths = [heads * 128, kvheads * 128, kvheads * 128, heads * 128]
            key = keys[0]
        elif kind == 'joint_swiglu':
            keys, widths = [scalar(k) for k in args[1]], [scalar(args[2]), scalar(args[2])]
            scalar_args = ()
            key = keys[0]
        else:
            key, norm_key = scalar(args[5]), scalar(args[6])
            scalar_args = (norm_key,)
            keys, widths = [key], [by_key[key].width]
        groups = [by_key[k].joint_info for k in keys]
        problem = json.dumps([pow2(rows), shape[0], inner, widths, groups, str(example.dtype)], separators=(',', ':'))
        identity = (kind, key, rows, problem)
        if identity in plan.tuned_problems:
            continue
        buffers = {}

        # Copy scalar metadata into closure defaults: timings may run several
        # candidates, while cached shapes never materialize synthetic inputs.
        def launch(config, kind=kind, shape=shape, dtype=example.dtype, keys=keys,
                   widths=widths, key=key, rows=rows, scalar_args=scalar_args, buffers=buffers):
            if not buffers:
                x = torch.full(shape, 0.125, device=plan.device, dtype=dtype)
                if kind == 'joint_qkv':
                    length = shape[1]
                    freqs = torch.zeros((1, 1, length, 64, 2, 2), device=plan.device, dtype=torch.float32)
                    freqs[..., 0, 0] = 1
                    freqs[..., 1, 1] = 1
                    buffers['call'] = lambda: torch.ops.k2_native.joint_qkv(
                        x, freqs, keys, scalar_args[0], scalar_args[1], scalar_args[2])
                elif kind == 'joint_swiglu':
                    buffers['call'] = lambda: torch.ops.k2_native.joint_swiglu(x, keys, widths[0])
                else:
                    width = widths[0]
                    residual = torch.full((*shape[:-1], width), 0.125, device=plan.device, dtype=dtype)
                    gate = torch.full((1, 1, width), 0.5, device=plan.device, dtype=dtype)
                    scale = torch.zeros_like(gate)
                    shift = torch.zeros_like(gate)
                    buffers['call'] = lambda: torch.ops.k2_native.residual_norm(
                        x, gate, residual, scale, shift, key, scalar_args[0])
            plan.module.set_route(OPERATORS[kind], key, rows, int(config['path'] == 'fused'))
            return buffers['call']

        with unset_fake_temporarily(), torch.no_grad():
            try:
                choice = plan.kernel_router.select(kind, problem, CONFIGS, launch)
            except Exception:
                plan.module.set_route(OPERATORS[kind], key, rows, 0)
                raise
            finally:
                plan.module.finish_tuning(plan.keys)
                buffers.clear()
            plan.module.set_route(OPERATORS[kind], key, rows, int(choice['path'] == 'fused'))
        plan.tuned_problems.add(identity)
    plan.module.reset_stats(plan.keys)
    plan.tuning_decisions = list(plan.kernel_router.decisions)
