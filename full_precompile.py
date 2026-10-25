# SPDX-License-Identifier: GPL-3.0-only
"""Strict full-graph compilation at model computation entry points."""
import copy
import functools
import logging
import sys
from contextlib import nullcontext

import torch


class FullGraphError(RuntimeError):
    """A selected model entry could not execute as a strict compiled graph."""


class FullGraphCall:
    def __init__(self, original, label, *, fuse_epilogue=False, native_schedule=True, joint_kernels=True,
                 optimize_io=True, fuse_input_quant=True):
        self.original = original
        self.label = label
        self.fuse_epilogue = bool(fuse_epilogue)
        self.native_schedule = bool(native_schedule)
        self.joint_kernels = bool(joint_kernels)
        self.optimize_io = bool(optimize_io)
        self.fuse_input_quant = bool(fuse_input_quant)
        self.native_plan = None
        self.native_runner = None
        self._native_flat_inputs = None
        self._native_library = None
        self._native_outputs = 1
        self._patch_epoch = None
        self.compiled = None
        self.bindings = None
        self.graph_count = 0
        self.graph_nodes = 0
        functools.update_wrapper(self, original)

    def _compile_krea_graph(self, gm, inputs):
        from torch._inductor import compile as compile_graph
        if self.native_plan is not None:
            from .backend_routing import tune_graph
            tune_graph(self.native_plan, gm, inputs)
        else:
            from .fa_routing import lower_attention
            lower_attention(gm)
        self.graph_count += 1
        self.graph_nodes = len(list(gm.graph.nodes))
        logging.info('[K2 full graph] captured complete DiT graph #%d: %d FX nodes',
                     self.graph_count, self.graph_nodes)
        result = compile_graph(gm, inputs, options={
            'cpp_wrapper': self.native_plan is not None,
            'triton.cudagraphs': False,
            'emulate_precision_casts': True,
            'max_fusion_size': 1,
            'epilogue_fusion': False,
            'pattern_matcher': False,
        })
        if self.native_plan is None:
            return result
        def record_inputs(*flat):
            self._native_flat_inputs = flat
            return result(*flat)
        return record_inputs

    def invalidate_native(self, *_args):
        self.native_runner = None
        self.compiled = None
        self._native_flat_inputs = None
        self._native_library = None
        if self.native_plan is not None:
            self.native_plan.close()
            self.native_plan = None

    def prepare_native(self, *args):
        epoch = getattr(args[0], 'patches_uuid', None) if args else None
        changed = self._patch_epoch is not None and epoch != self._patch_epoch
        self._patch_epoch = epoch
        if self.native_plan is not None and (changed or self.native_plan.configuration_changed()):
            self.invalidate_native()

    def _record_library(self, path, outputs):
        self._native_library = path
        self._native_outputs = outputs

    def __call__(self, *args, **kwargs):
        if self.native_runner is not None:
            try:
                return self.native_runner.run(*args, **kwargs)
            except RuntimeError as error:
                if 'K2_NATIVE_INPUT_SIGNATURE_CHANGED' not in str(error):
                    raise
                self.invalidate_native()
        owner = getattr(self.original, "__self__", None)
        children = getattr(owner, "modules", None)
        is_krea = (owner is not None and type(owner).__name__ == 'SingleStreamDiT'
                   and hasattr(owner, 'blocks') and hasattr(owner, 'txtfusion'))
        if is_krea:
            if self.bindings is None:
                from .graph_ops import ModelBindings
                self.bindings = ModelBindings(owner, fuse_epilogue=self.fuse_epilogue,
                                              joint_kernels=self.joint_kernels, optimize_io=self.optimize_io,
                                              fuse_input_quant=self.fuse_input_quant, native_execution=self.native_schedule)
        elif callable(children) and any(hasattr(child, '_v') for child in children()):
            raise FullGraphError(
                f'全部预编译：{self.label} 的动态权重读取尚未接入完整图。'
            )
        try:
            if is_krea and self.native_schedule:
                with self.bindings.active():
                    if self.native_plan is None:
                        from .native_schedule import NativePlan
                        x = args[0] if args else kwargs['x']
                        self.native_plan = NativePlan(self.bindings, x.device)
                    with self.native_plan.capture(self._record_library):
                        if self.compiled is None:
                            self.compiled = torch.compile(self.original, backend=self._compile_krea_graph,
                                                          fullgraph=True, dynamic=True)
                        output = self.compiled(*args, **kwargs)
                self.native_runner = self.native_plan.make_runner(
                    self._native_library, self._native_flat_inputs, args, kwargs, self._native_outputs)
                self._native_flat_inputs = None
                self.native_plan.save_configuration()
                logging.info('[K2 native schedule] C++ graph ready; hot forward uses no Python operator callbacks')
                return output
            if self.compiled is None:
                if is_krea:
                    self.compiled = torch.compile(
                        self.original, backend=self._compile_krea_graph,
                        fullgraph=True, dynamic=True,
                    )
                else:
                    self.compiled = torch.compile(
                        self.original, backend='inductor', fullgraph=True, dynamic=True,
                        options={'triton.cudagraphs': False, 'emulate_precision_casts': True},
                    )
            with self.bindings.active() if is_krea else nullcontext():
                return self.compiled(*args, **kwargs)
        except Exception as error:
            management = sys.modules.get("comfy.model_management")
            interrupted = getattr(management, "InterruptProcessingException", ())
            if isinstance(error, interrupted):
                raise
            # Never run the eager function as a hidden fallback. In particular,
            # do not turn an unsupported full graph into partial compilation.
            if self.native_plan is not None:
                self.invalidate_native()
            raise FullGraphError(
                f"全部预编译：{self.label} 完整图执行失败。"
                "请关闭全部预编译，或使用支持整图的模型、算子与权重管理方式。"
                f"原因：{type(error).__name__}: {error}"
            ) from error


def _native_lifecycle(patcher, call):
    if callable(getattr(patcher, 'add_callback', None)):
        from comfy.patcher_extension import CallbacksMP
        patcher.add_callback(CallbacksMP.ON_PRE_RUN, call.prepare_native)
        patcher.add_callback(CallbacksMP.ON_DETACH, call.invalidate_native)


def patch_krea2_fullgraph(model, *, fuse_epilogue=False):
    """One compilation entry for the entire DiT; no leaf/block compiler calls."""
    original = model.get_model_object('diffusion_model.forward')
    result = model.clone()
    call = FullGraphCall(original, 'DiT', fuse_epilogue=fuse_epilogue)
    result.add_object_patch('diffusion_model.forward', call)
    _native_lifecycle(result, call)
    return result


def text_transformer_path(text_model):
    name = getattr(text_model, "clip", None)
    paths = [f"{name}.transformer"] if isinstance(name, str) and name else []
    paths.append("transformer")
    for path in paths:
        current = text_model
        for part in path.split("."):
            current = getattr(current, part, None)
            if current is None:
                break
        if current is not None and callable(getattr(current, "forward", None)):
            return path
    raise ValueError("全部预编译：无法识别文本编码器的Transformer入口。")


def patch_full_models(model, clip, vae, *, fuse_epilogue=False):
    """Return managed clones; original wrappers and shared modules stay unpatched.

    ModelPatcher applies/restores object patches when the corresponding model is
    loaded/unloaded. Tokenization, scheduler, image IO and VSR are outside these
    model-level graphs. VAE encode/decode necessarily have separate inputs/graphs.
    """
    text_path = text_transformer_path(clip.cond_stage_model)
    dit_forward = model.get_model_object("diffusion_model.forward")
    te_forward = clip.patcher.get_model_object(f"{text_path}.forward")
    vae_model = vae.first_stage_model
    if vae.patcher.model is not vae_model:
        raise ValueError("全部预编译：VAE模型与其权重管理器不一致。")
    for label, function in [("DiT", dit_forward), ("TE", te_forward),
                            ("VAE encode", getattr(vae_model, "encode", None)),
                            ("VAE decode", getattr(vae_model, "decode", None))]:
        if not callable(function):
            raise ValueError(f"全部预编译：{label} 缺少可编译入口。")

    patched_model = model.clone()
    patched_clip = clip.clone()
    patched_vae = copy.copy(vae)
    patched_vae.patcher = vae.patcher.clone()
    call = FullGraphCall(dit_forward, "DiT", fuse_epilogue=fuse_epilogue)
    patched_model.add_object_patch("diffusion_model.forward", call)
    _native_lifecycle(patched_model, call)
    patched_clip.patcher.add_object_patch(f"{text_path}.forward", FullGraphCall(te_forward, "TE Transformer"))
    patched_vae.patcher.add_object_patch("encode", FullGraphCall(vae_model.encode, "VAE encode"))
    patched_vae.patcher.add_object_patch("decode", FullGraphCall(vae_model.decode, "VAE decode"))
    return patched_model, patched_clip, patched_vae
