# SPDX-License-Identifier: GPL-3.0-only
"""Strict full-graph compilation at model computation entry points."""
import copy
import functools
import sys

import torch


class FullGraphError(RuntimeError):
    """A selected model entry could not execute as a strict compiled graph."""


class FullGraphCall:
    def __init__(self, original, label):
        self.original = original
        self.label = label
        self.compiled = None
        functools.update_wrapper(self, original)

    def __call__(self, *args, **kwargs):
        owner = getattr(self.original, "__self__", None)
        children = getattr(owner, "modules", None)
        if callable(children) and any(hasattr(child, "_v") for child in children()):
            raise FullGraphError(
                f"全部预编译：{self.label} 使用动态VBAR权重驻留，当前不支持安全整图执行。"
                "请关闭全部预编译，或改用静态权重驻留配置。"
            )
        try:
            if self.compiled is None:
                self.compiled = torch.compile(
                    self.original,
                    backend="inductor",
                    fullgraph=True,
                    dynamic=True,
                    options={"triton.cudagraphs": False, "emulate_precision_casts": True},
                )
            return self.compiled(*args, **kwargs)
        except Exception as error:
            management = sys.modules.get("comfy.model_management")
            interrupted = getattr(management, "InterruptProcessingException", ())
            if isinstance(error, interrupted):
                raise
            # Never run the eager function as a hidden fallback. In particular,
            # do not turn an unsupported full graph into partial compilation.
            raise FullGraphError(
                f"全部预编译：{self.label} 完整图执行失败。"
                "请关闭全部预编译，或使用支持整图的模型、算子与权重管理方式。"
                f"原因：{type(error).__name__}: {error}"
            ) from error


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


def patch_full_models(model, clip, vae):
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
    patched_model.add_object_patch("diffusion_model.forward", FullGraphCall(dit_forward, "DiT"))
    patched_clip.patcher.add_object_patch(f"{text_path}.forward", FullGraphCall(te_forward, "TE Transformer"))
    patched_vae.patcher.add_object_patch("encode", FullGraphCall(vae_model.encode, "VAE encode"))
    patched_vae.patcher.add_object_patch("decode", FullGraphCall(vae_model.decode, "VAE decode"))
    return patched_model, patched_clip, patched_vae
