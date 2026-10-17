# SPDX-License-Identifier: GPL-3.0-only
import functools
import logging

import torch

import comfy.lora
import comfy.lora_convert
import comfy.sd
from comfy.ldm.krea2.model import SingleStreamDiT
from comfy.patcher_extension import PatcherInjection
from comfy.weight_adapter import LoRAAdapter
from comfy.weight_adapter.bypass import BypassForwardHook


_ENABLED = "k2_hot_lora_enabled"
_HOOKS = "k2_hot_lora_hooks"
_INJECTION = "k2_hot_lora"


class HotLoRAGroup:
    def __init__(self, hooks):
        self.hooks = tuple(hooks)

    def inject(self, patcher):
        active = []
        try:
            for hook in self.hooks:
                hook.inject()
                active.append(hook)
        except Exception:
            for hook in reversed(active):
                self.release(hook)
            raise

    @staticmethod
    def release(hook):
        hook.eject()
        # Cached workflow clones keep their adapters, but idle LoRAs need no VRAM.
        hook.adapter.weights = tuple(
            value.to(device="cpu") if isinstance(value, torch.Tensor) else value
            for value in hook.adapter.weights
        )

    def eject(self, patcher):
        for hook in reversed(self.hooks):
            self.release(hook)


def linear_target(model, key, adapter):
    if not isinstance(key, str) or not key.startswith("diffusion_model.") or not key.endswith(".weight"):
        return None
    if not isinstance(adapter, LoRAAdapter):
        return None
    up, down, alpha, mid, dora, reshape = adapter.weights
    if mid is not None or dora is not None or reshape is not None:
        return None
    try:
        target = model.model.get_submodule(key[:-len(".weight")])
    except AttributeError as error:
        raise ValueError(f"K2 LoRA key 不存在：{key}") from error
    if not hasattr(target, "in_features") or not hasattr(target, "out_features"):
        return None
    # Logical features remain valid even when INT4 storage packs the weight axes.
    if (up.ndim != 2 or down.ndim != 2 or down.shape[0] <= 0
            or up.shape[1] != down.shape[0]
            or down.shape[1] != target.in_features
            or up.shape[0] != target.out_features):
        raise ValueError(
            f"K2 LoRA A/B 维度不匹配：{key}；up={tuple(up.shape)}，"
            f"down={tuple(down.shape)}，层={target.in_features}→{target.out_features}"
        )
    return target


def load_hot_lora(model, clip, lora, strength_model, strength_clip, lora_metadata, original):
    if not isinstance(model.get_model_object("diffusion_model"), SingleStreamDiT):
        raise ValueError("LoRA 热拔插只适用于 Krea2 模型。")
    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    model_targets = set(key_map.values())
    if clip is not None:
        key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)
    converted = comfy.lora_convert.convert_lora(lora)
    try:
        loaded = comfy.lora.load_lora(converted, key_map)
    except KeyError as error:
        raise ValueError(f"K2 LoRA 缺少配对权重：{error}") from error
    if not loaded:
        raise ValueError("LoRA 没有匹配当前 K2 模型或文本编码器的 key。")

    hooks = []
    regular = {}
    accepted = set()
    ranks = set()
    for key, adapter in loaded.items():
        target = linear_target(model, key, adapter)
        if target is None:
            regular[key] = adapter
        else:
            accepted.add(key)
            ranks.add(adapter.weights[1].shape[0])
            if strength_model != 0:
                hooks.append(BypassForwardHook(target, adapter, multiplier=strength_model))

    if not accepted:
        return original(model, clip, lora, strength_model, strength_clip, lora_metadata=lora_metadata)

    result = model.clone()
    # Do not touch patches_uuid for a pure bypass LoRA: base weights stay shared.
    model_keys = set(accepted)
    regular_model = {key: value for key, value in regular.items() if key in model_targets}
    if regular_model and strength_model != 0:
        model_keys.update(result.add_patches(regular_model, strength_model))
    if hooks:
        group = HotLoRAGroup(tuple(model.get_attachment(_HOOKS) or ()) + tuple(hooks))
        result.set_attachments(_HOOKS, group.hooks)
        result.set_injections(_INJECTION, [PatcherInjection(group.inject, group.eject)])
    if lora_metadata:
        result.set_attachments("lora_metadata", lora_metadata)

    clip_keys = set()
    new_clip = None
    if clip is not None:
        new_clip = clip.clone()
        clip_keys.update(new_clip.add_patches(loaded, strength_clip))
        if lora_metadata:
            new_clip.patcher.set_attachments("lora_metadata", lora_metadata)
    for key in loaded:
        if key not in model_keys and key not in clip_keys:
            logging.warning("NOT LOADED %s", key)
    logging.info("K2 hot LoRA: %d linear hooks, ranks=%s", len(hooks), sorted(ranks))
    return result, new_clip


def install_lora_hook():
    original = comfy.sd.load_lora_for_models
    if getattr(original, "_k2_hot_lora_hook", False):
        return

    @functools.wraps(original)
    def load(model, clip, lora, strength_model, strength_clip, lora_metadata=None):
        if model is None or not model.get_attachment(_ENABLED):
            return original(model, clip, lora, strength_model, strength_clip, lora_metadata=lora_metadata)
        return load_hot_lora(model, clip, lora, strength_model, strength_clip, lora_metadata, original)

    load._k2_hot_lora_hook = True
    comfy.sd.load_lora_for_models = load


def enable_hot_lora(model):
    if not isinstance(model.get_model_object("diffusion_model"), SingleStreamDiT):
        raise ValueError("LoRA 热拔插只适用于 Krea2 模型。")
    install_lora_hook()
    result = model.clone()
    result.set_attachments(_ENABLED, True)
    return result
