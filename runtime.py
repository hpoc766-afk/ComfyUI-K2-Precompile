# SPDX-License-Identifier: GPL-3.0-only
import functools

import torch
import torch.nn.functional as F

from comfy.ldm.krea2.model import SingleStreamDiT


_STATE = "k2_precompile_prefill"


def fill_tokens(x, freqs, capacity):
    valid = x.shape[1]
    if valid > capacity:
        raise ValueError(
            f"K2 需要 {valid} 个 token，超过预填充容量 {capacity}。"
            "请增大 prefill_tokens，或设为 0 使用实际长度。"
        )
    if valid == capacity:
        return x, freqs, None
    x = F.pad(x, (0, 0, 0, capacity - valid))
    shape = list(freqs.shape)
    shape[2] = capacity
    padded_freqs = freqs.new_zeros(shape)
    padded_freqs[..., 0, 0] = 1
    padded_freqs[..., 1, 1] = 1
    padded_freqs[:, :, :valid] = freqs
    mask = torch.zeros((x.shape[0], 1, capacity), device=x.device, dtype=torch.bool)
    mask[:, :, :valid] = True
    return x, padded_freqs, mask


def block_forward(original, index, count, capacity):
    @functools.wraps(original)
    def forward(x, vec, freqs, mask=None, timestep_zero_index=None, transformer_options=None):
        options = transformer_options if transformer_options is not None else {}
        if capacity:
            if index == 0:
                if mask is not None:
                    raise ValueError("K2 token 预填充需要原生的无 mask 联合输入。")
                valid = x.shape[1]
                x, freqs, mask = fill_tokens(x, freqs, capacity)
                options[_STATE] = (valid, freqs, mask)
            valid, freqs, mask = options[_STATE]
        call_options = {key: value for key, value in options.items() if key != _STATE}
        result = original(
            x, vec, freqs, mask,
            timestep_zero_index=timestep_zero_index,
            transformer_options=call_options,
        )
        if capacity and index == count - 1:
            # The output projection must retain its original GEMM row count.
            result = result[:, :valid].contiguous()
            options.pop(_STATE)
        return result
    return forward


def _add(a, b):
    return a + b


def _multiply(a, b):
    return a * b


def compiled_block(block, original, add, multiply):
    def forward(x, vec, freqs, mask=None, timestep_zero_index=None, transformer_options=None):
        if timestep_zero_index is not None:
            return original(
                x, vec, freqs, mask, timestep_zero_index=timestep_zero_index,
                transformer_options=transformer_options,
            )
        prescale, preshift, pregate, postscale, postshift, postgate = block.mod(vec)
        pre = add(multiply(add(1, prescale), block.prenorm(x)), preshift)
        attention = block.attn(pre, freqs, mask, transformer_options=transformer_options)
        x = add(x, multiply(pregate, attention))
        post = add(multiply(add(1, postscale), block.postnorm(x)), postshift)
        return add(x, multiply(postgate, block.mlp(post)))
    return forward


def patch_krea2(model, prefill_tokens, compile_blocks=True):
    if prefill_tokens < 0:
        raise ValueError("prefill_tokens must be non-negative")
    diffusion = model.get_model_object("diffusion_model")
    if not isinstance(diffusion, SingleStreamDiT):
        raise ValueError("K2 precompile requires a Krea2 SingleStreamDiT model")
    patched = model.clone()
    if compile_blocks:
        # Each eager low-precision result is materialized before the next op.
        # Fusing the whole modulation expression changes BF16 INT4 inputs.
        settings = {"backend": "inductor", "fullgraph": True, "dynamic": False,
                    "options": {"triton.cudagraphs": False}}
        add = torch.compile(_add, **settings)
        multiply = torch.compile(_multiply, **settings)
    for index, block in enumerate(diffusion.blocks):
        key = f"diffusion_model.blocks.{index}"
        original = model.get_model_object(f"{key}.forward")
        if compile_blocks:
            original = compiled_block(block, original, add, multiply)
        patched.add_object_patch(
            f"{key}.forward",
            block_forward(original, index, len(diffusion.blocks), prefill_tokens),
        )
    return patched
