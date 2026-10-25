# SPDX-License-Identifier: GPL-3.0-only
"""Model-scoped DiT activation dtype, independent of weight storage formats."""
import logging
import uuid

ACTIVATION_DTYPES = ("auto", "fp16", "bf16")


def patch_activation_dtype(model, choice="auto"):
    if choice == "auto":
        return model
    if choice not in ACTIVATION_DTYPES:
        raise ValueError(f"Unknown K2 activation dtype: {choice}")
    import torch
    from comfy import ops
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[choice]
    result = model.clone()
    # set_model_compute_dtype also sets force_cast_weights, disabling quantized
    # matmuls. Patch the activation contract without that storage/compute flag.
    result.add_object_patch("manual_cast_dtype", dtype)
    result.add_object_patch("diffusion_model.dtype", dtype)
    owner = model.get_model_object("diffusion_model")
    for name, module in owner.named_modules():
        # Dense Comfy linears must cast at use when their stored weight dtype
        # differs from the input. MixedPrecisionOps already obeys input dtype.
        if isinstance(module, ops.disable_weight_init.Linear):
            result.add_object_patch(f"diffusion_model.{name}.comfy_cast_weights", True)
    result.patches_uuid = uuid.uuid4()
    logging.info("[K2] DiT activations: %s; weight storage and quantized matmuls retained", dtype)
    return result
