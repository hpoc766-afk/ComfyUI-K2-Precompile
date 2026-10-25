# SPDX-License-Identifier: GPL-3.0-only
"""Backward-compatible loader entry for complete Krea2 forward compilation."""
from .full_precompile import patch_krea2_fullgraph


def patch_krea2(model, prefill_tokens=0, compile_blocks=True, *, fuse_epilogue=False):
    # Retain old positional inputs for saved workflows. Compilation always uses
    # the entire DiT and actual token length; no padding or per-block compilation.
    if prefill_tokens < 0:
        raise ValueError('prefill_tokens must be non-negative')
    return (patch_krea2_fullgraph(model, fuse_epilogue=fuse_epilogue)
            if compile_blocks else model.clone())
