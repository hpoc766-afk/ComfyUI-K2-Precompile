# SPDX-License-Identifier: GPL-3.0-only
"""Choose Kohaku FA tiles before lowering the opaque frontend to Triton launch."""
import logging

import torch

from .KohakuKernelRouter import pow2
from .KohakuKernelRouter.attention import SAFE, candidates, launch

log = logging.getLogger(__name__)


def _example(node):
    return node.meta.get('example_value', node.meta.get('val'))


def lower_attention(gm, router=None, device=None):
    from torch._subclasses.fake_tensor import unset_fake_temporarily
    count = 0
    for node in gm.graph.nodes:
        if node.op != 'call_function' or str(node.target) not in (
                'k2_fullgraph.kohaku_fa', 'k2_fullgraph.kohaku_fa.default'):
            continue
        args = node.args
        q, k, v = [_example(value) for value in args[:3]]
        shape, kshape, vshape = [tuple(int(d) for d in value.shape) for value in (q, k, v)]
        has_gate, sigmoid = args[4] is not None, bool(args[5])
        config = SAFE
        if router is not None and args[3] is None:
            problem = (f'{q.dtype}-B{shape[0]}-H{shape[1]}-KV{kshape[1]}-D{shape[3]}'
                       f'-Q{pow2(shape[2],64)}-K{pow2(kshape[2],64)}'
                       f'-G{2 if has_gate and sigmoid else 1 if has_gate else 0}')
            buffers = {}

            def prepare(cfg, shape=shape, kshape=kshape, vshape=vshape, dtype=q.dtype,
                        has_gate=has_gate, sigmoid=sigmoid, buffers=buffers):
                if not buffers:
                    def packed(s):
                        return torch.full((s[0], s[2], s[1], s[3]), 0.125,
                                          device=device, dtype=dtype).transpose(1, 2)
                    buffers['q'], buffers['k'], buffers['v'] = packed(shape), packed(kshape), packed(vshape)
                    buffers['gate'] = (torch.full((shape[0], shape[2], shape[1] * shape[3]), 0.5,
                                                 device=device, dtype=dtype) if has_gate else None)
                return lambda: launch(buffers['q'], buffers['k'], buffers['v'], None,
                                      buffers['gate'], sigmoid, cfg)

            with unset_fake_temporarily(), torch.no_grad():
                try:
                    ranked = candidates(q, device)
                    config = router.select('kohaku_fa', problem, ranked or [SAFE], prepare)
                finally:
                    buffers.clear()
        node.target = torch.ops.k2_fullgraph.kohaku_fa_launch.default
        node.args = (*args, config['BM'], config['BN'], config['warps'], config['stages'])
        count += 1
    if count:
        gm.recompile()
        log.info('[Kohaku FA] %d tiled attention nodes; KV heads kept unexpanded', count)
    return count
