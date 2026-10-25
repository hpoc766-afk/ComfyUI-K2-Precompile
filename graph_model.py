# SPDX-License-Identifier: GPL-3.0-only
"""Traceable Krea2 math for the single model-level compilation.

Adapted from ComfyUI comfy/ldm/krea2/model.py (GPL-3.0). There are no compiler
calls here. Module hooks, attention patches, GQA, reference gates and entry
wrappers retain their native order. Backend kernels remain operator nodes.
"""
import torch
from comfy.ldm.krea2 import model as krea
from . import graph_ops, input_quant_ops
from .KohakuKernelRouter.attention import attention as kohaku_fa


def mlp(module, x, residual=None, next_norm=None, input_quant=None, next_quant_target=None):
    hidden = (input_quant_ops.swiglu(module, x, input_quant) if input_quant is not None
              else graph_ops.joint_swiglu(module, x))
    if hidden is None:
        gate, up = graph_ops.linears(x, (module.gate, module.up))
        hidden = graph_ops.activation_product(module.down, gate, up, 'silu')
    if residual is None:
        return module.down(hidden)
    if next_norm is not None:
        if next_quant_target is not None:
            return input_quant_ops.gated_norm_quant(module.down, next_norm[0], next_quant_target,
                hidden, residual[0], residual[1], next_norm[1], next_norm[2])
        result = graph_ops.linear_gated_norm(module.down, next_norm[0], hidden,
                                              residual[0], residual[1], next_norm[1], next_norm[2])
        if result is not None:
            return result
    return graph_ops.linear_gated(module.down, hidden, residual[0], residual[1])


def attention(module, x, freqs=None, mask=None, transformer_options=None, residual=None,
              fuse_qk_inputs=True, next_norm=None, input_quant=None, next_quant_target=None):
    options = {} if transformer_options is None else transformer_options
    patches = options.get('patches', {})
    extra = options.copy()
    input_patches = patches.get('attn1_patch', ()) if 'block_index' in options else ()
    prepared = (input_quant_ops.attention_inputs(module, x, input_quant, freqs)
                if input_quant is not None else
                None if input_patches or not fuse_qk_inputs else graph_ops.joint_attention_inputs(module, x, freqs))
    gate_is_sigmoid = prepared is not None
    if prepared is None and not input_patches and fuse_qk_inputs:
        prepared = graph_ops.attention_inputs(module, x, freqs)
    if prepared is not None:
        q, k, v, gate = prepared
        v = krea.rearrange(v, 'B L (H D) -> B H L D', H=module.kvheads)
    else:
        q, k, v, gate = graph_ops.linears(x, (module.wq, module.wk, module.wv, module.gate))
        q = krea.rearrange(q, 'B L (H D) -> B H L D', H=module.heads)
        k = krea.rearrange(k, 'B L (H D) -> B H L D', H=module.kvheads)
        v = krea.rearrange(v, 'B L (H D) -> B H L D', H=module.kvheads)
        q, k = module.qknorm(q, k)
        for patch in input_patches:
            out = patch(q, k, v, pe=freqs, attn_mask=mask, extra_options=extra)
            q, k, v = out.get('q', q), out.get('k', k), out.get('v', v)
            freqs, mask = out.get('pe', freqs), out.get('attn_mask', mask)
        if freqs is not None:
            q, k = graph_ops.rope(q, k, freqs)
    output_patches = patches.get('attn1_output_patch', ()) if 'block_index' in options else ()
    if not options.get('optimized_attention_override') and not options.get('wrappers'):
        if not output_patches:
            projected = kohaku_fa(q, k, v, mask, gate, gate_is_sigmoid)
        else:
            out = kohaku_fa(q, k, v, mask, None, False)
            for patch in output_patches:
                out = patch(out, extra)
            projected = (gate * out if gate_is_sigmoid
                         else graph_ops.activation_product(module.wo, gate, out, 'sigmoid'))
    else:
        q, k, v = (krea.AttentionTensorContainer(t) for t in (q, k, v))
        out = krea.optimized_attention_masked(
            q, k, v, module.heads, mask=mask, skip_reshape=True,
            enable_gqa=module.kvheads != module.heads,
            preferred_attention=module.comfy_attention, transformer_options=options)
        for patch in output_patches:
            out = patch(out, extra)
        projected = (gate * out if gate_is_sigmoid
                     else graph_ops.activation_product(module.wo, gate, out, 'sigmoid'))
    if residual is None:
        return module.wo(projected)
    if next_norm is not None:
        if next_quant_target is not None:
            return input_quant_ops.gated_norm_quant(module.wo, next_norm[0], next_quant_target,
                projected, residual[0], residual[1], next_norm[1], next_norm[2])
        result = graph_ops.linear_gated_norm(module.wo, next_norm[0], projected,
                                              residual[0], residual[1], next_norm[1], next_norm[2])
        if result is not None:
            return result
    return graph_ops.linear_gated(module.wo, projected, residual[0], residual[1])


def block(module, x, vec, freqs, mask=None, timestep_zero_index=None, transformer_options=None,
          mod_values=None, pre_normalized=None, next_norm=None, use_input_quant=False,
          pre_quantized=None, next_quant_target=None):
    options = {} if transformer_options is None else transformer_options
    ps, ph, pg, ss, sh, sg = module.mod(vec) if mod_values is None else mod_values
    if timestep_zero_index is not None:
        batch, cut = x.shape[0], timestep_zero_index
        pre = graph_ops.norm_mod(module.prenorm, x, ps, ph, cut)
        attn = module.attn(pre, freqs, mask, transformer_options=options)
        x = torch.cat((graph_ops.residual(x[:, :cut], pg[:batch], attn[:, :cut]),
                       graph_ops.residual(x[:, cut:], pg[batch:], attn[:, cut:])), dim=1)
        post = graph_ops.norm_mod(module.postnorm, x, ss, sh, cut)
        out = module.mlp(post)
        return torch.cat((graph_ops.residual(x[:, :cut], sg[:batch], out[:, :cut]),
                          graph_ops.residual(x[:, cut:], sg[batch:], out[:, cut:])), dim=1)
    if use_input_quant:
        packed = (input_quant_ops.norm_quant(module.prenorm, module.attn.wq, x, ps, ph)
                  if pre_quantized is None else pre_quantized)
        out, qa, qs = module.attn(x, freqs, mask, transformer_options=options, input_quant=packed,
            residual=(pg, x), next_norm=(module.postnorm, ss, sh), next_quant_target=module.mlp.gate)
        return module.mlp(out, residual=(sg, out), input_quant=(qa, qs),
                          next_norm=next_norm, next_quant_target=next_quant_target)
    pre = graph_ops.norm_mod(module.prenorm, x, ps, ph) if pre_normalized is None else pre_normalized
    out, post = _call_attention_normalized(module.attn, pre, freqs, mask, options, pg, x,
                                           (module.postnorm, ss, sh))
    if next_norm is not None:
        return _call_mlp_normalized(module.mlp, post, sg, out, next_norm)
    return _call_mlp(module.mlp, post, sg, out)


def _plain_module(module, function):
    return (getattr(module.forward, 'func', None) is function
            and not module._forward_hooks and not module._forward_pre_hooks)


def _call_attention_normalized(module, x, freqs, mask, options, gate, residual, next_norm):
    if _plain_module(module, attention):
        result = module(x, freqs, mask, transformer_options=options,
                        residual=(gate, residual), next_norm=next_norm)
        if isinstance(result, tuple):
            return result
    else:
        result = residual + gate * module(x, freqs, mask, transformer_options=options)
    return result, graph_ops.norm_mod(next_norm[0], result, next_norm[1], next_norm[2])


def _call_mlp_normalized(module, x, gate, residual, next_norm):
    if _plain_module(module, mlp):
        result = module(x, residual=(gate, residual), next_norm=next_norm)
        return result if isinstance(result, tuple) else (result, None)
    return residual + gate * module(x), None


def _call_attention(module, x, freqs, mask, options, gate, residual):
    # Dynamo preserves partial.func, but not arbitrary attributes on a partial.
    if _plain_module(module, attention):
        return module(x, freqs, mask, transformer_options=options, residual=(gate, residual))
    return residual + gate * module(x, freqs, mask, transformer_options=options)


def _call_mlp(module, x, gate, residual):
    if _plain_module(module, mlp):
        return module(x, residual=(gate, residual))
    return residual + gate * module(x)


def last(module, x, tvec, normalized=None, mod_values=None):
    if normalized is not None:
        return module.linear(normalized)
    scale, shift = module.modulation(tvec) if mod_values is None else mod_values
    return module.linear(graph_ops.norm_mod(module.norm, x, scale, shift))


def _norm_chain_block(module):
    if not (_plain_module(module, block) and _plain_module(module.attn, attention)
            and _plain_module(module.mlp, mlp)):
        return False
    mod = module.mod.forward
    if (not isinstance(mod, graph_ops.ManagedCall) or not mod.fuse_epilogue
            or getattr(mod.original, '__func__', None) is not krea.DoubleSharedModulation.forward
            or module.mod._forward_hooks or module.mod._forward_pre_hooks
            or not module.attn.forward.keywords.get('fuse_qk_inputs', False)):
        return False
    projections = (module.attn.wq, module.attn.wk, module.attn.wv, module.attn.gate,
                   module.attn.wo, module.mlp.gate, module.mlp.up, module.mlp.down)
    if not all(isinstance(p.forward, graph_ops.ManagedCall) and p.forward.joint_available
               and p.forward.joint_info is not None for p in projections):
        return False
    norms = (module.prenorm, module.postnorm, module.attn.qknorm.qnorm, module.attn.qknorm.knorm)
    return all(isinstance(n.forward, graph_ops.ManagedCall) and n.forward.norm_info for n in norms)


def _norm_quant_block(module):
    qkv = (module.attn.wq, module.attn.wk, module.attn.wv, module.attn.gate)
    mlp = (module.mlp.gate, module.mlp.up)
    return (all(m.forward.joint_info == qkv[0].forward.joint_info for m in qkv)
            and all(m.forward.joint_info == mlp[0].forward.joint_info for m in mlp)
            and module.prenorm.forward.norm_width == module.attn.wq.in_features
            and module.postnorm.forward.norm_width == module.mlp.gate.in_features)


def _norm_chain_last(module):
    mod, norm = module.modulation.forward, module.norm.forward
    return (_plain_module(module, last) and isinstance(mod, graph_ops.ManagedCall)
            and getattr(mod.original, '__func__', None) is krea.SimpleModulation.forward
            and not module.modulation._forward_hooks and not module.modulation._forward_pre_hooks
            and isinstance(norm, graph_ops.ManagedCall) and norm.norm_info)


def dit(module, x, timesteps, context, attention_mask=None, ref_latents=None,
        transformer_options=None, allow_norm_chain=False, allow_last_norm=False,
        allow_input_quant=False, **kwargs):
    options = {} if transformer_options is None else transformer_options.copy()
    temporal = x.ndim == 5
    if temporal:
        b5, c5, t5, h5, w5 = x.shape
        x = x.reshape(b5 * t5, c5, h5, w5)
    bs, _, h_orig, w_orig = x.shape
    context = module._unpack_context(context)
    img, imgpos, h_, w_ = module.process_img(x)
    img_tokens = img.shape[1]
    timestep_zero_index = None
    ref_method = kwargs.get('ref_latents_method', module.default_ref_method)
    if ref_method is not None and ref_latents is not None and len(ref_latents) > 0:
        ref_tokens, ref_pos, ref_num_tokens = [], [], []
        for index, ref in enumerate(ref_latents, 1):
            if ref.ndim == 5:
                rb, rc, rt, rh5, rw5 = ref.shape
                ref = ref.reshape(rb * rt, rc, rh5, rw5)
            ref = krea.comfy.utils.repeat_to_batch_size(ref, bs)
            kontext, kontext_ids, _, _ = module.process_img(ref, index=index)
            ref_tokens.append(kontext)
            ref_pos.append(kontext_ids)
            ref_num_tokens.append(kontext.shape[1])
        img = torch.cat([img] + ref_tokens, dim=1)
        imgpos = torch.cat([imgpos] + ref_pos, dim=1)
        del ref_tokens, ref_pos
        if ref_method == 'index_timestep_zero':
            timestep_zero_index = img_tokens
        options['reference_image_num_tokens'] = ref_num_tokens
    img = module.first(img)
    t = module.tmlp(graph_ops.timestep(timesteps, module.tdim).unsqueeze(1).to(img.dtype))
    tvec = module.tproj(t)
    if timestep_zero_index is not None:
        t0 = module.tmlp(graph_ops.timestep(torch.zeros_like(timesteps), module.tdim).unsqueeze(1).to(img.dtype))
        tvec = torch.cat((tvec, module.tproj(t0)), dim=0)
    context = module.txtfusion(context, mask=None, transformer_options=options)
    context = module.txtmlp(context)
    txtlen = context.shape[1]
    txtpos = torch.zeros(bs, txtlen, 3, device=context.device, dtype=torch.float32)
    for patch in options.get('patches', {}).get('post_input', ()):
        out = patch({'img': img, 'txt': context, 'img_ids': imgpos,
                     'txt_ids': txtpos, 'transformer_options': options})
        img, context = out['img'], out['txt']
        imgpos, txtpos = out['img_ids'], out['txt_ids']
    combined = torch.cat((context, img), dim=1)
    del context, img
    if timestep_zero_index is not None:
        timestep_zero_index += txtlen
    pos = torch.cat((txtpos, imgpos), dim=1)
    del txtpos, imgpos
    freqs = module.pe_embedder(pos)
    del pos
    options['total_blocks'] = len(module.blocks)
    options['block_type'] = 'single'
    options['img_slice'] = [txtlen, combined.shape[1]]
    chain = (allow_norm_chain and combined.shape[0] == 1 and timestep_zero_index is None
             and not options.get('patches') and not options.get('patches_replace')
             and not options.get('wrappers') and not options.get('optimized_attention_override'))
    quant_chain = chain and allow_input_quant and combined.is_cuda and combined.dtype in (torch.bfloat16, torch.float16)
    carried, carried_quant, final_mod = None, None, None
    chain_last = chain and allow_last_norm
    mod_values = module.blocks[0].mod(tvec) if chain else None
    for i, current in enumerate(module.blocks):
        options['block_index'] = i
        if chain:
            # Time-only modulation is independent of the hidden state. Keep one
            # block ahead so down's epilogue can produce the next normalized input.
            following = module.blocks[i + 1] if i + 1 < len(module.blocks) else None
            next_mod = following.mod(tvec) if following is not None else None
            if following is not None:
                target_norm = (following.prenorm, next_mod[0], next_mod[1])
            elif chain_last:
                final_mod = module.last.modulation(t)
                target_norm = (module.last.norm, final_mod[0], final_mod[1])
            else:
                target_norm = None
            result = current(combined, tvec, freqs, None, transformer_options=options,
                             mod_values=mod_values, pre_normalized=carried, next_norm=target_norm,
                             use_input_quant=quant_chain, pre_quantized=carried_quant,
                             next_quant_target=following.attn.wq if quant_chain and following is not None else None)
            if quant_chain and following is not None:
                combined, qa, qs = result
                carried, carried_quant = None, (qa, qs)
            elif target_norm is None:
                combined, carried, carried_quant = result, None, None
            else:
                combined, carried = result
                carried_quant = None
            mod_values = next_mod
        else:
            combined = current(combined, tvec, freqs, None,
                               timestep_zero_index=timestep_zero_index, transformer_options=options)
    final = (module.last(combined, t, normalized=carried, mod_values=final_mod)
             if chain_last else module.last(combined, t))
    del combined
    out = final[:, txtlen:txtlen + img_tokens, :]
    out = krea.rearrange(out, 'b (h w) (c ph pw) -> b c (h ph) (w pw)',
                         h=h_, w=w_, ph=module.patch, pw=module.patch, c=module.channels)
    out = out[:, :, :h_orig, :w_orig]
    if temporal:
        out = out.reshape(b5, t5, module.channels, h_orig, w_orig).movedim(1, 2)
    return out
