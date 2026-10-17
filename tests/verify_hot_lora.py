# SPDX-License-Identifier: GPL-3.0-only
"""Exercise the real LoRA nodes, K2 forward, compile reuse and hook lifecycle."""
import gc
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from _paths import configure
paths=configure(Path(__file__).stem)
COMFY=paths.comfy
ROOT=paths.root
OUT=paths.out
FILES = OUT/'test-loras'
FILES.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(COMFY))
sys.argv = [sys.argv[0]]
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
import nodes, folder_paths, comfy.ops, comfy.sd
from comfy.ldm.krea2.model import SingleStreamDiT
from comfy.model_patcher import ModelPatcher
from torch._dynamo.utils import counters

spec = importlib.util.spec_from_file_location('k2_hot_lora_validation', ROOT/'__init__.py', submodule_search_locations=[str(ROOT)])
plugin = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plugin
spec.loader.exec_module(plugin)
hot = sys.modules[spec.name+'.hot_lora']
patch_krea2 = sys.modules[spec.name+'.runtime'].patch_krea2

torch.set_grad_enabled(False)
torch.manual_seed(428)
diffusion = SingleStreamDiT(features=64, tdim=16, txtdim=32, heads=4, kvheads=2,
    multiplier=2, layers=2, patch=2, channels=4, txtlayers=2,
    txtheads=4, txtkvheads=2, device='cuda', dtype=torch.float32,
    operations=comfy.ops.manual_cast)
for parameter in diffusion.parameters():parameter.uniform_(-0.1,0.1)
base = torch.nn.Module();base.diffusion_model = diffusion
base.model_config = SimpleNamespace(unet_config={})
model = ModelPatcher(base,torch.device('cuda'),torch.device('cpu'))
linear = diffusion.blocks[0].attn.wq
original = linear.forward
weight = linear.weight.clone()
key = 'diffusion_model.blocks.0.attn.wq'

def checkpoint(rank, name, prefix=key, inputs=64, outputs=64):
    values={prefix+'.lora_up.weight':torch.randn(outputs,rank)*0.02,
            prefix+'.lora_down.weight':torch.randn(rank,inputs)*0.02,
            prefix+'.alpha':torch.tensor(float(rank))}
    save_file(values,str(FILES/name),metadata={'hot_test':name})
    return values
raw_a=checkpoint(2,'k2_hot_A.safetensors')
raw_b=checkpoint(3,'k2_hot_B.safetensors')
checkpoint(2,'k2_hot_wrong_key.safetensors',prefix='wrong_model.layer')
checkpoint(2,'k2_hot_wrong_shape.safetensors',inputs=63)
missing={key+'.lora_up.weight':torch.ones(64,2)}
save_file(missing,str(FILES/'k2_hot_missing_pair.safetensors'))
folder_paths.add_model_folder_path('loras',str(FILES))
loader=nodes.LoraLoader()
plain,_=loader.load_lora(model,None,'k2_hot_A.safetensors',0.5,0)
assert plain.patches and not plain.get_injections(hot._INJECTION)
enabled=hot.enable_hot_lora(model)
installed=comfy.sd.load_lora_for_models
hot.install_lora_hook();assert comfy.sd.load_lora_for_models is installed
plain_after,_=loader.load_lora(model,None,'k2_hot_A.safetensors',0.5,0)
assert plain_after.patches and not plain_after.get_injections(hot._INJECTION)
assert not model.get_attachment(hot._ENABLED)
assert enabled.clone().get_attachment(hot._ENABLED)

A,_=loader.load_lora(enabled,None,'k2_hot_A.safetensors',0.5,0)
B=nodes.LoraLoaderModelOnly().load_lora_model_only(enabled,'k2_hot_B.safetensors',-0.75)[0]
A2,_=loader.load_lora(enabled,None,'k2_hot_A.safetensors',0.5,0)
stack,_=loader.load_lora(A,None,'k2_hot_B.safetensors',-0.75,0)
zero=nodes.LoraLoaderModelOnly().load_lora_model_only(enabled,'k2_hot_A.safetensors',0)[0]
assert zero is enabled
assert not A.patches and A.patches_uuid==enabled.patches_uuid
assert A.model is B.model is enabled.model
assert A.get_attachment('lora_metadata')=={'hot_test':'k2_hot_A.safetensors'}
x=torch.randn(1,9,64,device='cuda')
base_y=original(x)
def delta(raw,strength):
    down=raw[key+'.lora_down.weight'].cuda();up=raw[key+'.lora_up.weight'].cuda()
    return F.linear(F.linear(x,down),up)*strength
expected_a=base_y+delta(raw_a,0.5)
expected_b=base_y+delta(raw_b,-0.75)
expected_stack=expected_a+delta(raw_b,-0.75)

def projection(patcher):
    patcher.patch_model(load_weights=False)
    try:return linear(x).clone()
    finally:
        patcher.unpatch_model(unpatch_weights=False)
        assert linear.forward==original,'LoRA hook leaked after eject'
        for hook in patcher.get_attachment(hot._HOOKS) or ():
            assert all(not isinstance(t,torch.Tensor) or t.device.type=='cpu' for t in hook.adapter.weights)

results={'projection_errors':{},'invalid_rejections':[]}
for label,patcher,expected in [('A',A,expected_a),('B',B,expected_b),('A_return',A2,expected_a),('stack',stack,expected_stack),('zero',zero,base_y),('base_after',enabled,base_y)]:
    actual=projection(patcher)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    results['projection_errors'][label]=float((actual-expected).abs().max())
assert torch.equal(weight,linear.weight)
A.patch_model(load_weights=False)
try:
    live_stack,_=loader.load_lora(A,None,'k2_hot_B.safetensors',-0.75,0)
    torch.testing.assert_close(linear(x),expected_a,rtol=0,atol=0)
finally:
    A.unpatch_model(unpatch_weights=False)
torch.testing.assert_close(projection(live_stack),expected_stack,rtol=0,atol=0)
results['clone_while_injected_restores_and_stacks']=True
for name in ['k2_hot_wrong_key.safetensors','k2_hot_wrong_shape.safetensors','k2_hot_missing_pair.safetensors']:
    try:loader.load_lora(enabled,None,name,1.0,0)
    except ValueError:results['invalid_rejections'].append(name)
    else:raise AssertionError('Invalid K2 adapter accepted: '+name)
assert linear.forward==original

# Keep unsupported DoRA on the original weight-patch path.
dora=dict(raw_a);dora[key+'.dora_scale']=torch.ones(64)
dora_model,_=comfy.sd.load_lora_for_models(enabled,None,dora,0.5,0)
assert dora_model.patches and not dora_model.get_injections(hot._INJECTION)
results['dora_native_fallback']=True

# Real CLIP clone/add_patches API on a small encoder, with independent strength.
clip=comfy.sd.CLIP(no_init=True)
clip.cond_stage_model=torch.nn.Module();clip.cond_stage_model.encoder=torch.nn.Linear(8,8)
clip.patcher=ModelPatcher(clip.cond_stage_model,torch.device('cuda'),torch.device('cpu'))
clip.tokenizer=None;clip.layer_idx=None;clip.tokenizer_options={};clip.use_clip_schedule=False;clip.apply_hooks_to_conds=None
mixed=dict(raw_a)
mixed.update({'text_encoders.encoder.lora_up.weight':torch.randn(8,2),
              'text_encoders.encoder.lora_down.weight':torch.randn(2,8)})
mixed_model,mixed_clip=comfy.sd.load_lora_for_models(enabled,clip,mixed,0.5,0.25,lora_metadata={'tag':'mixed'})
assert not mixed_model.patches and mixed_model.patches_uuid==enabled.patches_uuid
assert mixed_clip.patcher.patches['encoder.weight'][0][0]==0.25
assert mixed_clip.patcher.get_attachment('lora_metadata')=={'tag':'mixed'}
results['clip_strength_and_metadata_preserved']=True

# Changing adapters after compile must not compile the K2 arithmetic again.
compiled=hot.enable_hot_lora(patch_krea2(model,48,compile_blocks=True))
CA,_=loader.load_lora(compiled,None,'k2_hot_A.safetensors',0.5,0)
CB,_=loader.load_lora(compiled,None,'k2_hot_B.safetensors',-0.75,0)
padded=patch_krea2(model,48,compile_blocks=False)
RA,_=comfy.sd.load_bypass_lora_for_models(padded,None,raw_a,0.5,0)
RB,_=comfy.sd.load_bypass_lora_for_models(padded,None,raw_b,-0.75,0)
image=torch.randn(1,4,8,12,device='cuda');context=torch.randn(1,3,64,device='cuda')
time=torch.tensor([0.5],device='cuda')
def predict(patcher):
    patcher.patch_model(load_weights=False)
    try:return diffusion(image,time,context).clone()
    finally:patcher.unpatch_model(unpatch_weights=False)
ref_a=predict(RA);ref_b=predict(RB)
out_a=predict(CA);warm_graphs=counters['stats']['unique_graphs']
out_b=predict(CB);out_a2=predict(CA)
assert counters['stats']['unique_graphs']==warm_graphs,'LoRA switch recompiled arithmetic'
torch.testing.assert_close(out_a,ref_a,rtol=0,atol=0)
torch.testing.assert_close(out_b,ref_b,rtol=0,atol=0)
torch.testing.assert_close(out_a2,out_a,rtol=0,atol=0)
assert not torch.equal(out_a,out_b)
assert linear.forward==original
results.update(compiled_reference_max_abs=0.0,swap_added_graphs=0,base_weights_unchanged=bool(torch.equal(weight,linear.weight)),passed=True)
(OUT/'tiny-hot-lora.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print(json.dumps(results,indent=2),flush=True)
del model,enabled,plain,plain_after,A,B,A2,stack,live_stack,zero,dora_model,mixed_model,mixed_clip,clip,compiled,CA,CB,padded,RA,RB,patcher
gc.collect()
