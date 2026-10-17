# SPDX-License-Identifier: GPL-3.0-only
"""Validate installed loader with local K2 INT4/CLIP/VAE and one real block."""
import gc
import importlib.util
import json
from pathlib import Path
import sys
import time

from _paths import configure
paths=configure(Path(__file__).stem)
COMFY=paths.comfy
ROOT=paths.root
OUT=paths.out
sys.path.insert(0, str(COMFY))
sys.argv = [sys.argv[0]]
import torch
import comfy.model_management as mm
from torch._dynamo.utils import counters

spec = importlib.util.spec_from_file_location('k2_real_loader_validation', ROOT/'__init__.py', submodule_search_locations=[str(ROOT)])
plugin = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plugin
spec.loader.exec_module(plugin)
fill_tokens = sys.modules[spec.name+'.runtime'].fill_tokens

torch.set_grad_enabled(False)
started = time.perf_counter()
model,clip,vae = plugin.K2PrecompileLoader().load(
    paths.args.dit_name,
    paths.args.text_encoder_name,
    paths.args.vae_name, True, 64)
diffusion = model.get_model_object('diffusion_model')
original = diffusion.blocks[0].forward
results = {'model_class':type(diffusion).__name__,'clip_class':type(clip).__name__,
    'vae_class':type(vae).__name__,'block_count':len(diffusion.blocks),
    'object_patch_count':len(model.object_patches),'load_seconds':time.perf_counter()-started}
print('REAL_LOADER_OK',json.dumps(results),flush=True)
mm.load_models_gpu([model],memory_required=512*1024*1024)
print('REAL_MODEL_MANAGED_LOAD_OK',flush=True)
torch.manual_seed(314)
device = mm.get_torch_device()
dtype = diffusion.dtype
features = diffusion.blocks[0].attn.heads * diffusion.blocks[0].attn.headdim
x = torch.randn(1,32,features,device=device,dtype=dtype)*0.1
vec = torch.randn(1,1,features*6,device=device,dtype=dtype)*0.01
rope = torch.zeros(1,1,32,diffusion.blocks[0].attn.headdim//2,2,2,device=device)
rope[...,0,0] = 1;rope[...,1,1] = 1
xp,rp,mask = fill_tokens(x,rope,64)
options = {'block_index':0,'block_type':'single','total_blocks':len(diffusion.blocks),'img_slice':[0,32]}
for _ in range(2):
    warm = original(xp,vec,rp,mask,transformer_options=options)
    torch.cuda.synchronize()
    del warm
reference = original(xp,vec,rp,mask,transformer_options=options).clone()
torch.cuda.synchronize()
repeat = original(xp,vec,rp,mask,transformer_options=options).clone()
torch.cuda.synchronize()
results['native_repeat_max_abs'] = float((reference.float()-repeat.float()).abs().max())
print('REAL_BLOCK_REFERENCE_OK',results['native_repeat_max_abs'],flush=True)
before = counters['stats']['unique_graphs']
started = time.perf_counter()
compiled = diffusion.blocks[0](x,vec,rope,transformer_options=options.copy())
torch.cuda.synchronize()
results['compile_and_first_call_seconds'] = time.perf_counter()-started
results['compiled_graphs'] = counters['stats']['unique_graphs']-before
results['max_abs'] = float((compiled.float()-reference.float()).abs().max())
results['relative_l2'] = float((compiled.float()-reference.float()).norm()/reference.float().norm())
assert results['compiled_graphs']>0
assert torch.isfinite(compiled).all()
after = original(xp,vec,rp,mask,transformer_options=options)
torch.cuda.synchronize()
results['reference_after_max_abs'] = float((after.float()-reference.float()).abs().max())
print('COMPILE_COMPARISON',json.dumps(results),flush=True)
(OUT/'real-block-comparison.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
torch.testing.assert_close(compiled,reference,rtol=0,atol=0)
# A new input must reach the model, even though shape and capacity are unchanged.
x2=x*0.93
xp2,rp2,mask2=fill_tokens(x2,rope,64)
reference2=original(xp2,vec,rp2,mask2,transformer_options=options)
compiled2=diffusion.blocks[0](x2,vec,rope,transformer_options=options.copy())
torch.cuda.synchronize()
torch.testing.assert_close(compiled2,reference2,rtol=0,atol=0)
assert not torch.equal(compiled,compiled2)
results['changed_input_verified']=True
results['passed']=True
(OUT/'real-loader.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print(json.dumps(results,indent=2),flush=True)
mm.unload_all_models()
del model,clip,vae
gc.collect()
