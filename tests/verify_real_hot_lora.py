# SPDX-License-Identifier: GPL-3.0-only
"""Real K2 INT4, standard LoRA loader, rank16/32 switching and model management."""
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from _paths import configure
paths=configure(Path(__file__).stem)
COMFY=paths.comfy
ROOT=paths.root
OUT=paths.out
sys.path.insert(0,str(COMFY));sys.argv=[sys.argv[0]]
import torch
import nodes,comfy.sd,comfy.model_management as mm
from torch._dynamo.utils import counters

spec=importlib.util.spec_from_file_location('k2_real_hot_validation',ROOT/'__init__.py',submodule_search_locations=[str(ROOT)])
plugin=importlib.util.module_from_spec(spec);sys.modules[spec.name]=plugin;spec.loader.exec_module(plugin)
hot=sys.modules[spec.name+'.hot_lora']
torch.set_grad_enabled(False)
base,clip,vae=plugin.K2PrecompileLoader().load(
    paths.args.dit_name,
    paths.args.text_encoder_name,
    paths.args.vae_name,True,64,hot_lora=True)
assert base.get_attachment(hot._ENABLED)
diffusion=base.get_model_object('diffusion_model')
linear=diffusion.blocks[0].attn.wq
native_forward=linear.forward
packed=linear.weight._qdata
original_hash=hashlib.sha256(packed.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
loader=nodes.LoraLoader()
a_name=paths.args.lora_a
b_name=paths.args.lora_b
A,clip_a=loader.load_lora(base,clip,a_name,0.5,0.0)
raw_a=loader.loaded_lora[1]
B,clip_b=loader.load_lora(base,clip,b_name,0.5,0.0)
raw_b=loader.loaded_lora[1]
A_return,_=loader.load_lora(base,clip,a_name,0.5,0.0)
stack,_=loader.load_lora(A,clip_a,b_name,-0.25,0.0)
zero=nodes.LoraLoaderModelOnly().load_lora_model_only(base,a_name,0.0)[0]
assert zero is base
assert not A.patches and not B.patches and not stack.patches
assert A.patches_uuid==B.patches_uuid==stack.patches_uuid==base.patches_uuid
assert A.model is B.model is stack.model is base.model
RA,_=comfy.sd.load_bypass_lora_for_models(base,None,raw_a,0.5,0)
RB,_=comfy.sd.load_bypass_lora_for_models(base,None,raw_b,0.5,0)

def ranks(model):
    hooks=model.get_attachment(hot._HOOKS) or ()
    return {'hooks':len(hooks),'ranks':sorted({h.adapter.weights[1].shape[0] for h in hooks})}
results={'A':ranks(A),'B':ranks(B),'stack':ranks(stack),'clip_type':type(clip_a).__name__,'vae_type':type(vae).__name__}
print('REAL_HOT_LORA_LOAD_OK',json.dumps(results),flush=True)

torch.manual_seed(827)
features=linear.in_features
dtype=diffusion.dtype
device=mm.get_torch_device()
x=torch.randn(1,32,features,device=device,dtype=dtype)*0.1
vec=torch.randn(1,1,features*6,device=device,dtype=dtype)*0.01
rope=torch.zeros(1,1,32,diffusion.blocks[0].attn.headdim//2,2,2,device=device)
rope[...,0,0]=1;rope[...,1,1]=1
options={'block_index':0,'block_type':'single','total_blocks':28,'img_slice':[0,32]}
previous=[]
def predict(label,model):
    global previous
    mm.load_models_gpu([model],memory_required=512*1024*1024)
    for old in previous:
        for h in old.get_attachment(hot._HOOKS) or ():
            if h not in (model.get_attachment(hot._HOOKS) or ()):
                assert h.original_forward is None
                assert all(not isinstance(t,torch.Tensor) or t.device.type=='cpu' for t in h.adapter.weights)
    out=diffusion.blocks[0](x,vec,rope,transformer_options=options.copy())
    torch.cuda.synchronize()
    assert torch.isfinite(out).all()
    previous=[model]
    print('REAL_BLOCK_OK',label,flush=True)
    return out.detach().cpu().clone()

baseline=predict('baseline_warmup',base)
reference_a=predict('reference_A',RA)
out_a=predict('hot_A',A)
torch.testing.assert_close(out_a,reference_a,rtol=0,atol=0)
del RA;gc.collect()
reference_b=predict('reference_B',RB)
graphs=counters['stats']['unique_graphs']
out_b=predict('hot_B',B)
del RB;gc.collect()
out_return=predict('hot_A_return',A_return)
torch.testing.assert_close(out_b,reference_b,rtol=0,atol=0)
torch.testing.assert_close(out_return,out_a,rtol=0,atol=0)
assert not torch.equal(out_a,out_b)
assert counters['stats']['unique_graphs']==graphs,'Rank/file swap recompiled K2 arithmetic'
stack_out=predict('stack_A_B',stack)
assert not torch.equal(stack_out,out_a)
restored=predict('zero_and_base_restore',zero)
torch.testing.assert_close(restored,baseline,rtol=0,atol=0)
assert linear.forward==native_forward
current_hash=hashlib.sha256(linear.weight._qdata.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
assert current_hash==original_hash,'Packed INT4 base weights changed'
results.update(reference_A_max_abs=0.0,reference_B_max_abs=0.0,return_A_max_abs=0.0,
               base_restore_max_abs=0.0,rank_swap_added_graphs=0,packed_q_weight_sha256=original_hash,
               base_weights_shared=True,passed=True)
(OUT/'real-hot-lora.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print(json.dumps(results,indent=2),flush=True)
mm.unload_all_models()
del base,clip,vae,A,B,A_return,stack,zero,clip_a,clip_b
previous.clear();gc.collect()
