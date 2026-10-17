# SPDX-License-Identifier: GPL-3.0-only
"""GPU checks for strict model entry graphs and real ModelPatcher restoration."""
import copy
import importlib.util
from pathlib import Path
import sys
from _paths import configure

paths=configure(Path(__file__).stem)
sys.path.insert(0,str(paths.comfy));sys.argv=[sys.argv[0]]
import torch
from comfy.model_patcher import ModelPatcher
from torch._dynamo.utils import counters

spec=importlib.util.spec_from_file_location('_full_model_validation',paths.root/'full_precompile.py')
full=importlib.util.module_from_spec(spec);spec.loader.exec_module(full)
torch.set_grad_enabled(False);torch.manual_seed(1773)

class Network(torch.nn.Module):
    def __init__(self):
        super().__init__();self.layer=torch.nn.Linear(8,8,device='cuda')
    def forward(self,x):return torch.sin(self.layer(x))+x
class Autoencoder(Network):
    def encode(self,x):return torch.tanh(self.layer(x))
    def decode(self,x):return self.layer(x)+x
class Clip:
    def __init__(self,root):self.cond_stage_model=root;self.patcher=ModelPatcher(root,torch.device('cuda'),torch.device('cpu'))
    def clone(self):
        clone=copy.copy(self);clone.patcher=self.patcher.clone();return clone

root=torch.nn.Module();root.diffusion_model=Network()
model=ModelPatcher(root,torch.device('cuda'),torch.device('cpu'))
te_root=torch.nn.Module();te_root.clip='qwen3vl_4b';te_root.qwen3vl_4b=torch.nn.Module();te_root.qwen3vl_4b.transformer=Network()
clip=Clip(te_root)
vae=type('VAE',(),{})();vae.first_stage_model=Autoencoder();vae.patcher=ModelPatcher(vae.first_stage_model,torch.device('cuda'),torch.device('cpu'))
x=torch.randn(2,8,device='cuda')
originals={'dit':root.diffusion_model.forward,'te':te_root.qwen3vl_4b.transformer.forward,'encode':vae.first_stage_model.encode,'decode':vae.first_stage_model.decode}
references={key:fn(x).clone() for key,fn in originals.items()}
m,c,v=full.patch_full_models(model,clip,vae)
entries=[('dit',m,lambda:root.diffusion_model.forward),('te',c.patcher,lambda:te_root.qwen3vl_4b.transformer.forward),('encode',v.patcher,lambda:vae.first_stage_model.encode),('decode',v.patcher,lambda:vae.first_stage_model.decode)]
results={'cases':[],'graph_break_rejected':False}
for name,patcher,get in entries:
    before=counters['stats']['unique_graphs']
    patcher.patch_model(load_weights=False)
    try:
        first=get()(x);torch.cuda.synchronize()
        torch.testing.assert_close(first,references[name],rtol=1e-5,atol=1e-6)
        x2=x*.9;expected=originals[name](x2);changed=get()(x2);torch.cuda.synchronize()
        torch.testing.assert_close(changed,expected,rtol=1e-5,atol=1e-6)
        results['cases'].append({'entry':name,'max_abs':float((first-references[name]).abs().max()),'new_graphs':counters['stats']['unique_graphs']-before})
    finally:
        patcher.unpatch_model(unpatch_weights=False)
        assert get()==originals[name],name+' method was not restored'

# Preserve the original ConvRot CUDA calculation while supplying an opaque
# tensor-only operator and a FakeTensor shape rule to the compiler.
import comfy.quant_ops
class QuantLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        w=torch.randn(128,256,device='cuda',dtype=torch.bfloat16)*.02
        self.weight=torch.nn.Parameter(comfy.quant_ops.QuantizedTensor.from_float(w,'TensorCoreConvRotW4A4Layout'),requires_grad=False)
    def _forward(self,x,weight,bias):return torch.nn.functional.linear(x,weight,bias)
    def forward(self,x):return self._forward(x,self.weight,None)
qroot=torch.nn.Module();qroot.q=QuantLinear()
qpatch=ModelPatcher(qroot,torch.device('cuda'),torch.device('cpu'))
xq=torch.randn(1,32,256,device='cuda',dtype=torch.bfloat16)
expected_q=qroot.q(xq).clone();q_original=qroot.q._forward
full.patch_quantized_linears(qpatch)
qpatch.patch_model(load_weights=False)
try:
    qentry=full.FullGraphCall(qroot.q.forward,'ConvRot完整前向')
    actual_q=qentry(xq);torch.cuda.synchronize()
    torch.testing.assert_close(actual_q,expected_q,rtol=0,atol=0)
    results['convrot_full_graph_exact']=True
finally:
    qpatch.unpatch_model(unpatch_weights=False)
    assert qroot.q._forward==q_original

def unsupported(value):
    if value.sum().item()>0:return value+1
    return value-1
try:full.FullGraphCall(unsupported,'unsupported-control')(x)
except full.FullGraphError:results['graph_break_rejected']=True
else:raise AssertionError('A graph break silently fell back to eager')
results['restoration_checked']=True;results['passed']=True
import json
(paths.out/'result.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print(json.dumps(results,indent=2),flush=True)
# Dispose patchers while ComfyUI callback modules are still available.
import gc
del entries,patcher,model,clip,vae,m,c,v,root,te_root,originals,qroot,qpatch,qentry,q_original
gc.collect()
