# SPDX-License-Identifier: GPL-3.0-only
"""Run with the ComfyUI Python; no model downloads required."""
import gc
import importlib.util
import json
from pathlib import Path
import sys

from _paths import configure
paths=configure(Path(__file__).stem)
COMFY=paths.comfy
ROOT=paths.root
OUT=paths.out
sys.path.insert(0, str(COMFY))
sys.argv = [sys.argv[0]]
import torch
import comfy.ops
from comfy.ldm.krea2.model import SingleStreamDiT
from comfy.model_patcher import ModelPatcher
from torch._dynamo.utils import counters

spec = importlib.util.spec_from_file_location('k2_precompile_validation', ROOT/'__init__.py', submodule_search_locations=[str(ROOT)])
plugin = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plugin
spec.loader.exec_module(plugin)
patch_krea2 = sys.modules[spec.name+'.runtime'].patch_krea2

torch.set_grad_enabled(False)
torch.manual_seed(2417)
diffusion = SingleStreamDiT(features=64, tdim=16, txtdim=32, heads=4, kvheads=2,
    multiplier=2, layers=2, patch=2, channels=4, txtlayers=2,
    txtheads=4, txtkvheads=2, device='cuda', dtype=torch.float32,
    operations=comfy.ops.manual_cast)
for parameter in diffusion.parameters():
    parameter.uniform_(-0.1, 0.1)
base = torch.nn.Module()
base.diffusion_model = diffusion
model = ModelPatcher(base, torch.device('cuda'), torch.device('cpu'))
originals = [block.forward for block in diffusion.blocks]
results = {'schema_outputs': list(plugin.K2PrecompileLoader.RETURN_TYPES), 'cases': []}

def predict(x, context, capacity=None, compile_blocks=False):
    t = torch.tensor([0.5]*x.shape[0], device='cuda')
    if capacity is None:
        return diffusion(x, t, context)
    patched = patch_krea2(model, capacity, compile_blocks=compile_blocks)
    patched.patch_model(load_weights=False)
    try:
        return diffusion(x, t, context)
    finally:
        patched.unpatch_model(unpatch_weights=False)
        assert all(a == b.forward for a,b in zip(originals,diffusion.blocks)), 'forward patch leaked'

for batch,height,width,text_tokens in [(1,8,12,3),(2,6,8,5)]:
    x = torch.randn(batch,4,height,width,device='cuda')
    context = torch.randn(batch,text_tokens,64,device='cuda')
    native = predict(x,context)
    shapes = []
    hooks = [b.attn.register_forward_pre_hook(lambda m,a: shapes.append(a[0].shape[1])) for b in diffusion.blocks]
    last_shapes = []
    hooks += [diffusion.last.register_forward_pre_hook(lambda m,a:last_shapes.append(a[0].shape[1]))]
    padded = predict(x,context,48)
    for hook in hooks: hook.remove()
    torch.testing.assert_close(padded,native,rtol=1e-4,atol=1e-5)
    assert shapes == [48,48]
    valid = text_tokens + (height//2)*(width//2)
    assert last_shapes == [valid]
    before = counters['stats']['unique_graphs']
    compiled = predict(x,context,48,True)
    graphs = counters['stats']['unique_graphs']-before
    torch.testing.assert_close(compiled,padded,rtol=1e-3,atol=1e-4)
    assert graphs > 0, 'No compiled graph produced'
    results['cases'].append({'batch':batch,'latent':[height,width], 'text_tokens':text_tokens,
        'valid_tokens':valid,'capacity':48,'block_lengths':shapes,'projection_lengths':last_shapes,
        'padding_max_abs':float((padded-native).abs().max()),
        'compiled_max_abs':float((compiled-padded).abs().max()),'compiled_graphs':graphs})

x = torch.randn(1,4,8,12,device='cuda')
context = torch.randn(1,3,64,device='cuda')
try:
    predict(x,context,16)
except ValueError as error:
    assert '27' in str(error) and '16' in str(error)
    results['overflow_rejected'] = True
else:
    raise AssertionError('Token overflow was not rejected')

native = predict(x,context)
no_padding = predict(x,context,0,True)
torch.testing.assert_close(no_padding,native,rtol=1e-3,atol=1e-4)
results['zero_capacity_max_abs'] = float((native-no_padding).abs().max())
results['restored_after_all_cases'] = all(a == b.forward for a,b in zip(originals,diffusion.blocks))
results['passed'] = True
(OUT/'tiny-krea2.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print(json.dumps(results,indent=2),flush=True)
del model
gc.collect()
