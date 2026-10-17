# SPDX-License-Identifier: GPL-3.0-only
"""Probe strict full-graph compatibility of an actual K2 checkpoint."""
import gc
import importlib.util
import json
from pathlib import Path
import sys
from _paths import configure
paths=configure(Path(__file__).stem)
sys.path.insert(0,str(paths.comfy));sys.argv=[sys.argv[0]]
import torch
import nodes
import comfy.model_management as mm
from utils.extra_config import load_extra_path_config
extra_paths=paths.comfy/'extra_model_paths.yaml'
if extra_paths.is_file():load_extra_path_config(str(extra_paths))

spec=importlib.util.spec_from_file_location('_real_full_node',paths.root/'__init__.py',submodule_search_locations=[str(paths.root)])
plugin=importlib.util.module_from_spec(spec);sys.modules[spec.name]=plugin;spec.loader.exec_module(plugin)
full=sys.modules[spec.name+'.full_precompile']
torch.set_grad_enabled(False)
model,clip,vae=plugin.K2PrecompileLoader().load(paths.args.dit_name,paths.args.text_encoder_name,paths.args.vae_name,False,0,full_precompile=True)
diffusion=model.get_model_object('diffusion_model');native=diffusion.forward
mm.load_models_gpu([model],memory_required=512*1024*1024)
device=mm.get_torch_device();dtype=diffusion.dtype
torch.manual_seed(887)
x=torch.randn(1,16,1,8,12,device=device,dtype=dtype)
time=torch.tensor([.5],device=device);context=torch.randn(1,3,30720,device=device,dtype=dtype)
reference=native(x,time,context).detach().clone();torch.cuda.synchronize()
assert torch.isfinite(reference).all()
result={'checkpoint':paths.args.dit_name,'native_forward_finite':True,'full_graph_supported':False}
try:
    output=diffusion(x,time,context);torch.cuda.synchronize()
    assert output.shape==reference.shape and torch.isfinite(output).all()
    difference=(output-reference).float()
    result.update(full_graph_supported=True,output_finite=True,output_shape=list(output.shape),
                  max_abs=float(difference.abs().max()),
                  relative_l2=float(difference.norm()/reference.float().norm().clamp_min(1e-12)))
except full.FullGraphError as error:
    result.update(error=str(error),cause_type=type(error.__cause__).__name__)
(paths.out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2),flush=True)
mm.unload_all_models()
del model,clip,vae,diffusion,native;gc.collect()
if not result['full_graph_supported']:raise SystemExit(1)
