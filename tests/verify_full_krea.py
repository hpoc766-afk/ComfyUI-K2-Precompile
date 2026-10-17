# SPDX-License-Identifier: GPL-3.0-only
"""Compile a complete small resident Krea2 forward, without checkpoint downloads."""
import gc
import importlib.util
import json
from pathlib import Path
import sys
from _paths import configure
paths=configure(Path(__file__).stem)
sys.path.insert(0,str(paths.comfy));sys.argv=[sys.argv[0]]
import torch
import comfy.ops
from comfy.ldm.krea2.model import SingleStreamDiT
spec=importlib.util.spec_from_file_location('_full_krea_validation',paths.root/'full_precompile.py')
full=importlib.util.module_from_spec(spec);spec.loader.exec_module(full)
torch.set_grad_enabled(False);torch.manual_seed(4813)
model=SingleStreamDiT(features=64,tdim=16,txtdim=32,heads=4,kvheads=2,
    multiplier=2,layers=2,patch=2,channels=4,txtlayers=2,txtheads=4,txtkvheads=2,
    device='cuda',dtype=torch.float32,operations=comfy.ops.manual_cast)
for parameter in model.parameters():parameter.uniform_(-.1,.1)
x=torch.randn(1,4,8,12,device='cuda');context=torch.randn(1,3,64,device='cuda');time=torch.tensor([.5],device='cuda')
reference=model(x,time,context).detach().clone();torch.cuda.synchronize()
compiled=full.FullGraphCall(model.forward,'Krea2完整前向')
result={'model':'SingleStreamDiT','features':64,'blocks':2,'quantized':False,'resident':True}
try:
    output=compiled(x,time,context);torch.cuda.synchronize()
    result['max_abs']=float((output-reference).abs().max())
    torch.testing.assert_close(output,reference,rtol=1e-4,atol=1e-5)
    x2=x*.93;expected=model(x2,time,context);actual=compiled(x2,time,context);torch.cuda.synchronize()
    torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-5)
    result.update(passed=True,changed_input_checked=True)
except Exception as error:
    import traceback
    result.update(passed=False,error=str(error),traceback=traceback.format_exc())
(paths.out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2),flush=True)
del compiled,model;gc.collect()
if not result['passed']:raise SystemExit(1)
