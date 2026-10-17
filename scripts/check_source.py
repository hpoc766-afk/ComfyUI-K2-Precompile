"""CPU-only source/localization/CLI checks; no ComfyUI, torch or GPU required."""
# SPDX-License-Identifier: GPL-3.0-only
import ast
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def main():
    files=[p for p in ROOT.rglob('*.py') if not any(part in {'.git','.test-output','__pycache__','.venv','venv'} for part in p.relative_to(ROOT).parts)]
    for path in files:ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
    expected={'dit_name','text_encoder_name','vae_name','precompile','prefill_tokens','hot_lora'}
    translations=[]
    for language in ['zh','zh-CN']:
        node=json.loads((ROOT/'locales'/language/'nodeDefs.json').read_text(encoding='utf-8'))['K2PrecompileLoader']
        assert set(node['inputs'])==expected
        assert set(node['outputs'])=={'0','1','2'}
        translations.append(node)
    assert translations[0]==translations[1]
    tree=ast.parse((ROOT/'__init__.py').read_text(encoding='utf-8'))
    loader=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='K2PrecompileLoader')
    returns=next(ast.literal_eval(n.value) for n in loader.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='RETURN_TYPES' for t in n.targets))
    assert returns==('MODEL','CLIP','VAE')
    for path in (ROOT/'tests').glob('verify_*.py'):
        result=subprocess.run([sys.executable,str(path),'--help'],capture_output=True,text=True,timeout=15)
        assert result.returncode==0,(path.name,result.stderr)
        assert '--comfy-dir' in result.stdout and '--output-dir' in result.stdout
    for raw in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',(ROOT/'README.md').read_text(encoding='utf-8')):
        target=raw.strip('<>').split('#',1)[0]
        if target and '://' not in target:assert (ROOT/target).exists(),target
    assert 'GNU GENERAL PUBLIC LICENSE' in (ROOT/'LICENSE').read_text(encoding='utf-8')
    assert 'GPL-3.0-only' in (ROOT/'NOTICE').read_text(encoding='utf-8')
    print(f'PASS: {len(files)} Python sources parsed; two locales, node outputs, four CLI help interfaces, documentation links and license checked.')
    print('CPU checks only; GPU/model tests were not executed.')

if __name__=='__main__':main()
