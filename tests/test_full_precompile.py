# SPDX-License-Identifier: GPL-3.0-only
"""CPU contract tests for full-model compilation; no torch/ComfyUI dependency."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock,patch

ROOT=Path(__file__).resolve().parents[1]


def load_full(compiler):
    spec=importlib.util.spec_from_file_location('_full_graph_contract',ROOT/'full_precompile.py')
    module=importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules,{'torch':types.SimpleNamespace(compile=compiler)}):
        spec.loader.exec_module(module)
    return module


class Patcher:
    def __init__(self,model):self.model=model;self.object_patches={}
    def clone(self):
        clone=Patcher(self.model);clone.object_patches=dict(self.object_patches);return clone
    def get_model_object(self,path):
        if path in self.object_patches:return self.object_patches[path]
        out=self.model
        for part in path.split('.'):out=getattr(out,part)
        return out
    def add_object_patch(self,path,value):self.object_patches[path]=value


class Clip:
    def __init__(self,root):self.cond_stage_model=root;self.patcher=Patcher(root)
    def clone(self):
        clone=Clip(self.cond_stage_model);clone.patcher=self.patcher.clone();return clone


def bundle():
    dit=types.SimpleNamespace(forward=lambda x:x)
    model=Patcher(types.SimpleNamespace(diffusion_model=dit))
    transformer=types.SimpleNamespace(forward=lambda x:x)
    clip=Clip(types.SimpleNamespace(clip='qwen3vl_4b',qwen3vl_4b=types.SimpleNamespace(transformer=transformer)))
    autoencoder=types.SimpleNamespace(encode=lambda x:x,decode=lambda x:x)
    vae=types.SimpleNamespace(first_stage_model=autoencoder,patcher=Patcher(autoencoder))
    return model,clip,vae


class FullGraphContract(unittest.TestCase):
    def test_lazy_strict_compile_and_reuse(self):
        compiler=Mock(return_value=lambda x:x+1);module=load_full(compiler)
        original=Mock();entry=module.FullGraphCall(original,'DiT')
        compiler.assert_not_called()
        self.assertEqual(entry(3),4);self.assertEqual(entry(8),9)
        compiler.assert_called_once()
        self.assertTrue(compiler.call_args.kwargs['fullgraph'])
        self.assertTrue(compiler.call_args.kwargs['dynamic'])
        self.assertFalse(compiler.call_args.kwargs['options']['triton.cudagraphs'])
        original.assert_not_called()

    def test_error_has_no_eager_fallback(self):
        original=Mock(return_value='eager')
        compiled=Mock(side_effect=ValueError('unsupported operation'))
        module=load_full(Mock(return_value=compiled));entry=module.FullGraphCall(original,'TE')
        with self.assertRaises(module.FullGraphError) as caught:entry(1)
        self.assertIn('TE',str(caught.exception));self.assertIsInstance(caught.exception.__cause__,ValueError)
        original.assert_not_called()

    def test_dynamic_residency_rejected_before_compiling(self):
        compiler=Mock();module=load_full(compiler)
        class Managed:
            def modules(self):return iter([types.SimpleNamespace(_v=object())])
            def forward(self,x):return x
        entry=module.FullGraphCall(Managed().forward,'DiT')
        with self.assertRaises(module.FullGraphError):entry(1)
        compiler.assert_not_called()

    def test_user_interruption_is_preserved(self):
        class Interrupted(Exception):pass
        compiled=Mock(side_effect=Interrupted('cancelled'))
        module=load_full(Mock(return_value=compiled))
        with patch.dict(sys.modules,{'comfy.model_management':types.SimpleNamespace(InterruptProcessingException=Interrupted)}):
            with self.assertRaises(Interrupted):module.FullGraphCall(lambda x:x,'DiT')(1)

    def test_all_four_entrypoints_are_managed_clones(self):
        compiler=Mock();module=load_full(compiler);model,clip,vae=bundle()
        m,c,v=module.patch_full_models(model,clip,vae)
        self.assertIsNot(m,model);self.assertIsNot(c,clip);self.assertIsNot(v,vae)
        self.assertIsNot(v.patcher,vae.patcher)
        self.assertEqual(set(m.object_patches),{'diffusion_model.forward'})
        self.assertEqual(set(c.patcher.object_patches),{'qwen3vl_4b.transformer.forward'})
        self.assertEqual(set(v.patcher.object_patches),{'encode','decode'})
        self.assertFalse(model.object_patches);self.assertFalse(clip.patcher.object_patches);self.assertFalse(vae.patcher.object_patches)
        self.assertIs(v.first_stage_model,vae.first_stage_model)
        compiler.assert_not_called()

    def test_model_wrapping_does_not_inspect_or_replace_leaf_operators(self):
        module=load_full(Mock());model,clip,vae=bundle()
        leaf=types.SimpleNamespace(_forward=lambda x:x)
        native_leaf=leaf._forward
        model.model.diffusion_model.projection=leaf
        # A compiler adapter has no reason to enumerate quantized layers or
        # unpack their private layouts. These methods must remain unused.
        checks=[]
        for root in [model.model,clip.cond_stage_model,vae.first_stage_model]:
            scan=Mock(side_effect=AssertionError('Weight layout enumeration is outside the compiler boundary'))
            root.named_modules=scan;checks.append(scan)
        m,c,v=module.patch_full_models(model,clip,vae)
        for scan in checks:scan.assert_not_called()
        self.assertIs(leaf._forward,native_leaf)
        self.assertEqual(set(m.object_patches),{'diffusion_model.forward'})
        self.assertEqual(set(c.patcher.object_patches),{'qwen3vl_4b.transformer.forward'})
        self.assertEqual(set(v.patcher.object_patches),{'encode','decode'})

    def test_unknown_encoder_fails_before_patching(self):
        module=load_full(Mock());model,clip,vae=bundle();clip.cond_stage_model=object()
        with self.assertRaises(ValueError):module.patch_full_models(model,clip,vae)
        self.assertFalse(model.object_patches);self.assertFalse(vae.patcher.object_patches)

    def test_mismatched_vae_patcher_rejected(self):
        module=load_full(Mock());model,clip,vae=bundle();vae.patcher=Patcher(object())
        with self.assertRaises(ValueError):module.patch_full_models(model,clip,vae)


class LoaderOptions(unittest.TestCase):
    def load_node(self):
        name='_k2_loader_contract';spec=importlib.util.spec_from_file_location(name,ROOT/'__init__.py',submodule_search_locations=[str(ROOT)])
        module=importlib.util.module_from_spec(spec)
        class DiT:pass
        diffusion=DiT();model=types.SimpleNamespace(get_model_object=lambda key:diffusion)
        clip=object();vae=object()
        partial=Mock(return_value='partial-model');full=Mock(return_value=('full-model','full-clip','full-vae'));hot=Mock(side_effect=lambda m:('hot',m))
        stubs={name:module,'folder_paths':types.SimpleNamespace(get_filename_list=lambda key:[]),
          'nodes':types.SimpleNamespace(UNETLoader=lambda:types.SimpleNamespace(load_unet=lambda *a:(model,)),CLIPLoader=lambda:types.SimpleNamespace(load_clip=lambda *a,**kw:(clip,)),VAELoader=type('VAE',(),{'vae_list':lambda self:[],'load_vae':lambda self,*a:(vae,)})),
          'comfy.ldm.krea2.model':types.SimpleNamespace(SingleStreamDiT=DiT),
          name+'.runtime':types.SimpleNamespace(patch_krea2=partial),name+'.hot_lora':types.SimpleNamespace(enable_hot_lora=hot),name+'.full_precompile':types.SimpleNamespace(patch_full_models=full)}
        with patch.dict(sys.modules,stubs):spec.loader.exec_module(module)
        return module,model,clip,vae,partial,full,hot

    def test_old_workflow_defaults_are_unchanged(self):
        module,m,c,v,partial,full,hot=self.load_node()
        self.assertEqual(module.K2PrecompileLoader().load('dit','te','vae',False,5120),(m,c,v))
        partial.assert_not_called();full.assert_not_called();hot.assert_not_called()
        optional=module.K2PrecompileLoader.INPUT_TYPES()['optional']
        self.assertEqual(list(optional),['hot_lora','full_precompile'])
        self.assertFalse(optional['full_precompile'][1]['default'])

    def test_partial_mode_is_preserved(self):
        module,m,c,v,partial,full,hot=self.load_node()
        result=module.K2PrecompileLoader().load('dit','te','vae',True,5120)
        self.assertEqual(result,('partial-model',c,v))
        partial.assert_called_once_with(m,5120,compile_blocks=True)
        full.assert_not_called();hot.assert_not_called()

    def test_full_mode_takes_priority_and_preserves_hot_switch(self):
        module,m,c,v,partial,full,hot=self.load_node()
        result=module.K2PrecompileLoader().load('dit','te','vae',True,5120,hot_lora=True,full_precompile=True)
        self.assertEqual(result,(('hot','full-model'),'full-clip','full-vae'))
        full.assert_called_once_with(m,c,v);partial.assert_not_called();hot.assert_called_once_with('full-model')

if __name__=='__main__':unittest.main()
