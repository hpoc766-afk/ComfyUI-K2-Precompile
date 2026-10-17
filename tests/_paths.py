"""Portable paths/CLI for optional ComfyUI GPU validation scripts."""
# SPDX-License-Identifier: GPL-3.0-only
import argparse
from pathlib import Path
from types import SimpleNamespace


def configure(name):
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description='Optional ComfyUI/Krea2 GPU validation. Requires an existing compatible ComfyUI environment.')
    parser.add_argument('--comfy-dir',required=True,type=Path,help='ComfyUI source directory containing main.py')
    parser.add_argument('--output-dir',type=Path,help='Artifacts directory; defaults to .test-output/<test-name>')
    parser.add_argument('--dit-name',default='Krea2-Turbo-int4-ConvRot-ComfyUI.safetensors')
    parser.add_argument('--text-encoder-name',default='qwen3-vl-4b-heretic_fp8_e4m3fn.safetensors')
    parser.add_argument('--vae-name',default='qwen_image_vae.safetensors')
    parser.add_argument('--lora-a',default='krea2/1999.safetensors',help='First actual model LoRA, for the real hot-LoRA test')
    parser.add_argument('--lora-b',default='krea2/konya_karasue_krea2_lora_r32_a32_1536-000016.safetensors',help='Second actual model LoRA, for the real hot-LoRA test')
    args=parser.parse_args()
    comfy=args.comfy_dir.resolve()
    if not (comfy/'main.py').is_file():parser.error('--comfy-dir must contain main.py')
    out=args.output_dir.resolve() if args.output_dir else root/'.test-output'/name
    out.mkdir(parents=True,exist_ok=True)
    # ComfyUI uses native platform separators in its filename enumerations.
    import os
    args.lora_a=args.lora_a.replace('/',os.sep)
    args.lora_b=args.lora_b.replace('/',os.sep)
    return SimpleNamespace(root=root,comfy=comfy,out=out,args=args)
