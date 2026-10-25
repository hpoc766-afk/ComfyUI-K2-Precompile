# SPDX-License-Identifier: GPL-3.0-only
"""Build the fused RMSNorm/modulation/ConvRot INT4 input quantizer."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kitchen-dir', type=Path, required=True)
    parser.add_argument('--nvcc', default='nvcc')
    parser.add_argument('--vcvars', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    ops = args.kitchen_dir.resolve() / 'comfy_kitchen/backends/cuda/ops'
    original = ops / 'convrot_w4a4.cu'
    text = original.read_text(encoding='utf-8')
    marker = '\ntemplate<typename InType, int BLOCK_THREADS, bool STOCHASTIC>\n__global__ void quantize_int4_rowwise_kernel('
    if marker not in text:
        parser.error('This comfy-kitchen revision has no recognized ConvRot helper prefix')
    helpers = text[:text.index(marker)] + '\n} // original anonymous namespace\n'
    build = root / '.kernel-build' / 'norm_quant_objects'
    build.mkdir(parents=True, exist_ok=True)
    header = build / 'norm_quant_helpers.cuh'
    header.write_text(helpers, encoding='utf-8')
    source = root / 'kernels/norm_quant.cu'
    flags = ['-std=c++17', '-O3', '-arch=sm_89', '--use_fast_math', '--shared',
             '-I', str(build), '-I', str(ops), '-I', str(ops.parent)]
    version = subprocess.run([args.nvcc, '--version'], capture_output=True, text=True, check=True).stdout
    digest = hashlib.sha256(source.read_bytes() + helpers.encode() + version.encode() + repr(flags).encode())
    for path in sorted(ops.parent.rglob('*.cuh')):
        digest.update(path.name.encode()); digest.update(path.read_bytes())
    output = root / '.kernel-build' / ('norm_quant_' + digest.hexdigest()[:16] + ('.dll' if os.name == 'nt' else '.so'))
    command = [args.nvcc, str(source), *flags, '-o', str(output)]
    command += ['-Xcompiler', '/MD,/EHsc,/Zc:__cplusplus'] if os.name == 'nt' else ['-Xcompiler', '-fPIC']
    if os.name == 'nt' and args.vcvars:
        batch = build / 'build.cmd'
        batch.write_text('@echo off\ncall "' + str(args.vcvars.resolve()) + '"\n'
                         'if errorlevel 1 exit /b %errorlevel%\n' + subprocess.list2cmdline(command)
                         + '\nexit /b %errorlevel%\n', encoding='utf-8')
        command = ['cmd.exe', '/d', '/c', str(batch)]
    subprocess.run(command, cwd=build, check=True)
    manifest = {'library': output.name, 'source_hash': hashlib.sha256(source.read_bytes()).hexdigest(),
                'backend_source_hash': hashlib.sha256(original.read_bytes()).hexdigest(),
                'helpers_hash': hashlib.sha256(helpers.encode()).hexdigest(),
                'nvcc': version, 'flags': flags}
    (root / '.kernel-build/norm_quant.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
