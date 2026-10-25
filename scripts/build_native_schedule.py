# SPDX-License-Identifier: GPL-3.0-only
"""Build native scheduler CUDA primitives against the existing backend source."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cutlass-dir', type=Path, required=True)
    p.add_argument('--kitchen-dir', type=Path, required=True)
    p.add_argument('--nvcc', default='nvcc')
    p.add_argument('--vcvars', type=Path)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    cuda = a.kitchen_dir.resolve() / 'comfy_kitchen/backends/cuda'
    names = ['convrot_w4a4.cu', 'int8_linear.cu', 'per_tensor_quantize.cu',
             'cublas_gemm_int8.cu', 'cutlass_gemm_int8.cu']
    sources = [cuda / 'ops' / name for name in names] + [root / 'kernels/native_dispatch.cu']
    for path in sources:
        if not path.is_file():
            p.error(f'missing source: {path}')
    version = subprocess.run([a.nvcc, '--version'], capture_output=True, text=True, check=True).stdout
    build = root / '.kernel-build'
    objects_dir = build / 'native_objects'
    objects_dir.mkdir(parents=True, exist_ok=True)
    base = ['-std=c++17', '-O3', '-arch=sm_89', '-DCOMFY_HAVE_CUTLASS',
            '--expt-relaxed-constexpr', '-I', str(cuda), '-I', str(cuda / 'ops'),
            '-I', str(a.cutlass_dir.resolve() / 'include')]
    host = ['-Xcompiler', '/MD,/EHsc,/Zc:__cplusplus,/bigobj'] if os.name == 'nt' else ['-Xcompiler', '-fPIC']
    # Match the backend build for quantizers. The added norm/activation kernels
    # intentionally retain precise division/exp and explicit FMA/rounding sites.
    precise = ['--fmad=false']
    fast = ['--use_fast_math']
    headers = hashlib.sha256(version.encode())
    for path in sorted(cuda.rglob('*.cuh')) + sorted((a.cutlass_dir.resolve() / 'include').rglob('*.h')) + sorted((a.cutlass_dir.resolve() / 'include').rglob('*.hpp')):
        headers.update(path.name.encode()); headers.update(path.read_bytes())
    toolchain = headers.digest()
    objects, commands, component_hashes = [], [], []
    for i, source in enumerate(sources):
        flags = base + host + (precise if i == len(sources) - 1 else fast)
        digest = hashlib.sha256(toolchain + source.read_bytes() + repr(flags).encode()).hexdigest()
        obj = objects_dir / (source.stem + '_' + digest[:16] + ('.obj' if os.name == 'nt' else '.o'))
        if not obj.exists():
            commands.append([a.nvcc, str(source), '-c', *flags, '-o', str(obj)])
        objects.append(obj)
        component_hashes.append(digest)
    digest = hashlib.sha256(repr(component_hashes).encode()).hexdigest()
    output = build / ('native_dispatch_' + digest[:16] + ('.dll' if os.name == 'nt' else '.so'))
    if not output.exists():
        commands.append([a.nvcc, *map(str, objects), '--shared', *host,
                         '-lcublas', '-lcublasLt', '-o', str(output)])
    if os.name == 'nt' and a.vcvars and commands:
        batch = build / 'build_native_dispatch.cmd'
        lines = ['@echo off', 'call "' + str(a.vcvars.resolve()) + '"',
                 'if errorlevel 1 exit /b %errorlevel%']
        for cmd in commands:
            lines += [subprocess.list2cmdline(cmd), 'if errorlevel 1 exit /b %errorlevel%']
        lines += ['exit /b 0']
        batch.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        subprocess.run(['cmd.exe', '/d', '/c', str(batch)], cwd=build, check=True)
    else:
        for cmd in commands:
            subprocess.run(cmd, cwd=build, check=True)
    manifest = {'library': output.name, 'source_hash': hashlib.sha256(sources[-1].read_bytes()).hexdigest(),
                'dependencies_digest': digest, 'nvcc': version,
                'backend_flags': base + host + fast, 'added_kernel_flags': base + host + precise}
    (build / 'native_dispatch.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
