# SPDX-License-Identifier: GPL-3.0-only
"""Build the optional SM89 Q/K GEMM + RMSNorm + RoPE kernel."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cutlass-dir', type=Path, required=True)
    parser.add_argument('--nvcc', default='nvcc')
    parser.add_argument('--vcvars', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    include = args.cutlass_dir.resolve() / 'include'
    if not (include / 'cutlass/cutlass.h').is_file():
        parser.error('CUTLASS include/cutlass/cutlass.h was not found')
    source = root / 'kernels/qk_norm_rope.cu'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    build = root / '.kernel-build'
    build.mkdir(exist_ok=True)
    output = build / (f'qk_norm_rope_{digest}' + ('.dll' if os.name == 'nt' else '.so'))
    command = [args.nvcc, str(source), '-std=c++17', '-O3', '-arch=sm_89', '--shared',
               '-I', str(include), '--expt-relaxed-constexpr', '--fmad=false', '-o', str(output)]
    if os.name == 'nt':
        command += ['-Xcompiler', '/MD,/EHsc,/Zc:__cplusplus']
        if args.vcvars:
            batch = build / 'build_qk.cmd'
            batch.write_text('@echo off\ncall "' + str(args.vcvars.resolve()) + '"\n'
                             'if errorlevel 1 exit /b %errorlevel%\n'
                             + subprocess.list2cmdline(command) + '\nexit /b %errorlevel%\n', encoding='utf-8')
            command = ['cmd.exe', '/d', '/c', str(batch)]
    else:
        command += ['-Xcompiler', '-fPIC']
    subprocess.run(command, cwd=build, check=True)
    print(output)


if __name__ == '__main__':
    main()
