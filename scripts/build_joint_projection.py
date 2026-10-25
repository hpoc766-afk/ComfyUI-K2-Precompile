# SPDX-License-Identifier: GPL-3.0-only
"""Build the SM89 joint QKV and dual-GEMM SwiGLU projection kernels."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nvcc', default='nvcc')
    parser.add_argument('--vcvars', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root / 'kernels/joint_projection.cu'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    build = root / '.kernel-build'
    build.mkdir(exist_ok=True)
    output = build / ('joint_projection_' + digest + ('.dll' if os.name == 'nt' else '.so'))
    flags = ['-std=c++17', '-O3', '-arch=sm_89', '--shared', '--fmad=false']
    command = [args.nvcc, str(source), *flags, '-o', str(output)]
    if os.name == 'nt':
        command += ['-Xcompiler', '/MD,/EHsc,/Zc:__cplusplus']
        if args.vcvars:
            batch = build / 'build_joint_projection.cmd'
            batch.write_text('@echo off\ncall "' + str(args.vcvars.resolve()) + '"\n'
                             'if errorlevel 1 exit /b %errorlevel%\n'
                             + subprocess.list2cmdline(command) + '\nexit /b %errorlevel%\n', encoding='utf-8')
            command = ['cmd.exe', '/d', '/c', str(batch)]
    else:
        command += ['-Xcompiler', '-fPIC']
    subprocess.run(command, cwd=build, check=True)
    version = subprocess.run([args.nvcc, '--version'], capture_output=True, text=True, check=True).stdout
    (build / ('joint_projection_' + digest + '.json')).write_text(json.dumps(
        {'library': output.name, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
         'nvcc': version, 'flags': flags}, indent=2), encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
