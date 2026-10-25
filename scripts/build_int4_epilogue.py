"""Build the optional SM89 INT4 residual kernel; does not alter ComfyUI/kitchen.

Requires CUDA 12.8+, a C++17 compiler and CUTLASS headers at the pinned revision.
No dependencies are downloaded by this script or by the node.
"""
# SPDX-License-Identifier: GPL-3.0-only
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CUTLASS_REVISION = "d4b4b494c3c51bf6507e7ab09fbafd1e9fa94f39"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutlass-dir", type=Path, required=True)
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--vcvars", type=Path, help="Windows vcvars64.bat if cl is not on PATH")
    args = parser.parse_args()
    include = args.cutlass_dir.resolve() / "include"
    if not (include / "cutlass/cutlass.h").is_file():
        parser.error("--cutlass-dir must contain include/cutlass/cutlass.h")
    source = ROOT / "kernels/int4_residual.cu"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    build = ROOT / ".kernel-build"
    build.mkdir(exist_ok=True)
    output = build / (f"int4_residual_{digest}" + (".dll" if os.name == "nt" else ".so"))
    command = [args.nvcc, str(source), "-std=c++17", "-O3", "-arch=sm_89", "--shared",
               "-I", str(include), "--expt-relaxed-constexpr", "--fmad=false", "-o", str(output)]
    if os.name == "nt":
        command += ["-Xcompiler", "/MD,/EHsc,/Zc:__cplusplus"]
        if args.vcvars:
            # A local command file avoids nesting cmd.exe's special /S /C quoting.
            batch = build / "build.cmd"
            batch.write_text('@echo off\ncall "' + str(args.vcvars.resolve()) + '"\nif errorlevel 1 exit /b %errorlevel%\n'
                             + subprocess.list2cmdline(command) + '\nexit /b %errorlevel%\n', encoding="utf-8")
            command = ["cmd.exe", "/d", "/c", str(batch)]
        elif not shutil.which("cl"):
            parser.error("Run from a VS developer shell or supply --vcvars")
    else:
        command += ["-Xcompiler", "-fPIC"]
    subprocess.run(command, cwd=build, check=True)
    print(f"Built {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
