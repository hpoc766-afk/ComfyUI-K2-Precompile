# 构建原生执行与融合内核

普通模型加载、关闭预编译时不需要这些本地库。启用当前 DiT 原生整图编译前，先构建 `native_dispatch`；完整的 INT4 优化还使用 Q/K、输出残差、联合投影和归一化量化库。仓库不包含本机 DLL、PyTorch 扩展、模型权重或编译缓存。

## 本机使用的环境

- Windows 11，RTX 4060 Laptop 8 GB（SM89）。
- ComfyUI Python 3.12.10，PyTorch 2.9.1+cu130，ComfyUI 的 `comfy_aimdo` 动态 VBAR。
- 下列手工 CUDA 库使用 CUDA Toolkit 12.8、VS 2022 x64；C++ 执行计划由当前 PyTorch 的扩展构建器编译，需配置匹配该 PyTorch 的 CUDA/MSVC 环境。
- CUTLASS：`d4b4b494c3c51bf6507e7ab09fbafd1e9fa94f39`。
- comfy-kitchen CUDA 源码：`255a43879fe57bbcbecfdb273b46d772b00c5a90`。它是构建依赖源码；运行时仍使用 ComfyUI 环境里的 comfy-kitchen。

当前原生 CUDA 构建脚本面向 SM89。以上为实际测速环境；Linux 构建分支和其他显卡未在这轮实测。

## 准备依赖源码

在节点仓库根目录执行：

```bash
git clone https://github.com/NVIDIA/cutlass.git .kernel-build/deps/cutlass
git -C .kernel-build/deps/cutlass checkout d4b4b494c3c51bf6507e7ab09fbafd1e9fa94f39
git clone https://github.com/Comfy-Org/comfy-kitchen.git .kernel-build/deps/comfy-kitchen
git -C .kernel-build/deps/comfy-kitchen checkout 255a43879fe57bbcbecfdb273b46d772b00c5a90
```

如果目录已经存在，使用自己准备的相应依赖目录。依赖与编译输出都位于被 Git 忽略的 `.kernel-build/`。

## 构建入口

下面的 `python` 指 ComfyUI 使用的 Python，`nvcc` 指选定的 CUDA 编译器。可用 `--nvcc` 指定完整路径；Windows 可为各命令加 `--vcvars` 指向 `vcvars64.bat`。

先构建必需的量化/GEMM及通用计算库：

```bash
python scripts/build_native_schedule.py --cutlass-dir .kernel-build/deps/cutlass --kitchen-dir .kernel-build/deps/comfy-kitchen --nvcc nvcc
```

再构建其余融合库：

```bash
python scripts/build_qk_fusion.py --cutlass-dir .kernel-build/deps/cutlass --nvcc nvcc
python scripts/build_joint_projection.py --nvcc nvcc
python scripts/build_norm_quant.py --kitchen-dir .kernel-build/deps/comfy-kitchen --nvcc nvcc
```

Windows 示例，选用 CUDA 12.8：

```powershell
python scripts/build_norm_quant.py --kitchen-dir .kernel-build/deps/comfy-kitchen --nvcc "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin/nvcc.exe" --vcvars "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Auxiliary/Build/vcvars64.bat"
```

## 输出门控残差库：现存脚本文件名问题

`scripts/build_int4_epilogue.py` 目前仍引用旧的 `int4_residual.cu` 和旧产物名。实际源码是 `kernels/int4_epilogue.cu`，加载器要求 `int4_epilogue_<源码SHA256前16位>.dll/.so`。本次按要求保留该脚本，没有修正。

使用当前源码手工构建即可得到加载器需要的名称。在已初始化 MSVC x64 环境的 PowerShell 中：

```powershell
$k2Hash = (Get-FileHash kernels/int4_epilogue.cu -Algorithm SHA256).Hash.ToLower().Substring(0, 16)
nvcc kernels/int4_epilogue.cu -std=c++17 -O3 -arch=sm_89 --shared -I .kernel-build/deps/cutlass/include --expt-relaxed-constexpr --fmad=false -Xcompiler /MD,/EHsc,/Zc:__cplusplus -o ".kernel-build/int4_epilogue_${k2Hash}.dll"
```

Linux 的对应命令：

```bash
k2_hash=$(python -c 'import hashlib; from pathlib import Path; print(hashlib.sha256(Path("kernels/int4_epilogue.cu").read_bytes()).hexdigest()[:16])')
nvcc kernels/int4_epilogue.cu -std=c++17 -O3 -arch=sm_89 --shared -I .kernel-build/deps/cutlass/include --expt-relaxed-constexpr --fmad=false -Xcompiler -fPIC -o ".kernel-build/int4_epilogue_${k2_hash}.so"
```

这份手工命令没有在发布整理阶段重新执行；本机现有库来自同一源码和相应编译选项。

## 使用

构建完成后重启 ComfyUI，开启「启用预编译」与「融合前向算子」。C++ 执行计划在首次使用时通过 PyTorch 扩展构建器生成；Triton FA 在首次使用时编译。首次调用还包含形状调优，热态复用同一完整图及原生执行计划。CUDA Graph replay 关闭。

当前性能与验证范围见 [权重 I/O 与输入量化](io-input-quant.md)；归一化量化库、Kohaku FA 的最新变更本轮仅运行了速度测量，未追加数值或出图对照。
