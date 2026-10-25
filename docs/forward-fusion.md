# K2 前向融合（历史独立投影实现）

本文记录早期独立投影版本的数据流与验证。当前实现已接入 [Kohaku 后端调度与 FA](backend-routing.md)、[权重 I/O 与归一化量化融合](io-input-quant.md)。构建步骤见 [构建说明](build.md)。

下文 769 次启动、逐值对照和旧速度结果仅属于当时版本。当前 Kohaku FA、联合算子和归一化量化变更的数值与出图对照未在本轮追加；最新速度数据见 [测速记录](benchmarks/io-input-quant-4050.json)。

开启 **启用预编译** 和 **融合前向算子**。保存工作流的字段仍为 `fuse_epilogue`；它属于各个模型自己的配置。整个 DiT 仍只有一个 `torch.compile(fullgraph=True, dynamic=True)` 入口。这里的多个 GPU kernel 都由同一张完整图调用。

## 数据流

```text
RMSNorm + 调制（一个 kernel）
    ↓
同一输入的旋转/量化一次，供 Q、K、V、attention gate 使用
    ├─ Q INT4 GEMM + 每头 RMSNorm + RoPE（一个 kernel）
    ├─ K INT4 GEMM + 每头 RMSNorm + RoPE（一个 kernel）
    ├─ V GEMM
    └─ attention gate GEMM
    ↓
原生注意力后端
    ↓
sigmoid(attention gate) × 注意力输出（一个 kernel）
    ↓
wo GEMM + block 门控 + 残差（一个 kernel）
    ↓
RMSNorm + 调制（一个 kernel）
    ↓
同一输入的旋转/量化一次，供 MLP gate、up 使用
    ↓
两路 GEMM → SiLU(gate) × up（一个激活乘法 kernel）
    ↓
down GEMM + block 门控 + 残差（一个 kernel）
```

Q、K 是两次 GEMM 启动，各自将归一化和位置旋转做到收尾阶段。一个 CTA 处理若干 token 的完整 128 维 head，将低精度投影暂存在共享内存，完成 RMSNorm、RoPE 后写出。没有把原始 Q/K 投影写回显存。两次 GEMM 共用输入量化结果，但分别管理自身权重生命周期。

MLP 的 gate/up 仍是两次 GEMM，激活与相乘已融合。`down` 依赖激活结果；最终残差加法依赖 `down` 输出，因此这两步不能独立并行。输出 GEMM 内的残差读取与收尾计算由 CUTLASS visitor 组织。

## 精度与兼容

- 归一化、调制、激活乘法和 Q/K 后处理独立于权重量化格式；普通 BF16/FP16、INT8、FP8 E4M3/E5M2 继续使用原投影后端。
- GEMM 内 Q/K 融合和共享输入量化当前针对 SM89、ConvRot W4A4 INT4。Q/K GEMM 融合要求 batch=1、head_dim=128、无 bias 和标准完整旋转布局；其他配置使用原生投影及可用的后处理融合。
- 保留投影、归一化、调制各步和激活输出的 BF16/FP16 舍入。求和顺序匹配本机 PyTorch 2.9 的向量化 RMSNorm；RoPE 乘加顺序匹配本机 comfy-kitchen CUDA 后端。
- 权重由原有 `CastBiasWeightContext` 逐层准备、使用及释放。量化激活只在当前这组投影中共享，不跨前向、模型或 stream 保存。
- 模块的实际权重张量作为图算子的显式参数。动态 VBAR、LoRA、权重补丁仍由原生模块管理。
- 输入 attention patch 在 Q/K 归一化之后、RoPE 之前执行，保留这个插入位置；有该 patch 时不提前融合 Q/K 的 RoPE。
- 自定义投影、LoRA 和模块 hook 执行原模块调用。发生这类调用后清空组内量化缓存，后续投影重新读取当前输入。
- 本功能用于推理。

## 构建 Q/K 融合

新内核源码为 `kernels/qk_norm_rope.cu`，构建入口为 `scripts/build_qk_fusion.py`。依赖 CUTLASS；本机验证使用修订 `d4b4b494c3c51bf6507e7ab09fbafd1e9fa94f39`、CUDA 12.8 和 VS 2022。

Linux 示例：

```bash
python scripts/build_qk_fusion.py --cutlass-dir /path/to/cutlass --nvcc /path/to/cuda/bin/nvcc
```

Windows 示例：

```powershell
python scripts/build_qk_fusion.py --cutlass-dir C:/deps/cutlass --nvcc "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin/nvcc.exe" --vcvars "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Auxiliary/Build/vcvars64.bat"
```

产物写入 `.kernel-build/qk_norm_rope_<源码摘要>.dll` 或 `.so`，加载器使用相同摘要。未构建时，Q/K 仍可使用独立的归一化/RoPE 融合后处理。Triton 负责归一化调制、激活乘法及通用 Q/K 后处理。

现有 INT4 输出门控残差使用原来的 `int4_epilogue` 库。本次没有修改它的 CUDA 源码、库或旧构建脚本。

## 本机验证

RTX 4060 Laptop 8 GB，真实 Krea2 INT4 权重，batch=1，BF16，832×1216（4050 token），28 个主干 block：

- 首次及热态前向、变化输入、卸载后重新加载均与同进程原生前向逐值一致。
- 编译器每次捕获一张完整图；卸载重载引起权重 guards 变化后重新捕获完整图，本轮累计两次捕获。
- CUPTI 连续两次计数都是 **769 次启动、55 种符号**，记录来自最终整图接口。
- 上一版同尺寸融合完整图的计数是 1641 次；本次减少 872 次（约 53.1%）。

一次前向的启动分类：

| 类型 | 次数 |
|---|---:|
| 输入旋转及 INT4 量化 | 112 |
| Q/K GEMM + RMSNorm + RoPE | 56 |
| 其他 INT4 GEMM（含 56 次输出门控残差融合） | 168 |
| 归一化或归一化调制融合 | 66 |
| 激活乘法融合 | 64 |
| Flash Attention | 32 |
| Inductor 生成的其他 Triton kernel | 71 |
| 其他 CUDA kernel | 200 |
| 合计 | 769 |

最终版本同模型五组交错前向（交替先执行原生/融合）的 wall time 中位数：原生 2010.25 ms，融合完整图 1373.41 ms，约减少 31.7%。原生结果为 1921.33 / 2515.79 / 2035.95 / 2010.25 / 1993.47 ms，融合为 1380.65 / 1525.87 / 1356.73 / 1373.41 / 1359.74 ms。包含动态权重管理，有波动；它是单次 DiT 前向计时，不能直接当作整张图或八步采样耗时。

小模型验证包含 BF16/FP16 两种激活精度，各自覆盖普通权重、INT4、INT8、ConvRot INT8、FP8 E4M3 和 FP8 E5M2，并比较变化输入。FP8 E5M2 在本机沿用后端的反量化路径。另检查投影 hook、单个及整个 Q/K 归一化 hook、attention patch、rank 16/32 LoRA、参考图与 batch=2。最终接口还检查了原地修改输入的前置 hook，以及返回输入别名的自定义投影。

以上检查直接在内存中执行，没有新增测试文件。
