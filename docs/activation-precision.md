# DiT 激活精度与权重格式

K2 模型加载器新增可选参数 **DiT 激活精度**（API 字段 `activation_dtype`）：

| 值 | 行为 |
|---|---|
| `auto` | 默认值，保持 ComfyUI 的原有选择，旧工作流不需要修改 |
| `fp16` | DiT 使用 FP16 激活 |
| `bf16` | DiT 使用 BF16 激活 |

这个选择作用于 DiT 输入和条件特征，传播到投影输出、注意力、MLP、调制和残差。不改变独立文本编码器和 VAE 的精度设置；RMSNorm 统计、注意力 softmax 累加、位置旋转矩阵及量化 scale 仍使用各算子需要的 FP32 计算。

## 与权重格式组合

激活精度与已加载权重的存储精度分开。每个投影按自己的权重格式注册，不要求整个 checkpoint 的所有层使用同一种存储格式。

| 已加载权重 | FP16 激活计算路径 |
|---|---|
| 普通 FP16 | FP16 矩阵乘法 |
| 普通 BF16 / FP32 | 使用时转换为 FP16，原存储张量不改写 |
| ConvRot W4A4 INT4 | 保留 INT4 权重，旋转/量化输入，输出 FP16；兼容条件下使用融合 kernel |
| TensorWise INT8 | INT8 矩阵乘法，反量化输出 FP16 |
| ConvRot INT8 | 旋转/量化输入，INT8 矩阵乘法，输出 FP16 |
| TensorCore FP8 E4M3 | FP8 投影，输出 FP16；维度不适配时使用现有反量化计算路径 |
| TensorCore FP8 E5M2 | 保留 FP8 存储，现有反量化路径使用 FP16 输入及矩阵乘法 |

这里指项目已经接入的 ComfyUI/comfy-kitchen 布局。FP16 选项不会把 GGUF、NF4、MXFP8、NVFP4 或其他未接入格式自动转换成本项目支持的格式。

## 实现

`precision.py` 通过 ModelPatcher 的模型级 object patches 设置 `manual_cast_dtype` 与 DiT 的 dtype 元数据。普通 Comfy 线性层启用使用时转换；MixedPrecisionOps 保持输入精度驱动的量化计算。

没有调用 `set_model_compute_dtype()`：该接口会同时设置 `force_cast_weights=True`，使量化层绕开原量化矩阵乘法。新的精度设置不启用该标记，不改写量化权重的原始字节或 scale。设置由对应 ModelPatcher 管理，随模型挂载/撤回，避免多个模型互相覆盖。

VBAR 的视图按 ComfyUI 的工作几何创建；CPU 存储 dtype 与工作 dtype 不同时，先转换载荷再复制，避免把 BF16、FP16 或 FP32 字节按另一种浮点格式解释。CPU 原存储保持不变。

Kohaku FA、Q/K GEMM 后处理、门控残差、联合投影及归一化量化 kernel 已有 FP16 分支；它们根据输入/输出张量精度选择分支。形状调优的缓存键包含激活 dtype，FP16 与 BF16 分开选择配置。仍然只有一个完整 DiT 编译入口。原生算子的 Inductor 外部调用接口在编译前明确登记，避免隐式登记时递归格式化整段 IR 的开销。

FP8 的动态输入 scale 对 FP16 输入先转 FP32 再做除法，避免 scale 在转 FP32 之前就已经下溢。全零输入使用正的最小 FP32 正常值作为动态 scale；checkpoint 中显式提供的 input scale 保持原值。

## 使用

重启 ComfyUI 后，在 K2 模型加载器选择 `fp16`。该参数同时适用于关闭预编译、完整 DiT 编译，以及开启融合前向算子的路径。已有 API 工作流可在加载器 inputs 中加入：

```json
{"activation_dtype": "fp16"}
```

独立构建和依赖版本见 [构建说明](build.md)。改动涉及 `native_runtime.cpp`，首次使用会生成新的 C++ 执行计划库；CUDA 融合源码无需因精度选项重新修改。

## FP16 前向测速

真实 ConvRot INT4 Krea2 权重，RTX 4060 Laptop 8 GB，PyTorch 2.9.1+cu130，FP16 激活，batch=1，4050 token（3952 图像＋98 文本，无额外填充），28 层，VBAR 预留 1536 MiB。

一次首次调用、两次预热之后，五次热态前向为 1151.197 / 1147.247 / 1154.944 / 1148.661 / 1150.637 ms，中位数 **1150.637 ms**。首次调用含编译/调优约 62.516 秒，使用已有磁盘缓存，不能当作空缓存编译耗时。

输入输出均为 FP16，`force_cast_weights=False`，捕获为一张 806 节点完整图。此处使用构造 latent 和文本特征，不含独立 TE、采样器、VAE。原始记录见 [FP16 测速](benchmarks/fp16-4050-forward.json)。这是单独一次速度测量，没有做数值/出图对照，也未逐一运行其他权重格式。

此前 1.175 秒/次的公开测速使用 **BF16 激活**，不是 FP16 速度结果。两份记录为独立运行，未做同一实例的 FP16/BF16 交错比较。此次不新增数值校验文件，不将先前版本的精度对照作为本版本的验收结果。
