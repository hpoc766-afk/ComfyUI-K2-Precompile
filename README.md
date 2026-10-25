# ComfyUI-K2-Precompile · Kohaku 后端调度

用于 ComfyUI 的 Krea2 模型加载节点，提供可选的模型计算预编译和 LoRA 热切换。

节点名称：**K2 模型加载器**，分类：`K2/模型加载`。输出依次为模型、文本编码器和 VAE，可连接普通 LoRA、文本编码及采样节点。

融合路径使用 **KohakuKernelRouter**：编译准备时按形状比较联合 kernel 与原后端组合，缓存并绑定获胜实现；注意力使用适配 GQA 的 **Kohaku FA**，48 个 Q 头直接共享 12 个 K/V 头，不复制 K/V、不保存完整注意力分数表。复用部分来自 KohakuMossTrainer，保留其许可及命名要求，见 [后端选择说明](docs/backend-routing.md)。

## 安装

在已有 ComfyUI 目录执行：

```bash
git clone https://github.com/hpoc766-afk/ComfyUI-K2-Precompile.git custom_nodes/ComfyUI-K2-Precompile
```

重启 ComfyUI 后使用。更新：

```bash
git -C custom_nodes/ComfyUI-K2-Precompile pull --ff-only
```

只保留一份安装，避免旧副本或目录联接造成同名节点重复注册。需要支持 Krea2、`PatcherInjection` 与 LoRA 旁路 API 的 ComfyUI，以及相互兼容的 PyTorch、comfy-kitchen 和 GPU 编译环境。本节点不下载权重，也不自动升级依赖。

节点还包含有界权重缓冲复用、输入量化与权重搬运重叠，以及兼容 INT4 路径的 RMSNorm/调制/旋转量化融合；见 [权重 I/O 与输入量化](docs/io-input-quant.md)。

启用当前原生整图路径前，按 [构建说明](docs/build.md) 生成 `native_dispatch` 和所需融合库。首次调用还会编译 C++ 执行入口与 Triton FA。关闭预编译时可直接沿用普通模型加载。

## 参数

| 参数 | 作用 |
|---|---|
| 主干模型 | 选择 Krea2 DiT 权重 |
| 文本编码器 | 选择 Krea2 使用的文本编码器权重 |
| 变分自编码器 | 选择图像与潜空间转换的 VAE 权重 |
| DiT 激活精度 | `auto` / `fp16` / `bf16`；独立选择激活精度，保留各层权重的存储格式及量化计算 |
| 启用预编译 | 默认关闭；整个 K2 DiT 前向以单入口、`fullgraph=True` 编译 |
| 预填充令牌数 | 旧工作流兼容参数，当前使用实际 token 长度，不执行预填充 |
| 启用 LoRA 热插拔 | 默认关闭；后续普通 LoRA 节点使用兼容的低秩旁路 |
| 融合前向算子 | 默认关闭；融合归一化调制、激活乘法、Q/K 后处理，以及兼容 INT4 路径的 Q/K 投影和输出门控残差 |
| 全部预编译（模型计算） | 默认关闭；在完整 DiT 编译之外，另编译 TE Transformer 和 VAE 编解码入口 |

开启 `启用预编译` 即使用完整 DiT 编译。`全部预编译` 还会编译 TE/VAE，要求它们的后端支持完整追踪。两个开关都关闭时沿用原生计算。

**FP16 激活**可配合已有的 BF16/FP16/FP32 普通权重、ConvRot INT4、TensorWise/ConvRot INT8，以及 FP8 E4M3/E5M2 权重使用。它不启用强制权重反量化，也不改变 TE/VAE 精度；默认 `auto` 保留旧工作流行为。格式范围和计算方式见 [激活精度说明](docs/activation-precision.md)。

## 独立 Token 计数器

新增 **K2 Token 计数器**（`K2TokenCounter`，分类 `K2/工具`），与模型加载器分开使用。输入正向提示词、latent，可选负向提示词与对齐倍数。使用本地 Krea2 tokenizer，只读取 latent 形状，不加载 TE/DiT/VAE 权重，也不修改采样或预填充设置。运行后在节点内显示统计，并输出六个整数及说明文本：正向原提示词、负向原提示词、latent、DiT 文本、联合长度、预填充参考。

- 原提示词分词数不含模板；DiT 文本长度按 Krea2 编码器裁掉系统和用户开头后的实际序列统计，保留结尾特殊 token。12 层文本特征不乘入序列长度。
- latent token 数为 `ceil(H_latent / 2) × ceil(W_latent / 2)`，包含奇数尺寸的 patch 补齐，不乘 batch 或帧数。
- 联合长度为 `latent_tokens + max(positive_text_tokens, negative_text_tokens)`，正负不相加。仅使用正向的 CFG=1 场景可能比这个参考值更小。
- 参考容量向上对齐，默认 64；对齐倍数设为 1 得到精确联合长度。例如 832×1216 图像对应 latent 152×104，占 3,952 token；DiT 文本为 98 时，联合 4,050，参考 4,096。

将同一个 STRING 提示词输出分支连接到计数器和文本编码节点，确保内容一致；输入框可右键转为输入。计数结果用于查看实际输入规模。旧参数 `预填充令牌数` 保留以加载已有工作流，整图模式不使用它。从独立空 latent 节点计数时也可连接整数输出；若 latent 已依赖同一个加载器的下游采样/编码，不要回接形成循环。

当前统计纯文本 Krea2 文生图，不计额外参考图 latent，也不支持图片文本嵌入或 `embedding:` 展开。使用原生 Krea2 模板；如外部插件改变了文本模板/编码裁剪规则，应以实际 conditioning 长度为准。新节点和节点内显示脚本在 ComfyUI 重新加载插件后可用。

## 实现方式

**完整 DiT 编译**只在模型前向入口调用 `torch.compile(fullgraph=True, dynamic=True)`。图覆盖文本融合、时间步处理、全部主干 block 和最终输出层。没有逐 block 或逐算子编译入口。编译后日志显示 `captured complete DiT graph` 及 FX 节点数量；输入布局、LoRA 或其他 guards 变化时可能重新捕获另一张完整图。

量化投影通过图内算子调用 INT4、INT8、FP8 E4M3/E5M2、BF16/FP16 的对应计算路径。原生执行计划在编译准备阶段注册权重格式和 VBAR 元数据，运行时由 C++ 管理权重准备、传输、完成事件和临时缓冲，不逐算子回调 Python。普通线性 LoRA 通过低秩旁路计算；不支持的投影或权重处理会报告错误，不能保证所有第三方补丁均可进入原生计划。

开启 **融合前向算子** 后，SM89 ConvRot INT4 可用共享输入量化、Q/K GEMM 内归一化与 RoPE、输出投影门控残差，以及 RMSNorm/调制/旋转量化融合。Kohaku 在编译准备阶段按形状择优，热态使用绑定的原生路径。输入激活及打包结果不跨前向缓存，低精度舍入点在内核中显式保留。早期独立投影实现及历史记录见 [前向融合说明](docs/forward-fusion.md)，当前路径见 [后端调度](docs/backend-routing.md) 和 [权重 I/O 与输入量化](docs/io-input-quant.md)。

**全部预编译模式**额外为 TE Transformer、VAE `encode/decode` 各建立完整编译入口。TE/VAE 的动态权重及量化能力仍由各自后端决定。此模式不启用 CUDA Graph。模型级入口通过 ModelPatcher 挂载与恢复，DiT 内部的图适配方法只在调用期间安装，并在返回或异常时恢复。

BF16 参考测速（提交 `1993915`）环境为 Windows 11、RTX 4060 Laptop 8 GB、Python 3.12.10、PyTorch 2.9.1+cu130，真实 Krea2-Turbo ConvRot INT4 权重，BF16 激活，动态 VBAR 预留 1536 MiB。832×1216 对应 3952 图像 token，加构造的 98 文本 token，共 **4050 token，额外填充 0**。`4096` 仅是调优长度分桶。

同一模型实例、五轮交错热态 DiT 前向：

| 配置 | 中位耗时 |
|---|---:|
| 之前的编译路径，已含 Kohaku FA | 1.359 秒 |
| 缓冲复用与输入量化/搬运重叠 | 1.207 秒 |
| 再加归一化调制/旋转量化融合 | **1.175 秒** |

合计减少约 13.5%。三种配置各捕获一张完整图；最终路径每次前向实际执行 56 次归一化量化融合。每次 H2D 仍约 2.451 GiB。此处测的是构造输入下的模型前向，不含采样器、独立文本编码和 VAE，不能作为完整出图耗时。原始计时与运行计数见 [测速数据](docs/benchmarks/io-input-quant-4050.json)。

FP16 激活也已在真实 INT4、4050 token、28 层下完成前向测速：五次热态中位数 **1.151 秒**，输入输出均为 FP16，量化计算保持开启，一张 806 节点完整图。该记录与上表为独立运行，见 [FP16 使用与测速](docs/activation-precision.md)。其他权重格式沿用对应已接入路径，本次未逐格式追加实测。

**当前 Kohaku FA 和归一化量化变更仅完成上述速度测量，尚未追加数值或出图对照。** 旧版前向的逐值一致、769 次 kernel 启动和旧采样计时属于历史实现，不代表当前版本。

`fullgraph=True` 要求每个入口可完整追踪。不支持的操作或执行错误会标出组件并抛出异常，不会静默转成部分编译或 eager 执行。

**LoRA 热插拔**接入普通 `LoraLoader`、`LoraLoaderModelOnly` 及调用标准加载 API 的节点。标准线性 LoRA 使用 `base(x) + scale · up(down(x))`，保留实际 rank，不展开稠密 AB，不将增量重新量化到底座。串联时按顺序注入、反序撤回，闲置因子移回 CPU。文本编码器沿用原 `strength_clip` 规则；LoKr、DoRA、mid/reshape 等不兼容形式继续原生挂载。

## 边界

- 全部预编译是**各模型入口分别形成完整图**，不把分词、ComfyUI 调度、采样器循环、VSR 或文件读写合成一张工作流图。
- K2 DiT 已接入原生量化和动态 VBAR 算子；TE/VAE 的严格整图能力单独由其后端决定。
- 首次编译的耗时应与复用同一模型实例后的执行耗时分开记录。形状或模型配置改变可能重新编译。

## 源码与检查

- [`precision.py`](precision.py)：模型级 FP16/BF16 激活选择，与权重存储格式分开。
- [`__init__.py`](__init__.py)：节点输入、模型加载与模式选择。
- [`runtime.py`](runtime.py)：旧加载器参数到完整 DiT 编译入口的兼容接口。
- [`graph_ops.py`](graph_ops.py)：图内原生算子、当前权重依赖与可逆方法适配。
- [`graph_model.py`](graph_model.py)：完整 DiT 的可追踪计算流程。
- [`forward_kernels.py`](forward_kernels.py)：归一化调制、激活乘法和 Q/K 后处理融合。
- [`projection_fusion.py`](projection_fusion.py)：组内共享量化和 Q/K 投影收尾融合接入。
- [`kernels/qk_norm_rope.cu`](kernels/qk_norm_rope.cu)：INT4 Q/K GEMM、归一化和 RoPE 融合内核。
- [`full_precompile.py`](full_precompile.py)：严格完整图入口及受管理的方法替换。
- [`hot_lora.py`](hot_lora.py)：普通 LoRA 节点的旁路接入与还原。
- [`token_counter.py`](token_counter.py)：独立的真实 tokenizer、latent patch 和预填充参考计数。
- [`web/token_counter.js`](web/token_counter.js)：节点内只读统计显示。

无需 GPU 的源码和接口检查：

```bash
python scripts/check_source.py
python -m unittest discover -s tests -p "test_*.py"
```

可选 GPU 验证使用已有 ComfyUI Python：

```bash
python tests/verify_full_precompile.py --comfy-dir "PATH_TO_COMFYUI"
python tests/verify_full_krea.py --comfy-dir "PATH_TO_COMFYUI"
python tests/verify_hot_lora.py --comfy-dir "PATH_TO_COMFYUI"
```

`verify_full_precompile.py` 检查计算入口、变化输入与方法恢复；`verify_full_krea.py` 使用小型随机 Krea2 网络检查完整前向。实际权重检查见 `tests/verify_real_loader.py`、`tests/verify_real_hot_lora.py` 和 `tests/verify_real_full_precompile.py`；最后一项以小输入探测 DiT 整图兼容性，不是完整生成。脚本支持 `--help`，默认产物写入 `.test-output/`。CI 只检查 CPU 源码、接口和编译控制，不代替实际模型的兼容性与性能验证。

## 许可证

本项目自有代码采用 [GNU GPL v3](LICENSE)。`KohakuKernelRouter/` 的复用文件采用保留在 [third_party/KohakuMoss/LICENSE](third_party/KohakuMoss/LICENSE) 中的 KohakuMoss 许可；comfy-kitchen 和 CUTLASS 的许可与归属保存在 `third_party/`。模型结构和旁路 API 使用 [ComfyUI](https://github.com/Comfy-Org/ComfyUI) 的实现；来源和修改说明见 [NOTICE](NOTICE)。本仓库不分发模型权重。
