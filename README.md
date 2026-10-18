# ComfyUI-K2-Precompile

用于 ComfyUI 的 Krea2 模型加载节点，提供可选的模型计算预编译和 LoRA 热切换。

节点名称：**K2 模型加载器**，分类：`K2/模型加载`。输出依次为模型、文本编码器和 VAE，可连接普通 LoRA、文本编码及采样节点。

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

## 参数

| 参数 | 作用 |
|---|---|
| 主干模型 | 选择 Krea2 DiT 权重 |
| 文本编码器 | 选择 Krea2 使用的文本编码器权重 |
| 变分自编码器 | 选择图像与潜空间转换的 VAE 权重 |
| 启用预编译 | 默认关闭；编译 DiT block 外围的逐元素调制与残差运算 |
| 预填充令牌数 | 仅逐元素模式使用；文本与图像联合 token 总容量，0 表示不补齐 |
| 启用 LoRA 热插拔 | 默认关闭；后续普通 LoRA 节点使用兼容的低秩旁路 |
| 全部预编译（模型计算） | 默认关闭；分别严格编译 TE Transformer、DiT 前向及 VAE 编解码，优先于逐元素模式 |

全部预编译开启时，`启用预编译` 与 `预填充令牌数` 不再参与配置，模型使用实际输入形状。两个编译开关都关闭时沿用原生计算。

## 独立 Token 计数器

新增 **K2 Token 计数器**（`K2TokenCounter`，分类 `K2/工具`），与模型加载器分开使用。输入正向提示词、latent，可选负向提示词与对齐倍数。使用本地 Krea2 tokenizer，只读取 latent 形状，不加载 TE/DiT/VAE 权重，也不修改采样或预填充设置。运行后在节点内显示统计，并输出六个整数及说明文本：正向原提示词、负向原提示词、latent、DiT 文本、联合长度、预填充参考。

- 原提示词分词数不含模板；DiT 文本长度按 Krea2 编码器裁掉系统和用户开头后的实际序列统计，保留结尾特殊 token。12 层文本特征不乘入序列长度。
- latent token 数为 `ceil(H_latent / 2) × ceil(W_latent / 2)`，包含奇数尺寸的 patch 补齐，不乘 batch 或帧数。
- 联合长度为 `latent_tokens + max(positive_text_tokens, negative_text_tokens)`，正负不相加。仅使用正向的 CFG=1 场景可能比这个参考值更小。
- 参考容量向上对齐，默认 64；对齐倍数设为 1 得到精确联合长度。例如 832×1216 图像对应 latent 152×104，占 3,952 token；DiT 文本为 98 时，联合 4,050，参考 4,096。

将同一个 STRING 提示词输出分支连接到计数器和文本编码节点，确保内容一致；输入框可右键转为输入。参考值可填写到加载器的 `预填充令牌数`。从独立空 latent 节点计数时也可连接整数输出；若 latent 已依赖同一个加载器的下游采样/编码，不要回接形成循环。

当前统计纯文本 Krea2 文生图，不计额外参考图 latent，也不支持图片文本嵌入或 `embedding:` 展开。使用原生 Krea2 模板；如外部插件改变了文本模板/编码裁剪规则，应以实际 conditioning 长度为准。新节点和节点内显示脚本在 ComfyUI 重新加载插件后可用。

## 实现方式

**逐元素模式**使用 `torch.compile / Inductor` 分别编译加法、乘法，归一化、Attention、MLP 和权重读取仍沿用原生实现。指定 token 容量时，主干输入补齐到该容量，attention mask 隔离空位，RoPE 保留真实位置；最后一个 block 后截回有效长度，再执行原输出投影。

**全部预编译模式**将 DiT 的完整前向、TE 的 Transformer 前向、VAE 的 `encode` 和 `decode` 注册为受模型管理器控制的编译入口。首次实际调用使用 `torch.compile(backend="inductor", fullgraph=True, dynamic=True)` 构建计划，后续调用复用兼容计划。模型卸载时恢复原方法；LoRA 或输入变化可能使已有 guards 不匹配而产生新计划。编译层只包装原模型计算入口，不识别或展开量化布局，不注册格式专用算子，也不替换内部线性层。所有算子及其编译支持由原模型后端提供。此模式不启用 CUDA Graph。

`fullgraph=True` 要求每个入口可完整追踪。不支持的操作或执行错误会标出组件并抛出异常，不会静默转成部分编译或 eager 执行。

**LoRA 热插拔**接入普通 `LoraLoader`、`LoraLoaderModelOnly` 及调用标准加载 API 的节点。标准线性 LoRA 使用 `base(x) + scale · up(down(x))`，保留实际 rank，不展开稠密 AB，不将增量重新量化到底座。串联时按顺序注入、反序撤回，闲置因子移回 CPU。文本编码器沿用原 `strength_clip` 规则；LoKr、DoRA、mid/reshape 等不兼容形式继续原生挂载。

## 边界

- 全部预编译是**各模型入口分别形成完整图**，不把分词、ComfyUI 调度、采样器循环、VSR 或文件读写合成一张工作流图。
- 动态 VBAR 权重驻留当前不支持严格完整图；需要兼容的静态权重驻留。量化格式是否可编译取决于原后端，本节点不提供格式专用补丁；不支持时会报错，可关闭后使用逐元素或原生模式。
- 首次编译可能耗时并增加内存占用；不保证加速、任意形状免重编译或浮点结果逐值相同。补齐也不会跳过空位上的 Linear/MLP 计算。

## 源码与检查

- [`__init__.py`](__init__.py)：节点输入、模型加载与模式选择。
- [`runtime.py`](runtime.py)：逐元素编译与 token 预填充。
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
python tests/verify_runtime.py --comfy-dir "PATH_TO_COMFYUI"
python tests/verify_hot_lora.py --comfy-dir "PATH_TO_COMFYUI"
```

`verify_full_precompile.py` 检查计算入口、变化输入与方法恢复；`verify_full_krea.py` 使用小型随机 Krea2 网络检查完整前向。实际权重检查见 `tests/verify_real_loader.py`、`tests/verify_real_hot_lora.py` 和 `tests/verify_real_full_precompile.py`；最后一项以小输入探测 DiT 整图兼容性，不是完整生成。脚本支持 `--help`，默认产物写入 `.test-output/`。CI 只检查 CPU 源码、接口和编译控制，不代替实际模型的兼容性与性能验证。

## 许可证

[GNU GPL v3](LICENSE)。模型结构和旁路 API 使用 [ComfyUI](https://github.com/Comfy-Org/ComfyUI) 的实现；上游说明见 [NOTICE](NOTICE)。本仓库不分发模型权重。
