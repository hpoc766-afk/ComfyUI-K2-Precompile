# ComfyUI-K2-Precompile

Krea2 模型加载、可选逐元素预编译，以及沿用普通 LoRA 节点操作方式的低秩旁路热切换。

独立公开源码仓库：<https://github.com/hpoc766-afk/ComfyUI-K2-Precompile>。许可证：[GPL-3.0](LICENSE)。

## 安装

进入已有的 ComfyUI 目录，执行：

```bash
git clone https://github.com/hpoc766-afk/ComfyUI-K2-Precompile.git custom_nodes/ComfyUI-K2-Precompile
```

然后重启 ComfyUI，添加“**K2 模型加载器**”节点。更新使用：

```bash
git -C custom_nodes/ComfyUI-K2-Precompile pull --ff-only
```

如果已有旧副本或目录联接，请确认只加载一份，避免同名节点重复注册。

需要已有支持 Krea2 `SingleStreamDiT`、`PatcherInjection` 和 `comfy.weight_adapter` 旁路 API 的 ComfyUI。开启预编译还需要该环境的 `torch.compile / Inductor` 能在目标 GPU 上正常执行。ComfyUI、PyTorch 与 comfy-kitchen 应互相兼容；本节点不自动升级它们，也不下载模型。

## 节点参数

节点：`K2PrecompileLoader`，分类：`K2/模型加载`。
界面输出依次为“模型／文本编码器／变分自编码器”，可接原有 LoRA、文本编码和采样节点。
六个参数均提供中文名称与提示，内部参数标识保持不变，已有工作流无需重新接线。

| 输入 | 作用 |
|---|---|
| 主干模型 | 选择 Krea2 主干权重文件 |
| 文本编码器 | 选择文本编码器，固定按 Krea2 类型加载 |
| 变分自编码器 | 选择图像与潜空间转换所用的权重文件 |
| 启用预编译 | 默认关闭；关闭时不编译、不补齐；LoRA 热插拔由独立开关控制 |
| 预填充令牌数 | 开启编译时，主干联合序列的令牌总容量；默认 5120，0 表示不补齐 |
| 启用 LoRA 热插拔 | 默认关闭；开启后验证 K2 权重名称与适配器维度，让下游 LoRA 加载节点使用低秩旁路热插拔 |

常用的 K2 工作流模型示例（权重不随本仓库分发）：

- DiT：`Krea2-Turbo-int4-ConvRot-ComfyUI.safetensors`
- 文本编码器：`qwen3-vl-4b-heretic_fp8_e4m3fn.safetensors`
- VAE：`qwen_image_vae.safetensors`

`prefill_tokens` 包含文本和图像 token，不是文本最大长度，也不是要添加的空位数量。
例如原实验的 832×1216、98 个文本 token，总计为 4050；容量 5120 会补 1070 行。
参考图也会占用容量；超过指定容量会明确报错，需增大容量或设为 0。

只补齐 DiT 主干 block 的输入。有效 token 由 attention mask 隔离，保留原始 RoPE，
最后一个 block 后裁回真实总长度，再执行原输出投影。临时张量随本次前向结束释放。

使用 `torch.compile / Inductor`，首次实际采样时编译；不在点击加载时进行虚拟推理。
主干的调制参数、归一化、Attention、MLP 及其权重读取/LoRA 处理沿用原生实现，
编译的是 block 外围的逐元素调制与残差计算，不是整个 DiT 的完整图编译。
加法、乘法分别编译，保留每一步 BF16 输出的舍入边界，避免改变 INT4 输入。
含零时间步参考图的特殊分支沿用原生 block；其 token 补齐和 mask 仍生效。
不要求完整权重常驻显存，不启用 CUDA Graph，不编译文本编码器或 VAE。

开启“LoRA 热插拔”后，照常将本节点连接到原来的 LoRA 加载节点，选择文件与强度即可。
标准 `LoraLoader`、`LoraLoaderModelOnly`，以及调用它们的 rgthree LoRA 加载器均会接入。
此开关可与“启用预编译”分别启用，不增加单独的 LoRA 加载节点。

只对本加载器标记的 K2 模型分流，先用原生映射解析 key，再核对实际层的输入/输出维度及 A/B rank。
没有匹配 key、缺少 A/B 配对或标准线性 LoRA 形状错误时会报错。
支持的标准线性 LoRA 使用原生低秩旁路 `base(x) + strength · alpha/rank · up(down(x))`，
保留实际 rank，不补齐 rank，不展开稠密 AB，不将增量合并或重新量化到底座权重。
串联 LoRA 节点可以叠加；切换文件/强度、设为 0 均沿用原节点的操作方式。
采样切换模型时按相反顺序撤回 hook，并将闲置 adapter 权重移回 CPU。

文本编码器仍按原来的 `strength_clip` 与元数据规则加载。
LoKr、DoRA、带 mid/reshape 等不属于本次标准线性路径的格式继续原生挂载；不能将它们算作旁路热插拔。
该旁路与原生合并后重量化的路径可能产生不同画面。
自定义加载器若绕过 `comfy.sd.load_lora_for_models`，则不在本 hook 覆盖范围；原 LoRA 节点的文件缓存行为保持不变。

这是实验功能。首次运行可能明显更慢；改变输入或 LoRA 可能触发重编译。
补齐不会省掉空位上的 Linear/MLP 计算；更大容量不代表更快。
预编译可能改变浮点融合顺序，尚不承诺与关闭编译逐值相同或带来加速。

## 实现边界与验证

核心实现位于 [`runtime.py`](runtime.py) 与 [`hot_lora.py`](hot_lora.py)，节点入口为 [`__init__.py`](__init__.py)。

本节点检查的是原生映射 key、A/B 配对是否存在，以及实际线性层的维度。它没有接入研究原型中额外的来源／版本／绑定代次有效位，也不预编译独立的低秩 A/B 算子；不要将另一套实验实现的结果直接当作本节点的验收结论。

预编译和热 LoRA 都是实验选项，默认关闭。没有保证任意 ComfyUI 版本、任意 rank、并发请求或推理中途替换都可用，也没有发布稳定加速或无损画质承诺。

CPU 源码检查，不加载 ComfyUI、PyTorch 或模型：

```bash
python scripts/check_source.py
```

可选 GPU 验证脚本，使用已有 ComfyUI Python 运行：

```bash
python tests/verify_runtime.py --comfy-dir "PATH_TO_COMFYUI"
python tests/verify_hot_lora.py --comfy-dir "PATH_TO_COMFYUI"
python tests/verify_real_loader.py --comfy-dir "PATH_TO_COMFYUI"
python tests/verify_real_hot_lora.py --comfy-dir "PATH_TO_COMFYUI" --lora-a "krea2/1999.safetensors" --lora-b "krea2/konya_karasue_krea2_lora_r32_a32_1536-000016.safetensors"
```

前两个脚本使用小型随机模型／合成 LoRA，但仍需 CUDA；后两个需要本机实际权重，覆盖加载、实际 block、切换及还原，不是完整出图性能基准。默认产物写入 `.test-output/`，可用 `--output-dir` 指定目录。模型文件名可通过 `--dit-name`、`--text-encoder-name`、`--vae-name` 指定。所有脚本支持 `--help`。

GitHub CI 只执行 CPU 源码、翻译和 CLI 检查，不代表重跑了 GPU 或实际模型测试。

## 许可证与上游

源码采用 [GNU GPL v3](LICENSE)。Krea2 结构和旁路 API 来自 [ComfyUI](https://github.com/Comfy-Org/ComfyUI)，数学与模型设计来自相应上游；说明见 [NOTICE](NOTICE)。本仓库不分发 ComfyUI 本体、权重或编译产物。
