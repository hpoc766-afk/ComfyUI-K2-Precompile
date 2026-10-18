# SPDX-License-Identifier: GPL-3.0-only
import folder_paths
import nodes
from comfy.ldm.krea2.model import SingleStreamDiT

from .hot_lora import enable_hot_lora
from .runtime import patch_krea2
from .full_precompile import patch_full_models
from .token_counter import K2TokenCounter


class K2PrecompileLoader:
    @classmethod
    def INPUT_TYPES(cls):
        dit_files = folder_paths.get_filename_list("diffusion_models")
        text_files = folder_paths.get_filename_list("text_encoders")
        vae_files = nodes.VAELoader.vae_list(nodes.VAELoader)
        dit_default = "Krea2-Turbo-int4-ConvRot-ComfyUI.safetensors"
        text_default = "qwen3-vl-4b-heretic_fp8_e4m3fn.safetensors"
        vae_default = "qwen_image_vae.safetensors"
        return {
            "required": {
                "dit_name": (dit_files, {"default": dit_default} if dit_default in dit_files else {}),
                "text_encoder_name": (text_files, {"default": text_default} if text_default in text_files else {}),
                "vae_name": (vae_files, {"default": vae_default} if vae_default in vae_files else {}),
                "precompile": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "逐元素模式：首次采样编译调制/残差；仅在全部预编译关闭时生效。",
                }),
                "prefill_tokens": ("INT", {
                    "default": 5120, "min": 0, "max": 65536, "step": 1,
                    "tooltip": "仅逐元素模式使用的文本+图像token容量；0不补齐。全部预编译忽略此项。",
                }),
            },
            "optional": {
                "hot_lora": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "验证 K2 权重名称及适配器维度后，让后续 LoRA 加载节点使用可逆低秩旁路热插拔。",
                }),
                "full_precompile": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "严格编译TE Transformer、DiT前向和VAE编解码；需要兼容的静态权重驻留，动态VBAR不支持。优先于逐元素模式，不支持时报错。",
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("模型", "文本编码器", "变分自编码器")
    FUNCTION = "load"
    CATEGORY = "K2/模型加载"
    DESCRIPTION = "加载K2模型，可选择逐元素预编译、TE/DiT/VAE完整图编译与LoRA热插拔。"
    EXPERIMENTAL = True

    def load(self, dit_name, text_encoder_name, vae_name, precompile, prefill_tokens, hot_lora=False, full_precompile=False):
        model, = nodes.UNETLoader().load_unet(dit_name, "default")
        if not isinstance(model.get_model_object("diffusion_model"), SingleStreamDiT):
            raise ValueError("DiT 文件不是 Krea2 模型，请选择 K2 主干权重。")
        clip, = nodes.CLIPLoader().load_clip(text_encoder_name, type="krea2")
        vae, = nodes.VAELoader().load_vae(vae_name)
        if full_precompile:
            model, clip, vae = patch_full_models(model, clip, vae)
        elif precompile:
            model = patch_krea2(model, prefill_tokens, compile_blocks=True)
        if hot_lora:
            model = enable_hot_lora(model)
        return model, clip, vae


NODE_CLASS_MAPPINGS = {"K2PrecompileLoader": K2PrecompileLoader, "K2TokenCounter": K2TokenCounter}
NODE_DISPLAY_NAME_MAPPINGS = {"K2PrecompileLoader": "K2 模型加载器", "K2TokenCounter": "K2 Token 计数器"}
WEB_DIRECTORY = "./web"
