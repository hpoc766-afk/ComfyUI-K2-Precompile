# SPDX-License-Identifier: GPL-3.0-only
"""Token counts for ordinary Krea2 text-to-image inputs; no model inference."""
from numbers import Integral
import re

from comfy.text_encoders.krea2 import Krea2Tokenizer


_TOKEN_KEY = "qwen3vl_4b"


def _token_ids(tokenizer, text, **kwargs):
    batches = tokenizer.tokenize_with_weights(text, **kwargs)[_TOKEN_KEY]
    if len(batches) != 1:
        raise ValueError("K2 Token 计数器仅支持单段纯文本分词。")
    ids = [pair[0] for pair in batches[0]]
    if not all(isinstance(token, Integral) for token in ids):
        raise ValueError("K2 Token 计数器不支持图片或多向量 embedding。")
    return ids


def text_token_counts(tokenizer, text):
    if re.search(r"(^|\s)embedding:", text):
        raise ValueError("K2 Token 计数器仅支持纯文本；embedding: 的实际展开长度不在统计范围内。")
    prompt_ids = _token_ids(tokenizer, text, skip_template=True, min_length=0)
    full_ids = _token_ids(tokenizer, text)
    # Match Krea2TEModel.encode_token_weights: strip through the second
    # im_start/user/newline prefix, retaining the assistant/end suffix.
    offset = -1
    starts = 0
    for index, token in enumerate(full_ids):
        if token == 151644 and starts < 2:
            offset = index
            starts += 1
    if len(full_ids) > offset + 3:
        if full_ids[offset + 1] == 872 and full_ids[offset + 2] == 198:
            offset += 3
    return len(prompt_ids), len(full_ids), len(full_ids[offset:])


class K2TokenCounter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("STRING", {"multiline": True, "default": "", "tooltip": "与正向文本编码节点使用相同提示词。"}),
                "latent": ("LATENT", {"tooltip": "读取实际 latent 形状，不编码、不采样、不移动张量。"}),
            },
            "optional": {
                "negative": ("STRING", {"multiline": True, "default": "", "tooltip": "与负向文本编码节点使用相同提示词；文本容量取正负序列中较长者。"}),
                "alignment": ("INT", {"default": 64, "min": 1, "max": 4096, "step": 1, "tooltip": "将联合 token 数向上对齐到此倍数；设为 1 返回精确联合长度。"}),
            },
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("正向提示词 token", "负向提示词 token", "latent token", "DiT 文本 token", "联合 token", "预填充参考 token", "统计说明")
    FUNCTION = "count"
    CATEGORY = "K2/工具"
    OUTPUT_NODE = True
    DESCRIPTION = "使用 Krea2 本地 tokenizer 统计提示词与实际文本序列，按 latent 的 2×2 patch 计算预填充参考。不加载模型权重，也不改预填充设置。"

    def __init__(self):
        self.tokenizer = Krea2Tokenizer()

    def count(self, positive, latent, negative="", alignment=64):
        if alignment < 1:
            raise ValueError("token 对齐倍数必须大于零。")
        shape = tuple(latent["samples"].shape)
        if len(shape) not in (4, 5) or any(size <= 0 for size in shape):
            raise ValueError("需要有效的 K2 latent：[B,C,H,W] 或 [B,C,T,H,W]。")
        pos_raw, pos_te, pos_text = text_token_counts(self.tokenizer, positive)
        neg_raw, neg_te, neg_text = text_token_counts(self.tokenizer, negative)
        height, width = shape[-2:]
        rows, columns = (height + 1) // 2, (width + 1) // 2
        latent_tokens = rows * columns
        text_tokens = max(pos_text, neg_text)
        total = latent_tokens + text_tokens
        prefill = ((total + alignment - 1) // alignment) * alignment
        frames = shape[2] if len(shape) == 5 else 1
        report = (
            f"正向：提示词 {pos_raw}，TE 输入 {pos_te}，DiT 文本 {pos_text} token\n"
            f"负向：提示词 {neg_raw}，TE 输入 {neg_te}，DiT 文本 {neg_text} token\n"
            f"latent：{height}×{width} → {rows}×{columns} = {latent_tokens} token\n"
            f"文本容量：max({pos_text}, {neg_text}) = {text_tokens}\n"
            f"联合长度：{latent_tokens} + {text_tokens} = {total}\n"
            f"预填充参考：{prefill}（向上对齐 {alignment}）\n"
            f"每条序列计数；batch={shape[0]}，frames={frames}，不乘入序列长度。"
        )
        return {"ui": {"text": [report]}, "result": (pos_raw, neg_raw, latent_tokens, text_tokens, total, prefill, report)}
