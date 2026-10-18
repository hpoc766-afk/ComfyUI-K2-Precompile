# SPDX-License-Identifier: GPL-3.0-only
"""CPU node contracts; real-tokenizer/source matching is validated separately."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class Tokenizer:
    def __init__(self):
        self.calls = []

    def tokenize_with_weights(self, text, **kwargs):
        self.calls.append((text, kwargs))
        size = {"positive": 93, "negative": 19, "long negative": 200, "": 0}.get(text, 1)
        ids = list(range(1000, 1000 + size))
        if text == "image embedding":
            ids = [{"type": "image"}]
        if not kwargs.get("skip_template"):
            prefix = [151644] + [700] * 28 + [151645, 198, 151644, 872, 198]
            ids = prefix + ids + [151645, 198, 151644, 77091, 198]
        return {"qwen3vl_4b": [[(token, 1.) for token in ids]]}


def load_counter():
    spec = importlib.util.spec_from_file_location("_token_counter_contract", ROOT / "token_counter.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"comfy.text_encoders.krea2": SimpleNamespace(Krea2Tokenizer=Tokenizer)}):
        spec.loader.exec_module(module)
    return module


class TokenCounterTests(unittest.TestCase):
    def setUp(self):
        self.module = load_counter()
        self.counter = self.module.K2TokenCounter()
        self.latent = {"samples": SimpleNamespace(shape=(1, 16, 1, 152, 104))}

    def test_prompt_and_conditioning_are_distinct(self):
        result = self.counter.count("positive", self.latent, "negative")
        self.assertEqual(result["result"][:6], (93, 19, 3952, 98, 4050, 4096))
        self.assertIn("TE 输入 132", result["result"][6])
        self.assertEqual(result["ui"]["text"], [result["result"][6]])
        self.assertEqual(self.counter.tokenizer.calls, [
            ("positive", {"skip_template": True, "min_length": 0}), ("positive", {}),
            ("negative", {"skip_template": True, "min_length": 0}), ("negative", {}),
        ])

    def test_negative_uses_longest_sequence_not_sum(self):
        result = self.counter.count("positive", self.latent, "long negative")["result"]
        self.assertEqual(result[:6], (93, 200, 3952, 205, 4157, 4160))

    def test_odd_latent_is_padded_to_patch_size(self):
        latent = {"samples": SimpleNamespace(shape=(1, 16, 1, 167, 114))}
        result = self.counter.count("positive", latent)["result"]
        self.assertEqual(result[2:6], (4788, 98, 4886, 4928))

    def test_batch_frames_channels_do_not_multiply_sequence_length(self):
        latent = {"samples": SimpleNamespace(shape=(2, 16, 3, 152, 104))}
        result = self.counter.count("positive", latent)["result"]
        self.assertEqual(result[2:6], (3952, 98, 4050, 4096))
        self.assertIn("batch=2，frames=3", result[6])

    def test_4d_latent_and_exact_alignment(self):
        latent = {"samples": SimpleNamespace(shape=(1, 16, 152, 104))}
        result = self.counter.count("positive", latent, alignment=1)["result"]
        self.assertEqual(result[4:6], (4050, 4050))

    def test_empty_prompt_keeps_only_conditioning_suffix(self):
        result = self.counter.count("", self.latent)["result"]
        self.assertEqual(result[:6], (0, 0, 3952, 5, 3957, 3968))

    def test_invalid_shape_and_alignment_fail_clearly(self):
        for shape in [(16, 152, 104), (1, 16, 0, 104)]:
            with self.assertRaisesRegex(ValueError, "latent"):
                self.counter.count("positive", {"samples": SimpleNamespace(shape=shape)})
        with self.assertRaisesRegex(ValueError, "对齐"):
            self.counter.count("positive", self.latent, alignment=0)

    def test_embeddings_are_not_silently_counted_as_one_token(self):
        for text in ["embedding:style", "a cat embedding:style"]:
            with self.assertRaisesRegex(ValueError, "embedding"):
                self.counter.count(text, self.latent)
        self.assertEqual(self.counter.tokenizer.calls, [])
        with self.assertRaisesRegex(ValueError, "embedding"):
            self.counter.count("image embedding", self.latent)

    def test_standalone_outputs_only_counts_and_report(self):
        spec = self.counter.INPUT_TYPES()
        self.assertEqual(set(spec["required"]), {"positive", "latent"})
        self.assertEqual(set(spec["optional"]), {"negative", "alignment"})
        self.assertEqual(self.counter.RETURN_TYPES, ("INT",) * 6 + ("STRING",))
        self.assertTrue(self.counter.OUTPUT_NODE)
        sample = self.latent["samples"]
        self.counter.count("positive", self.latent)
        self.assertIs(self.latent["samples"], sample)


if __name__ == "__main__":
    unittest.main()
