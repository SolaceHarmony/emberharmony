#!/usr/bin/env python
"""Config resolver checks for the realtime Moshi parity harness."""

from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dump_moshi_realtime import checkpoint_floating_dtype, resolve_checkpoint


def write_safetensor_header(path: Path, header: dict) -> None:
    data = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(data)) + data)


class MoshiRealtimeConfigTest(unittest.TestCase):
    def resolve(self, config: dict):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(json.dumps(config))
            info = resolve_checkpoint(str(root))
            return {
                "root": root,
                "moshi": info.moshi_weights,
                "mimi": info.mimi_weights,
                "tokenizer": info.tokenizer,
                "model_type": info.model_type,
                "lm_gen_config": info.lm_gen_config,
                "lm_config": info.lm_config,
            }

    def test_nested_lm_config_matches_rust_loader(self):
        result = self.resolve(
            {
                "lm_config": {
                    "moshi_name": "nested-moshi.safetensors",
                    "mimi_name": "nested-mimi.safetensors",
                    "tokenizer_name": "nested-tokenizer.model",
                    "model_type": "moshi",
                    "lm_gen_config": {"use_sampling": False, "top_k": 17},
                    "dim": 4096,
                }
            }
        )

        self.assertEqual(result["moshi"], result["root"] / "nested-moshi.safetensors")
        self.assertEqual(result["mimi"], result["root"] / "nested-mimi.safetensors")
        self.assertEqual(result["tokenizer"], result["root"] / "nested-tokenizer.model")
        self.assertEqual(result["model_type"], "moshi")
        self.assertEqual(result["lm_gen_config"], {"use_sampling": False, "top_k": 17})
        self.assertEqual(result["lm_config"], {"dim": 4096})

    def test_root_strings_override_nested_strings(self):
        result = self.resolve(
            {
                "moshi_name": "root-moshi.safetensors",
                "mimi_name": "root-mimi.safetensors",
                "tokenizer_name": "root-tokenizer.model",
                "model_type": "moshi",
                "lm_config": {
                    "moshi_name": "nested-moshi.safetensors",
                    "mimi_name": "nested-mimi.safetensors",
                    "tokenizer_name": "nested-tokenizer.model",
                    "model_type": "hibiki",
                    "dim": 4096,
                },
            }
        )

        self.assertEqual(result["moshi"], result["root"] / "root-moshi.safetensors")
        self.assertEqual(result["mimi"], result["root"] / "root-mimi.safetensors")
        self.assertEqual(result["tokenizer"], result["root"] / "root-tokenizer.model")
        self.assertEqual(result["model_type"], "moshi")
        self.assertEqual(result["lm_config"], {"dim": 4096})

    def test_null_root_string_falls_back_to_nested_string(self):
        result = self.resolve(
            {
                "moshi_name": None,
                "lm_config": {
                    "moshi_name": "nested-moshi.safetensors",
                    "dim": 4096,
                },
            }
        )

        self.assertEqual(result["moshi"], result["root"] / "nested-moshi.safetensors")
        self.assertEqual(result["lm_config"], {"dim": 4096})

    def test_null_root_generation_config_uses_defaults(self):
        result = self.resolve(
            {
                "lm_gen_config": None,
                "lm_config": {
                    "lm_gen_config": {"use_sampling": False, "top_k": 17},
                    "dim": 4096,
                },
            }
        )

        self.assertEqual(result["lm_gen_config"], {})
        self.assertEqual(result["lm_config"], {"dim": 4096})

    def test_auto_dtype_uses_safetensor_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bf16 = root / "bf16.safetensors"
            f32 = root / "f32.safetensors"
            write_safetensor_header(
                bf16,
                {"a": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}},
            )
            write_safetensor_header(
                f32,
                {"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
            )

            self.assertEqual(checkpoint_floating_dtype(bf16), "bfloat16")
            self.assertEqual(checkpoint_floating_dtype(f32), "float32")

    def test_auto_dtype_rejects_mixed_safetensor_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mixed.safetensors"
            write_safetensor_header(
                path,
                {
                    "a": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]},
                    "b": {"dtype": "F32", "shape": [1], "data_offsets": [2, 6]},
                },
            )

            with self.assertRaises(SystemExit):
                checkpoint_floating_dtype(path)


if __name__ == "__main__":
    unittest.main()
