"""Validate user inputs and temporal padding without model weights or a GPU."""

import argparse
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from scripts.inference import load_inputs, model_files, parse_args


class Reader:
    def __init__(self, frames):
        self.frames = frames

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def __iter__(self):
        return iter(self.frames)

    def get_meta_data(self):
        return {"fps": 12}


class InferenceTests(unittest.TestCase):
    def args(self, **kwargs):
        values = dict(
            input=Path("input.mp4"),
            task="inpainting",
            num_frames=None,
            fps=None,
            height=None,
            width=None,
            mask=None,
        )
        values.update(kwargs)
        return argparse.Namespace(**values)

    def test_padding_preserves_original_length_and_mask_alignment(self):
        with tempfile.TemporaryDirectory() as folder:
            mask = Path(folder) / "mask.png"
            Image.new("L", (32, 16), 255).save(mask)
            frames = [np.full((16, 32, 3), index, np.uint8) for index in range(6)]
            with patch("imageio.v2.get_reader", return_value=Reader(frames)):
                video, masks, length, height, width, fps = load_inputs(self.args(mask=mask))
            self.assertEqual(
                (len(video), len(masks), length, height, width, fps), (9, 9, 6, 16, 32, 12)
            )
            np.testing.assert_array_equal(video[-1], frames[-1])

    def test_outpainting_defaults_to_double_canvas(self):
        with patch("imageio.v2.get_reader", return_value=Reader([np.zeros((16, 32, 3), np.uint8)])):
            _, _, _, height, width, _ = load_inputs(self.args(task="outpainting"))
        self.assertEqual((height, width), (32, 64))

    def test_rejects_invalid_dimensions_and_excess_frame_requests(self):
        for changes in [dict(height=18), dict(num_frames=3), dict(width=48)]:
            with self.subTest(changes=changes):
                with patch(
                    "imageio.v2.get_reader", return_value=Reader([np.zeros((16, 32, 3), np.uint8)])
                ):
                    with self.assertRaises(ValueError):
                        load_inputs(self.args(**changes))

    def test_inpainting_requires_an_explicit_mask_mode(self):
        args = [
            "--input",
            "in.mp4",
            "--output",
            "out.mp4",
            "--task",
            "inpainting",
            "--prompt",
            "A tree",
            "--model-dir",
            "models",
            "--lora",
            "model.ckpt",
        ]
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(args)

    def test_model_validation_requires_encoders_and_all_shards(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaises(ValueError):
                model_files(root)
            for index in range(1, 7):
                (root / f"diffusion_pytorch_model-{index:05d}-of-00006.safetensors").touch()
            with self.assertRaises(FileNotFoundError):
                model_files(root)
            for name in [
                "models_t5_umt5-xxl-enc-bf16.pth",
                "Wan2.1_VAE.pth",
                "google/umt5-xxl/tokenizer_config.json",
            ]:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            shards, encoders = model_files(root)
            self.assertEqual((len(shards), len(encoders)), (6, 2))


if __name__ == "__main__":
    unittest.main()
