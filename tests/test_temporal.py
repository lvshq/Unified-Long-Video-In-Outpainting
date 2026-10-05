"""CPU regression tests for temporal coverage, blending, and the final solver step."""

import importlib.util
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    # Load lightweight modules without importing unrelated GPU pipelines.
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


utils = load_module("wan_utils", "diffsynth/pipelines/wan_video_utils.py")
scheduler_module = load_module("flow_match", "diffsynth/schedulers/flow_match.py")


class TemporalTests(unittest.TestCase):
    def test_windows_cover_short_exact_and_long_clips(self):
        for length in [1, 3, 19, 20, 21, 42, 201]:
            with self.subTest(length=length):
                views = utils.get_views(length, 20, 6)
                covered = {frame for start, end in views for frame in range(start, end)}
                self.assertEqual(covered, set(range(length)))
                self.assertEqual(len(views), len(set(views)))
                self.assertTrue(all(0 <= a < b <= length for a, b in views))

    def test_invalid_windows_raise(self):
        for args in [(0, 20, 6), (3, 0, 1), (10, 4, 0), (10, 4, 5)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                utils.get_views(*args)

    def test_hamming_blending_preserves_values_and_spatial_axes(self):
        torch.manual_seed(1)
        for length in [1, 3, 20, 43]:
            # Width != window length catches slicing the spatial axis by mistake.
            latents = torch.randn(2, 4, length, 3, 7)
            result = utils.blend_windows(latents, utils.get_views(length, 20, 6), lambda x: x)
            torch.testing.assert_close(result, latents)

    def test_uncovered_frames_fail_loudly(self):
        with self.assertRaises(ValueError):
            utils.blend_windows(torch.ones(1, 1, 5, 1, 1), [(0, 2)], lambda x: x)

    def test_constant_velocity_reaches_zero_noise_for_both_solvers(self):
        for high_order in [False, True]:
            for steps in [1, 2, 5]:
                with self.subTest(high_order=high_order, steps=steps):
                    scheduler = scheduler_module.FlowMatchScheduler(
                        num_inference_steps=steps, shift=1, sigma_min=0, extra_one_step=True
                    )
                    latents = torch.ones(1, 1, 7, 2, 3)
                    calls = []

                    def predict(x, timestep):
                        calls.append(float(timestep))
                        return torch.ones_like(x)

                    for index, timestep in enumerate(scheduler.timesteps):
                        latents = utils.blend_windows(
                            latents,
                            utils.get_views(7, 4, 2),
                            lambda x: utils.denoise_window(
                                scheduler, predict, x, timestep, index, high_order
                            ),
                        )
                    torch.testing.assert_close(
                        latents, torch.zeros_like(latents), atol=1e-6, rtol=0
                    )
                    self.assertEqual(len(calls), steps * 3 * (2 if high_order else 1))

    def test_midpoint_matches_second_order_solution_for_linear_velocity(self):
        scheduler = scheduler_module.FlowMatchScheduler(
            num_inference_steps=1, shift=1, sigma_min=0, extra_one_step=True
        )
        result = utils.denoise_window(
            scheduler, lambda x, t: x, torch.ones(1), scheduler.timesteps[0], 0
        )
        torch.testing.assert_close(result, torch.tensor([0.5]))

    def test_zero_length_midpoint_step_is_finite(self):
        scheduler = scheduler_module.FlowMatchScheduler(num_inference_steps=2, sigma_min=0)
        sample = torch.ones(1)
        result = scheduler.step_solver(sample, sample, scheduler.timesteps[-1], sample)
        torch.testing.assert_close(result, sample)


if __name__ == "__main__":
    unittest.main()
