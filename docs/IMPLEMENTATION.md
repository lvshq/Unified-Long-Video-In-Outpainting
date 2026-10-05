# Implementation and release notes

This source release preserves the vendored DiffSynth backend and the supplied
masked-video training objective, and adds configurable entry points and
correctness fixes. It is not a claim of bit-for-bit reproduction of paper results.

## Paper and code distinctions

| Topic | Supplied code and this release |
| --- | --- |
| High-order solver | The paper describes endpoint-averaged Heun updates. The supplied `step_mid` / `step_solver` pair implements explicit midpoint RK2. The CLI calls it `midpoint` and preserves that numerical method. |
| Training loss | The paper writes the objective on reconstructed RGB frames. Training here weights squared latent flow-prediction errors, then applies the scheduler's timestep weighting. Region errors are averaged over the full tensor, not divided by each region's area. |
| Interior masks | The paper describes 1–4 rectangles. This code samples 1–3 rectangles, retaining the supplied training distribution. |
| Temporal window | The original convention is `floor(window_frames / 4)` latent frames. An 81-frame budget uses 20 latent frames with stride `floor(20/3)=6`. |
| Memory | DiT attention operates within temporal windows. VAE inputs/outputs and global latent/accumulation buffers still scale with sequence length. |
| Assets | Complete InpaintBench data and trained editing adapters were absent from the source folder. `configs/example_prompts.json` is a subset of available prompts, not a complete benchmark manifest. |

## Release fixes

- Include clips shorter than a denoising window and reject strides that leave gaps.
- Complete the final denoising step to zero noise. Previously the loop omitted the
  last scheduler timestep; this fix can change outputs relative to old experiments.
- Apply Hamming weights along time and normalize without biasing nonzero weight
  sums. Accumulate in float32 with a compact temporal weight buffer.
- Simplify the midpoint expression algebraically to avoid division by zero-length
  intervals. This does not convert the method into Heun.
- Validate canvas sizes, mask alignment, frame counts, and inference options.
  Explicit binary masks are applied before VAE encoding and again in latent space;
  the original gray-overlay and central-latent-hole modes remain available.
- Reject unsupported ControlNet/VACE/image conditioning and TeaCache options in
  the co-denoising path, where they were previously not applied correctly. Use
  the inherited non-co-denoising API for those other model variants.
- Replace local video/checkpoint dictionaries with configurable arguments and
  retain available prompt text in JSON.
- Remove pasted metric logs, expose evaluation arguments, add mask crops for CLIP,
  require aligned video inputs, and handle variable-size/empty spatial crops.
- Add a training seed, missing-cache errors, clean notebook outputs, dependency
  lists, and CPU regression checks.
- Replace the upstream PyPI upload workflow with source validation.

Notebook source cells are retained as an exploratory record. They reference local
assets and may assume earlier interactive state. Prefer the documented scripts.

## Validation scope

The CPU suite checks temporal coverage, Hamming blending, short clips, midpoint
and Euler integration through the final step, input validation, and frame padding.
It does not load Wan or learned metric weights. GPU memory, model quality, training
convergence, and paper metrics require a CUDA environment and the relevant assets.
