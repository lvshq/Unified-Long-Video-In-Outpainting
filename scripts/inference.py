"""Run mask-guided inpainting or centered outpainting with a local Wan + LoRA model."""

import argparse
import json
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input video.")
    parser.add_argument("--output", type=Path, required=True, help="Output MP4.")
    parser.add_argument("--task", choices=["inpainting", "outpainting"], required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="blurry, low quality, subtitles, distorted")
    parser.add_argument("--model-dir", type=Path, required=True, help="Wan2.1-T2V-14B directory.")
    parser.add_argument("--lora", type=Path, required=True, help="Trained LoRA checkpoint.")
    parser.add_argument("--lora-scale", type=float, default=0.7)
    masks = parser.add_mutually_exclusive_group()
    masks.add_argument(
        "--mask", type=Path, help="Binary mask image/video: white edits, black keeps."
    )
    masks.add_argument(
        "--gray-overlay", action="store_true", help="Input already has gray (128±2) holes."
    )
    masks.add_argument(
        "--center-mask", action="store_true", help="Use the legacy central latent hole."
    )
    parser.add_argument(
        "--height", type=int, help="Output height; defaults to input height (2x for outpainting)."
    )
    parser.add_argument(
        "--width", type=int, help="Output width; defaults to input width (2x for outpainting)."
    )
    parser.add_argument("--num-frames", type=int, help="Limit to the first N frames; default: all.")
    parser.add_argument(
        "--fps", type=float, help="Output FPS; defaults to source FPS (no temporal resampling)."
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--strength", type=float, default=0.7)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument(
        "--window-frames", type=int, default=81, help="Denoising window budget in frames."
    )
    parser.add_argument(
        "--overlap-factor", type=int, default=3, help="Window/stride ratio; 0 uses one full window."
    )
    parser.add_argument("--solver", choices=["midpoint", "euler"], default="midpoint")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--persistent-dit-params",
        type=int,
        help="GPU-resident DiT parameter budget for offloading.",
    )
    args = parser.parse_args(argv)
    if args.steps < 1 or args.window_frames < 4 or args.overlap_factor < 0:
        parser.error("steps must be positive, window-frames >= 4, and overlap-factor >= 0")
    if not 0 < args.strength <= 1:
        parser.error("strength must be in (0, 1]")
    if args.num_frames is not None and args.num_frames < 1:
        parser.error("num-frames must be positive")
    if args.fps is not None and args.fps <= 0:
        parser.error("fps must be positive")
    if args.task == "inpainting" and not (args.mask or args.gray_overlay or args.center_mask):
        parser.error("inpainting requires --mask, --gray-overlay, or --center-mask")
    if args.task == "outpainting" and (args.mask or args.gray_overlay or args.center_mask):
        parser.error("mask options are only used for inpainting")
    if args.output.resolve() == args.input.resolve() or args.output.exists():
        parser.error("output must be a new file, distinct from the input")
    return args


def model_files(model_dir):
    """Resolve all six DiT shards and the two required encoder files."""
    shards = sorted(model_dir.glob("diffusion_pytorch_model-*-of-*.safetensors"))
    if len(shards) != 6:
        raise ValueError(
            f"Expected 6 Wan2.1-T2V-14B DiT shards in {model_dir}; found {len(shards)}."
        )
    encoders = [model_dir / "models_t5_umt5-xxl-enc-bf16.pth", model_dir / "Wan2.1_VAE.pth"]
    for path in encoders + [model_dir / "google/umt5-xxl/tokenizer_config.json"]:
        if not path.is_file():
            raise FileNotFoundError(path)
    return [str(path) for path in shards], [str(path) for path in encoders]


def load_inputs(args):
    """Load aligned frames, repeat the last frame to 4k+1, and retain output length."""
    import imageio.v2 as imageio
    from PIL import Image

    with imageio.get_reader(str(args.input)) as reader:
        metadata = reader.get_meta_data()
        frames = []
        for index, frame in enumerate(reader):
            if args.num_frames is not None and index >= args.num_frames:
                break
            frames.append(Image.fromarray(frame).convert("RGB"))
    if not frames:
        raise ValueError("The input video has no frames.")
    if args.num_frames is not None and len(frames) != args.num_frames:
        raise ValueError("num-frames exceeds the source video length.")
    source_width, source_height = frames[0].size
    if source_height % 16 or source_width % 16:
        raise ValueError("Input width and height must be multiples of 16; resize before inference.")
    if any(frame.size != frames[0].size for frame in frames):
        raise ValueError("Every input frame must have the same dimensions.")
    multiplier = 2 if args.task == "outpainting" else 1
    height = args.height or source_height * multiplier
    width = args.width or source_width * multiplier
    if height <= 0 or width <= 0 or height % 16 or width % 16:
        raise ValueError("Output width and height must be positive multiples of 16.")
    if args.task == "inpainting" and (width, height) != frames[0].size:
        raise ValueError("Inpainting output dimensions must match the input.")
    if args.task == "outpainting" and (height < source_height or width < source_width):
        raise ValueError("Outpainting cannot shrink the input canvas.")
    masks = None
    if args.mask:
        if args.mask.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            with Image.open(args.mask) as mask:
                masks = [mask.convert("L").copy() for _ in frames]
        else:
            with imageio.get_reader(str(args.mask)) as reader:
                masks = [Image.fromarray(frame).convert("L") for frame in reader]
            if args.num_frames is not None:
                masks = masks[: args.num_frames]
        if len(masks) != len(frames) or any(mask.size != frames[0].size for mask in masks):
            raise ValueError("Mask video length and mask dimensions must match the input.")
    original_length = len(frames)
    while len(frames) % 4 != 1:
        frames.append(frames[-1].copy())
        if masks is not None:
            masks.append(masks[-1].copy())
    fps = args.fps or metadata.get("fps")
    if fps is None or not 0 < fps < float("inf"):
        raise ValueError("Cannot read source FPS; pass --fps explicitly.")
    return frames, masks, original_length, height, width, fps


def main(argv=None):
    args = parse_args(argv)
    for path in [args.input, args.lora] + ([args.mask] if args.mask else []):
        if not path.is_file():
            raise FileNotFoundError(path)
    shards, encoders = model_files(args.model_dir)
    frames, masks, length, height, width, fps = load_inputs(args)

    import torch
    from diffsynth import ModelManager, WanVideoPipeline, save_video, load_state_dict
    from diffsynth.models.lora import GeneralLoRAFromPeft

    if not torch.cuda.is_available():
        raise RuntimeError("Wan2.1-T2V-14B inference requires a CUDA GPU.")
    manager = ModelManager(device="cpu", torch_dtype=torch.bfloat16)
    manager.load_models([shards, *encoders], torch_dtype=torch.bfloat16)
    dit = manager.fetch_model("wan_video_dit")
    if dit is None:
        raise ValueError("The checkpoint shards were not recognized as a Wan text-to-video model.")
    adapter = load_state_dict(str(args.lora))
    loader = GeneralLoRAFromPeft()
    if not loader.get_name_dict(adapter) or loader.match(dit, adapter) is None:
        raise ValueError("Expected a Wan LoRA adapter saved by the supplied training script.")
    manager.load_lora(str(args.lora), state_dict=adapter, lora_alpha=args.lora_scale)
    pipe = WanVideoPipeline.from_model_manager(manager, torch_dtype=torch.bfloat16, device="cuda")
    pipe.enable_vram_management(num_persistent_param_in_dit=args.persistent_dit_params)
    output = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        input_video=frames,
        input_mask=masks,
        based_on_input_mask=args.gray_overlay,
        inpaint_video=args.task == "inpainting",
        outpaint_video=args.task == "outpainting",
        inpaint_padding=args.center_mask,
        height=height,
        width=width,
        num_frames=len(frames),
        denoising_strength=args.strength,
        num_inference_steps=args.steps,
        cfg_scale=args.cfg_scale,
        denoising_batch_frames=args.window_frames,
        overlap_factor=args.overlap_factor,
        use_high_order_solver=args.solver == "midpoint",
        seed=args.seed,
        tiled=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_video(output[:length], str(args.output), fps=fps)
    config = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    config.update(
        output_frames=length, encoded_frames=len(frames), height=height, width=width, fps=fps
    )
    args.output.with_suffix(".json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Saved {length} frames to {args.output}")


if __name__ == "__main__":
    main()
