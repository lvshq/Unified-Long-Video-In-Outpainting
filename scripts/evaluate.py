"""Evaluate aligned videos without editing Python source paths."""

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    tasks = parser.add_subparsers(dest="metric", required=True)
    spatial = tasks.add_parser("ssim-lpips", help="Background preservation against the source.")
    spatial.add_argument("--reference", required=True)
    spatial.add_argument("--prediction", required=True)
    spatial.add_argument("--mask", required=True)
    spatial.add_argument("--mask-format", choices=["binary", "gray-overlay"], default="binary")
    spatial.add_argument("--region", choices=["unmasked", "masked"], default="unmasked")
    spatial.add_argument("--lpips-net", choices=["alex", "vgg", "squeeze"], default="alex")
    spatial.add_argument("--output", type=Path, required=True, help="Summary JSON.")
    clip = tasks.add_parser(
        "clip", help="Text similarity; pass a binary mask for edited-region crops."
    )
    clip.add_argument("--video", required=True)
    clip.add_argument("--prompt", required=True)
    clip.add_argument("--mask", help="Optional binary image/video; white is the edited region.")
    clip.add_argument("--model", default="ViT-B-32")
    clip.add_argument("--pretrained", default="openai")
    clip.add_argument("--sample-fps", type=float)
    clip.add_argument("--output", type=Path, required=True, help="Per-frame CSV.")
    temporal = tasks.add_parser("tlpips", help="LPIPS between consecutive output frames.")
    temporal.add_argument("--video", required=True)
    temporal.add_argument("--lpips-net", choices=["alex", "vgg", "squeeze"], default="vgg")
    temporal.add_argument("--sample-fps", type=float)
    temporal.add_argument("--output", type=Path, required=True, help="Per-pair CSV.")
    for subparser in [spatial, clip, temporal]:
        subparser.add_argument("--device", help="Defaults to cuda if available, otherwise cpu.")
    args = parser.parse_args(argv)
    if getattr(args, "sample_fps", None) is not None and args.sample_fps <= 0:
        parser.error("sample-fps must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.metric == "ssim-lpips":
        from gen_SSIM_LPIPS import evaluate_video_with_mask

        result = evaluate_video_with_mask(
            args.reference,
            args.prediction,
            args.mask,
            region=args.region,
            mask_format=args.mask_format,
            lpips_net=args.lpips_net,
            device=args.device,
        )
        if result["num_valid"] == 0:
            raise ValueError("No frames contain the requested evaluation region.")
        summary = {key: value for key, value in result.items() if not key.endswith("_per_frame")}
        summary.update(region=args.region, mask_format=args.mask_format, lpips_net=args.lpips_net)
        args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    elif args.metric == "clip":
        from gen_clip_score import per_frame_clipscore_to_csv

        frames = per_frame_clipscore_to_csv(
            args.video,
            args.prompt,
            str(args.output),
            model_name=args.model,
            pretrained=args.pretrained,
            sample_fps=args.sample_fps,
            device=args.device,
            mask_path=args.mask,
        )
        summary = {
            "clip_mean": float(frames.clipscore.mean()),
            "num_frames": len(frames),
            "region": "edit_bbox" if args.mask else "full_frame",
        }
    else:
        from gen_tLPIPS import tlpips_for_video

        score, frames = tlpips_for_video(
            args.video,
            net=args.lpips_net,
            sample_fps=args.sample_fps,
            device=args.device,
            out_csv=str(args.output),
        )
        summary = {"tlpips_mean": score, "num_pairs": len(frames)}
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
