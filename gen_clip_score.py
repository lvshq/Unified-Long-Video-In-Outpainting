# pip install open_clip_torch torch torchvision decord
import os
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import open_clip
from decord import VideoReader, cpu
from PIL import Image


def frame_indices_from_fps(num_frames: int, fps: float, sample_fps: float) -> np.ndarray:
    """
    Pick indices by sampling at sample_fps. If sample_fps >= fps, returns all frames.
    """
    if sample_fps is not None and (not math.isfinite(sample_fps) or sample_fps <= 0):
        raise ValueError("sample_fps must be finite and positive")
    if sample_fps is None:
        return np.arange(num_frames)

    if sample_fps >= fps:
        return np.arange(num_frames)

    step = fps / sample_fps  # e.g. fps=30, sample_fps=2 => step=15 frames
    idx = np.arange(0, num_frames, step)
    idx = np.round(idx).astype(int)
    idx = np.unique(np.clip(idx, 0, num_frames - 1))
    return idx


@torch.no_grad()
def per_frame_clipscore_to_csv(
    video_path: str,
    prompt: str,
    out_csv: str,
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    sample_fps: float | None = None,  # None => all frames; else sample at this fps
    batch_size: int = 32,
    device: str | None = None,
    mask_path: str | None = None,
):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load CLIP model + preprocess
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    # Encode text once
    text_tokens = tokenizer([prompt]).to(device)
    text_feat = model.encode_text(text_tokens)
    text_feat = F.normalize(text_feat, dim=-1)  # (1, D)

    # Read video
    vr = VideoReader(video_path, ctx=cpu(0))
    num_frames = len(vr)
    fps = float(vr.get_avg_fps()) if hasattr(vr, "get_avg_fps") else None
    if fps is None or math.isnan(fps) or fps <= 0:
        # Fallback if fps is missing: assume 30, but time_sec will be approximate.
        fps = 30.0

    mask_reader, static_mask = None, None
    if mask_path:
        if Path(mask_path).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            with Image.open(mask_path) as image:
                static_mask = np.asarray(image.convert("L")).copy()
        else:
            mask_reader = VideoReader(str(mask_path), ctx=cpu(0))
            if len(mask_reader) != num_frames:
                raise ValueError("Mask and prediction frame counts must match.")

    idx = frame_indices_from_fps(num_frames, fps, sample_fps)

    rows = []
    # Process in batches
    for start in range(0, len(idx), batch_size):
        batch_idx = idx[start : start + batch_size]
        frames = vr.get_batch(batch_idx).asnumpy()  # uint8 RGB

        imgs, valid_indices = [], []
        for fi, fr in zip(batch_idx, frames):
            image = Image.fromarray(fr)
            if mask_path:
                mask = (
                    static_mask
                    if static_mask is not None
                    else mask_reader[int(fi)].asnumpy().mean(axis=-1)
                )
                if mask.shape != fr.shape[:2]:
                    raise ValueError("Mask and prediction dimensions must match.")
                ys, xs = np.where(mask >= 128)
                if len(xs) == 0:
                    continue
                image = image.crop(
                    (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
                )
            imgs.append(preprocess(image).unsqueeze(0))
            valid_indices.append(fi)
        if not imgs:
            continue
        images = torch.cat(imgs, dim=0).to(device)

        # Encode images
        img_feat = model.encode_image(images)
        img_feat = F.normalize(img_feat, dim=-1)  # (B, D)

        # Cosine similarity per frame (dot since normalized)
        cosine = (img_feat @ text_feat.T).squeeze(1)  # (B,)

        # CLIPScore-like scaling: max(100*cosine, 0)
        clipscore = torch.clamp(100.0 * cosine, min=0.0)

        # Save results
        for i, fi in enumerate(valid_indices):
            time_sec = float(fi) / fps
            rows.append(
                {
                    "frame_idx": int(fi),
                    "time_sec": time_sec,
                    "clipscore": float(clipscore[i].item()),
                    "cosine": float(cosine[i].item()),
                }
            )

    if not rows:
        raise ValueError("No frames with a nonempty evaluation region.")
    df = pd.DataFrame(rows).sort_values("frame_idx").reset_index(drop=True)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)

    return df


if __name__ == "__main__":
    import sys
    from scripts.evaluate import main

    main(["clip", *sys.argv[1:]])
