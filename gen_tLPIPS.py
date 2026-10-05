"""Consecutive-frame LPIPS evaluation (lower is smoother, without motion compensation)."""

import os
import math
import numpy as np
import pandas as pd
import torch
import lpips
from decord import VideoReader, cpu


def frame_indices_from_fps(num_frames: int, fps: float, sample_fps: float | None) -> np.ndarray:
    if sample_fps is not None and (not math.isfinite(sample_fps) or sample_fps <= 0):
        raise ValueError("sample_fps must be finite and positive")
    if sample_fps is None:
        return np.arange(num_frames)
    if sample_fps >= fps:
        return np.arange(num_frames)
    step = fps / sample_fps
    idx = np.arange(0, num_frames, step)
    idx = np.round(idx).astype(int)
    idx = np.unique(np.clip(idx, 0, num_frames - 1))
    return idx


def _frames_uint8_to_lpips_tensor(frames_uint8: np.ndarray, device: str) -> torch.Tensor:
    """
    frames_uint8: (B, H, W, 3) in RGB uint8
    returns: (B, 3, H, W) float in [-1, 1] (what lpips expects)
    """
    x = torch.from_numpy(frames_uint8).to(device=device, dtype=torch.float32)  # (B,H,W,3)
    x = x.permute(0, 3, 1, 2).contiguous() / 255.0  # -> [0,1]
    x = x * 2.0 - 1.0  # -> [-1,1]
    return x


@torch.no_grad()
def tlpips_for_video(
    video_path: str,
    net: str = "vgg",  # "vgg" (common), "alex", "squeeze"
    sample_fps: float | None = None,  # None => all frames
    batch_size_pairs: int = 16,  # pairs per batch (uses 2*batch frames)
    device: str | None = None,
    out_csv: str | None = None,  # if set, saves per-pair LPIPS
) -> tuple[float, pd.DataFrame | None]:
    if batch_size_pairs < 1:
        raise ValueError("batch_size_pairs must be positive")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # LPIPS model
    loss_fn = lpips.LPIPS(net=net).to(device)
    loss_fn.eval()

    # Read video
    vr = VideoReader(video_path, ctx=cpu(0))
    num_frames = len(vr)
    fps = float(vr.get_avg_fps()) if hasattr(vr, "get_avg_fps") else None
    if fps is None or math.isnan(fps) or fps <= 0:
        fps = 30.0  # fallback

    idx = frame_indices_from_fps(num_frames, fps, sample_fps)
    if len(idx) < 2:
        raise ValueError("Need at least 2 frames to compute tLPIPS.")

    # consecutive pairs on sampled indices: (idx[i], idx[i+1])
    pairs = [(int(idx[i]), int(idx[i + 1])) for i in range(len(idx) - 1)]

    rows = []
    lpips_vals = []

    for start in range(0, len(pairs), batch_size_pairs):
        batch_pairs = pairs[start : start + batch_size_pairs]
        a_idx = np.array([p[0] for p in batch_pairs], dtype=np.int64)
        b_idx = np.array([p[1] for p in batch_pairs], dtype=np.int64)

        # Decode frames for both sides
        a_frames = vr.get_batch(a_idx).asnumpy()  # (B,H,W,3) uint8 RGB
        b_frames = vr.get_batch(b_idx).asnumpy()

        a = _frames_uint8_to_lpips_tensor(a_frames, device)
        b = _frames_uint8_to_lpips_tensor(b_frames, device)

        # LPIPS returns shape (B,1,1,1) or (B,1) depending on version
        d = loss_fn(a, b).view(-1)  # (B,)

        d_cpu = d.detach().cpu().numpy()
        lpips_vals.extend(d_cpu.tolist())

        for i, (fa, fb) in enumerate(batch_pairs):
            # time stamp for the transition (you can also store both endpoints)
            time_a = fa / fps
            time_b = fb / fps
            rows.append(
                {
                    "frame_a": fa,
                    "frame_b": fb,
                    "time_a_sec": time_a,
                    "time_b_sec": time_b,
                    "lpips": float(d_cpu[i]),
                }
            )

    df = pd.DataFrame(rows)
    tlpips = float(np.mean(lpips_vals))  # the scalar tLPIPS

    if out_csv is not None:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        df.to_csv(out_csv, index=False)

    return tlpips, (df if out_csv is not None else None)


if __name__ == "__main__":
    import sys
    from scripts.evaluate import main

    main(["tlpips", *sys.argv[1:]])
