import numpy as np
import cv2
import torch
import lpips
from collections import defaultdict
from pathlib import Path
from PIL import Image
from decord import VideoReader, cpu
from skimage.metrics import structural_similarity as structural_ssim


# -----------------------
# Video loading utilities
# -----------------------
def load_video_rgb_uint8(path, resize_to=None, max_frames=None):
    """
    Returns frames: (T, H, W, 3) uint8 RGB
    """
    vr = VideoReader(path, ctx=cpu(0))
    T = len(vr) if max_frames is None else min(len(vr), max_frames)
    if T < 1:
        raise ValueError(f"No frames available in {path}")
    frames = vr.get_batch(list(range(T))).asnumpy()  # uint8 RGB

    if resize_to is not None:
        W, H = resize_to
        out = []
        for f in frames:
            out.append(cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA))
        frames = np.stack(out, axis=0)

    return frames


# -----------------------
# Mask extraction (robust)
# -----------------------
def mask_from_gray_overlay(ref_rgb, mask_rgb, diff_thresh=25):
    """
    ref_rgb, mask_rgb: (H,W,3) uint8 RGB
    Returns binary mask M: (H,W) uint8 with 1 in overlay region.
    Uses per-pixel L2 diff between mask frame and ref frame.
    """
    diff = mask_rgb.astype(np.float32) - ref_rgb.astype(np.float32)
    dist = np.linalg.norm(diff, axis=2)  # (H,W), safe
    M = (dist > diff_thresh).astype(np.uint8)

    # Optional cleanup
    M = cv2.medianBlur(M, 5)
    # fill small holes / smooth boundary a bit
    kernel = np.ones((3, 3), np.uint8)
    M = cv2.morphologyEx(M, cv2.MORPH_CLOSE, kernel, iterations=1)

    return M  # 0/1


def bbox_from_mask(M, pad=8):
    ys, xs = np.where(M > 0)
    if len(xs) == 0:
        return None
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    y1 = min(M.shape[0] - 1, y1 + pad)
    x1 = min(M.shape[1] - 1, x1 + pad)
    return (y0, y1 + 1, x0, x1 + 1)  # slice-style


def composite_for_region(pred_rgb, ref_rgb, region_mask, mode="masked"):
    """
    Make two images that are identical outside the eval region, so LPIPS focuses on the eval region.

    mode="masked": evaluate masked region -> outside region set to ref for BOTH
    mode="unmasked": evaluate unmasked region -> inside masked set to ref for BOTH

    Returns:
      pred_comp, ref_comp: (H,W,3) uint8
    """
    M = region_mask.astype(bool)
    pred_c = pred_rgb.copy()
    ref_c = ref_rgb.copy()

    if mode == "masked":
        # outside masked region becomes identical (ref) in both
        pred_c[~M] = ref_rgb[~M]
        ref_c[~M] = ref_rgb[~M]
    elif mode == "unmasked":
        # inside masked region becomes identical (ref) in both
        pred_c[M] = ref_rgb[M]
        ref_c[M] = ref_rgb[M]
    else:
        raise ValueError("mode must be 'masked' or 'unmasked'")
    return pred_c, ref_c


# -----------------------
# Metrics
# -----------------------
@torch.no_grad()
def lpips_score_batch(pred_uint8, ref_uint8, net="alex", device=None, batch=32):
    """Evaluate aligned crops in bounded batches, allowing different crop sizes."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = lpips.LPIPS(net=net).to(device).eval()
    groups = defaultdict(list)
    for index, frame in enumerate(pred_uint8):
        groups[frame.shape].append(index)
    scores = np.empty(len(pred_uint8), dtype=np.float32)
    for indices in groups.values():
        for start in range(0, len(indices), batch):
            ids = indices[start : start + batch]

            def to_tensor(frames):
                array = np.stack([frames[index] for index in ids])
                return (
                    torch.from_numpy(array)
                    .permute(0, 3, 1, 2)
                    .to(device=device, dtype=torch.float32)
                    / 127.5
                    - 1
                )

            scores[ids] = (
                loss_fn(to_tensor(pred_uint8), to_tensor(ref_uint8)).flatten().cpu().numpy()
            )
    return scores


def masked_ssim(pred_rgb, ref_rgb, eval_mask, use_grayscale=False):
    """
    Compute SSIM averaged ONLY over eval_mask region.
    Returns scalar SSIM for one frame.
    """
    M = eval_mask.astype(bool)
    if M.sum() == 0:
        return np.nan

    if use_grayscale:
        a = cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2GRAY)
        b = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2GRAY)
        ssim_val, ssim_map = structural_ssim(a, b, data_range=255, full=True)
    else:
        ssim_val, ssim_map = structural_ssim(
            pred_rgb, ref_rgb, data_range=255, channel_axis=2, full=True
        )

    # ssim_map shape:
    # - grayscale: (H,W)
    # - color: (H,W) in recent skimage (it returns per-pixel scalar map even with channel_axis)
    return float(np.mean(ssim_map[M]))


# -----------------------
# Main: video evaluation
# -----------------------
def evaluate_video_with_mask(
    ref_video,  # "original" / ground-truth aligned to pred
    pred_video,  # generated video to evaluate
    mask_video,  # gray-overlay mask video aligned to ref
    resize_to=None,  # e.g. (512,512) as (W,H)
    max_frames=None,
    diff_thresh=25,  # mask extraction threshold
    region="unmasked",  # "masked" or "unmasked"
    lpips_net="alex",
    ssim_grayscale=False,
    crop_to_bbox=False,  # speeds up + reduces background influence
    device=None,
    mask_format="gray-overlay",
):
    if region not in {"masked", "unmasked"}:
        raise ValueError("region must be masked or unmasked")
    if mask_format not in {"binary", "gray-overlay"}:
        raise ValueError("mask_format must be binary or gray-overlay")
    ref = load_video_rgb_uint8(ref_video, resize_to=resize_to, max_frames=max_frames)
    pred = load_video_rgb_uint8(pred_video, resize_to=resize_to, max_frames=max_frames)
    if Path(mask_video).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        with Image.open(mask_video) as image:
            image = image.convert("RGB")
            if resize_to is not None:
                image = image.resize(resize_to, resample=Image.Resampling.NEAREST)
            mask_frame = np.asarray(image)
        mvid = np.repeat(mask_frame[None], len(ref), axis=0)
    else:
        mvid = load_video_rgb_uint8(mask_video, resize_to=resize_to, max_frames=max_frames)
    if ref.shape != pred.shape or ref.shape != mvid.shape:
        raise ValueError(
            "Reference, prediction, and mask must have identical frame counts and dimensions."
        )
    T = len(ref)

    ssim_list = []
    pred_comp_list = []
    ref_comp_list = []

    for t in range(T):
        M = (
            (mvid[t].mean(axis=-1) >= 128).astype(np.uint8)
            if mask_format == "binary"
            else mask_from_gray_overlay(ref[t], mvid[t], diff_thresh=diff_thresh)
        )

        # Choose eval region mask
        eval_mask = M if region == "masked" else (1 - M)

        if crop_to_bbox:
            bb = bbox_from_mask(eval_mask, pad=8)
            if bb is None:
                ssim_list.append(np.nan)
                pred_comp_list.append(None)
                ref_comp_list.append(None)
                continue
            y0, y1, x0, x1 = bb
            ref_crop = ref[t][y0:y1, x0:x1]
            pred_crop = pred[t][y0:y1, x0:x1]
            eval_crop = eval_mask[y0:y1, x0:x1]
        else:
            ref_crop = ref[t]
            pred_crop = pred[t]
            eval_crop = eval_mask

        # SSIM on eval region only
        ssim_list.append(masked_ssim(pred_crop, ref_crop, eval_crop, use_grayscale=ssim_grayscale))

        # LPIPS: composite so outside eval region is identical in both
        pred_c, ref_c = composite_for_region(
            pred_crop, ref_crop, eval_crop, mode="masked"
        )  # eval_crop is "masked region in this crop"
        pred_comp_list.append(pred_c)
        ref_comp_list.append(ref_c)

    # LPIPS batch over frames that produced crops
    # Filter frames with NaN ssim (no valid mask)
    valid_idx = [i for i, v in enumerate(ssim_list) if not np.isnan(v)]
    if len(valid_idx) == 0:
        return {
            "num_frames": int(T),
            "num_valid": 0,
            "ssim_per_frame": np.array(ssim_list, dtype=np.float32),
            "lpips_per_frame": np.array([], dtype=np.float32),
            "ssim_mean": np.nan,
            "lpips_mean": np.nan,
        }

    pred_comp = [pred_comp_list[i] for i in valid_idx]
    ref_comp = [ref_comp_list[i] for i in valid_idx]

    lpips_vals = lpips_score_batch(pred_comp, ref_comp, net=lpips_net, device=device)

    ssim_arr = np.array(ssim_list, dtype=np.float32)
    ssim_valid = ssim_arr[valid_idx]

    return {
        "num_frames": int(T),
        "num_valid": int(len(valid_idx)),
        "ssim_per_frame": ssim_arr,
        "lpips_per_frame": lpips_vals,
        "ssim_mean": float(np.mean(ssim_valid)),
        "ssim_median": float(np.median(ssim_valid)),
        "lpips_mean": float(np.mean(lpips_vals)),
        "lpips_median": float(np.median(lpips_vals)),
    }


if __name__ == "__main__":
    import sys
    from scripts.evaluate import main

    main(["ssim-lpips", *sys.argv[1:]])
