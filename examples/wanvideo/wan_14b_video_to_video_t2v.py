import torch

from diffsynth import ModelManager, WanVideoPipeline, save_video
from torchvision.transforms import v2
from einops import rearrange
from PIL import Image
import imageio
import numpy as np


def pad_frame(image, target_size=(832, 832)):
    """
    Pads a PIL image to the target size with a black background.

    Args:
      image (PIL.Image): Input image.
      target_size (tuple): (width, height) of the desired output image.

    Returns:
      A new PIL.Image of size target_size with the input image centered.
    """
    target_w, target_h = target_size
    padded = Image.new("RGB", target_size, (0, 0, 0))  # Black background
    w, h = image.size
    left = (target_w - w) // 2
    top = (target_h - h) // 2
    padded.paste(image, (left, top))
    return padded

def load_video_as_pil_frames(video_path, num_frames=81):
    """
    Loads a video and returns a list of PIL.Image frames.
    If the video has more frames than `num_frames`, it samples evenly spaced frames.
    """
    reader = imageio.get_reader(video_path)
    total_frames = reader.count_frames()
    print(f"total_frames: {total_frames}")
    assert total_frames >= num_frames, f"Video must have at least {num_frames} frames."

    # Generate evenly spaced frame indices from 0 to total_frames-1
    indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)

    frames = []
    for i in indices:
        frame = reader.get_data(i)
        frame = Image.fromarray(frame).convert("RGB")
        frames.append(frame)
    reader.close()
    return frames

if __name__ == "__main__":
    # Load model
    model_manager = ModelManager(device="cuda")
    model_manager.load_models(
        [
            [
                f"models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-0000{i+1}-of-00006.safetensors"
                for i in range(6)
            ],
            "models/Wan-AI/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth",
            "models/Wan-AI/Wan2.1-T2V-14B/Wan2.1_VAE.pth",
        ],
        torch_dtype=torch.bfloat16, # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
    )
    pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")
    pipe.enable_vram_management(num_persistent_param_in_dit=None) # You can set `num_persistent_param_in_dit` to a small number to reduce VRAM required.
    # Force VAE to move to the right device
    pipe.vae.model = pipe.vae.model.to(dtype=pipe.torch_dtype, device=pipe.device)

    # Load and process video
    # input_video_path = "data/example_dataset/train/video_two_dogs.mp4"
    input_video_path = "outputs/rabbit.mp4"
    prompt = ("Expand the scene around the rabbit enjoying a watermelon, naturally filling the borders with a cohesive continuation of the existing environment. "
              "Preserve the rabbit’s appearance, playful posture, the bright red color of the watermelon, and the overall lighting so that "
              "the newly outpainted areas blend seamlessly with the original content, creating a believable, extended frame.")
    video = load_video_as_pil_frames(input_video_path, num_frames=21)

    denoising_strength = 0.8
    num_inference_steps = 40
    pad_video_on_original_frame = False

    if pad_video_on_original_frame:
        video = [pad_frame(frame, (832, 832)) for frame in video]

    output_video = pipe(
        prompt=prompt,
        negative_prompt="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
        input_video=video,
        denoising_strength=denoising_strength,
        seed=42,
        height=832,
        width=832,
        num_frames=21,
        num_inference_steps=num_inference_steps,
        tiled=True,
        outpaint_video=True,
    )
    save_video(output_video, f"outputs/v2v_using_t2v_denoising_strength_{denoising_strength}_steps_{num_inference_steps}.mp4", fps=9, quality=5)
