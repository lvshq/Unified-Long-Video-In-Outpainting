import torch

from diffsynth import ModelManager, WanVideoPipeline, save_video
from torchvision.transforms import v2
from einops import rearrange
from PIL import Image
import imageio
import numpy as np


def load_video_as_pil_frames(video_path, num_frames=81):
    """
    Loads a video and returns a list of PIL.Image frames (for use with WanVideoPipeline.pipe()).
    """
    reader = imageio.get_reader(video_path)
    total_frames = reader.count_frames()
    print(f"total_frames: {total_frames}")
    assert total_frames >= num_frames, f"Video must have at least {num_frames} frames."

    frames = []
    for i in range(num_frames):
        frame = reader.get_data(i)
        frame = Image.fromarray(frame).convert("RGB")
        frames.append(frame)
    reader.close()
    return frames

def load_video_tensor(video_path, num_frames=21, height=480, width=832):
    '''
    @video_path
    @num_frames: should be 4*n + 1, e.g. 21, 41, 81
    @height
    @width
    '''
    reader = imageio.get_reader(video_path)
    total_frames = reader.count_frames()
    print(f"total_frames: {total_frames}")
    assert total_frames >= num_frames, f"Video must have at least {num_frames} frames."

    transform = v2.Compose([
        v2.CenterCrop(size=(height, width)),
        v2.Resize(size=(height, width), antialias=True),
        v2.ToTensor(),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    frames = []
    for i in range(num_frames):
        frame = reader.get_data(i)
        frame = Image.fromarray(frame)
        frame = transform(frame)
        frames.append(frame)
    reader.close()

    video_tensor = torch.stack(frames, dim=0)  # (T, C, H, W)
    video_tensor = rearrange(video_tensor, 't c h w -> c t h w')
    return video_tensor


# def generate_with_latents(pipe, prompt, negative_prompt, video_tensor, num_frames=81, steps=50, seed=42):
#     print("start generate_with_latents")
#     device = pipe.device
#     torch.manual_seed(seed)
#
#     video_tensor = video_tensor.to(dtype=pipe.torch_dtype, device=device)
#     print(f"video_tensor.shape: {video_tensor.shape}")
#     video_tensor = video_tensor.unsqueeze(0)
#     print(f"video_tensor.shape after expansion: {video_tensor.shape}")
#     video_latents = pipe.encode_video(video_tensor)[0]  # [B, C, T, H, W]
#
#     pipe.scheduler.set_timesteps(steps)
#
#     video = pipe(prompt=prompt,
#         negative_prompt=negative_prompt,
#         input_image=None,
#         input_video=video_latents,
#         denoising_strength=1.0,
#         seed=seed,
#         rand_device="cpu",
#         height=480,
#         width=832,
#         num_frames=num_frames,
#         cfg_scale=5.0,
#         num_inference_steps=steps)
#     return video


if __name__ == "__main__":
    # Load model
    model_manager = ModelManager(device="cuda")
    model_manager.load_models(
        ["models/Wan-AI/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
        torch_dtype=torch.float32 # Image Encoder is loaded with float32
    )
    model_manager.load_models(
        [
            [
                f"models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-0000{i+1}-of-00007.safetensors"
                for i in range(7)
            ],
            "models/Wan-AI/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth",
            "models/Wan-AI/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth",
        ],
        torch_dtype=torch.bfloat16, # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
    )
    pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")
    pipe.enable_vram_management(num_persistent_param_in_dit=6 * 10**9) # You can set `num_persistent_param_in_dit` to a small number to reduce VRAM required.
    # Force VAE to move to the right device
    pipe.vae.model = pipe.vae.model.to(dtype=pipe.torch_dtype, device=pipe.device)

    # Load and process video
    input_video_path = "data/example_dataset/train/video_two_dogs.mp4"
    prompt = "The golden retriever and a black swan are playing and fighting for a ball"
    # video_tensor = load_video_tensor(input_video_path)
    video = load_video_as_pil_frames(input_video_path, num_frames=5)

    output_video = pipe(
        prompt=prompt,
        negative_prompt="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
        input_video=video,
        denoising_strength=1.1,
        num_frames=5,
        num_inference_steps=200,
        seed=42, tiled=True
    )
    save_video(output_video, "outputs/v2v_output1.mp4", fps=5, quality=3)
