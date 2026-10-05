# Modified for Unified Long Video Inpainting and Outpainting; see NOTICE and docs/IMPLEMENTATION.md.
import types

from ..models import ModelManager
from ..models.wan_video_dit import WanModel
from ..models.wan_video_text_encoder import WanTextEncoder
from ..models.wan_video_vae import WanVideoVAE
from ..models.wan_video_image_encoder import WanImageEncoder
from ..models.wan_video_vace import VaceWanModel
from ..schedulers.flow_match import FlowMatchScheduler
from .base import BasePipeline
from .wan_video_utils import blend_windows, denoise_window, get_views
from ..prompters import WanPrompter
import torch, os
from einops import rearrange
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Optional

from ..vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear
from ..models.wan_video_text_encoder import T5RelativeEmbedding, T5LayerNorm
from ..models.wan_video_dit import RMSNorm, sinusoidal_embedding_1d
from ..models.wan_video_vae import RMS_norm, CausalConv3d, Upsample
from ..models.wan_video_motion_controller import WanMotionControllerModel


class WanVideoPipeline(BasePipeline):
    def __init__(self, device="cuda", torch_dtype=torch.float16, tokenizer_path=None):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.motion_controller: WanMotionControllerModel = None
        self.vace: VaceWanModel = None
        self.model_names = [
            "text_encoder",
            "dit",
            "vae",
            "image_encoder",
            "motion_controller",
            "vace",
        ]
        self.height_division_factor = 16
        self.width_division_factor = 16
        self.use_unified_sequence_parallel = False
        self.train_frames = 81  # number of frames per “clip” for co-denoising

    def enable_vram_management(self, num_persistent_param_in_dit=None):
        dtype = next(iter(self.text_encoder.parameters())).dtype
        enable_vram_management(
            self.text_encoder,
            module_map={
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Embedding: AutoWrappedModule,
                T5RelativeEmbedding: AutoWrappedModule,
                T5LayerNorm: AutoWrappedModule,
            },
            module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        dtype = next(iter(self.dit.parameters())).dtype
        enable_vram_management(
            self.dit,
            module_map={
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv3d: AutoWrappedModule,
                torch.nn.LayerNorm: AutoWrappedModule,
                RMSNorm: AutoWrappedModule,
            },
            module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
            max_num_param=num_persistent_param_in_dit,
            overflow_module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        dtype = next(iter(self.vae.parameters())).dtype
        enable_vram_management(
            self.vae,
            module_map={
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv2d: AutoWrappedModule,
                RMS_norm: AutoWrappedModule,
                CausalConv3d: AutoWrappedModule,
                Upsample: AutoWrappedModule,
                torch.nn.SiLU: AutoWrappedModule,
                torch.nn.Dropout: AutoWrappedModule,
            },
            module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        if self.image_encoder is not None:
            dtype = next(iter(self.image_encoder.parameters())).dtype
            enable_vram_management(
                self.image_encoder,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                },
                module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        if self.motion_controller is not None:
            dtype = next(iter(self.motion_controller.parameters())).dtype
            enable_vram_management(
                self.motion_controller,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                },
                module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        if self.vace is not None:
            enable_vram_management(
                self.vace,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                    RMSNorm: AutoWrappedModule,
                },
                module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=self.device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
            )
        self.enable_cpu_offload()

    def fetch_models(self, model_manager: ModelManager):
        text_encoder_model_and_path = model_manager.fetch_model(
            "wan_video_text_encoder", require_model_path=True
        )
        if text_encoder_model_and_path is not None:
            self.text_encoder, tokenizer_path = text_encoder_model_and_path
            self.prompter.fetch_models(self.text_encoder)
            self.prompter.fetch_tokenizer(
                os.path.join(os.path.dirname(tokenizer_path), "google/umt5-xxl")
            )
        self.dit = model_manager.fetch_model("wan_video_dit")
        self.vae = model_manager.fetch_model("wan_video_vae")
        self.image_encoder = model_manager.fetch_model("wan_video_image_encoder")
        self.motion_controller = model_manager.fetch_model("wan_video_motion_controller")
        self.vace = model_manager.fetch_model("wan_video_vace")

    @staticmethod
    def from_model_manager(
        model_manager: ModelManager, torch_dtype=None, device=None, use_usp=False
    ):
        if device is None:
            device = model_manager.device
        if torch_dtype is None:
            torch_dtype = model_manager.torch_dtype
        pipe = WanVideoPipeline(device=device, torch_dtype=torch_dtype)
        pipe.fetch_models(model_manager)
        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size
            from ..distributed.xdit_context_parallel import usp_attn_forward, usp_dit_forward

            for block in pipe.dit.blocks:
                block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
            pipe.dit.forward = types.MethodType(usp_dit_forward, pipe.dit)
            pipe.sp_size = get_sequence_parallel_world_size()
            pipe.use_unified_sequence_parallel = True
        return pipe

    def denoising_model(self):
        return self.dit

    def encode_prompt(self, prompt, positive=True):
        prompt_emb = self.prompter.encode_prompt(prompt, positive=positive, device=self.device)
        return {"context": prompt_emb}

    def encode_image(
        self,
        image,
        end_image,
        num_frames,
        height,
        width,
        tiled=False,
        tile_size=(34, 34),
        tile_stride=(18, 16),
    ):
        image = self.preprocess_image(image.resize((width, height))).to(self.device)
        clip_context = self.image_encoder.encode_image([image])
        msk = torch.ones(1, num_frames, height // 8, width // 8, device=self.device)
        msk[:, 1:] = 0
        if end_image is not None:
            end_image = self.preprocess_image(end_image.resize((width, height))).to(self.device)
            vae_input = torch.concat(
                [
                    image.transpose(0, 1),
                    torch.zeros(3, num_frames - 2, height, width).to(image.device),
                    end_image.transpose(0, 1),
                ],
                dim=1,
            )
            if self.dit.has_image_pos_emb:
                clip_context = torch.concat(
                    [clip_context, self.image_encoder.encode_image([end_image])], dim=1
                )
            msk[:, -1:] = 1
        else:
            vae_input = torch.concat(
                [
                    image.transpose(0, 1),
                    torch.zeros(3, num_frames - 1, height, width).to(image.device),
                ],
                dim=1,
            )

        msk = torch.concat(
            [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1
        )
        msk = msk.view(1, msk.shape[1] // 4, 4, height // 8, width // 8)
        msk = msk.transpose(1, 2)[0]
        y = self.vae.encode(
            [vae_input.to(dtype=self.torch_dtype, device=self.device)],
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )[0]
        y = y.to(dtype=self.torch_dtype, device=self.device)
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        clip_context = clip_context.to(dtype=self.torch_dtype, device=self.device)
        y = y.to(dtype=self.torch_dtype, device=self.device)
        return {"clip_feature": clip_context, "y": y}

    def encode_control_video(
        self, control_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)
    ):
        control_video = self.preprocess_images(control_video)
        control_video = torch.stack(control_video, dim=2).to(
            dtype=self.torch_dtype, device=self.device
        )
        latents = self.encode_video(
            control_video, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
        ).to(dtype=self.torch_dtype, device=self.device)
        return latents

    def prepare_controlnet_kwargs(
        self,
        control_video,
        num_frames,
        height,
        width,
        clip_feature=None,
        y=None,
        tiled=True,
        tile_size=(34, 34),
        tile_stride=(18, 16),
    ):
        if control_video is not None:
            control_latents = self.encode_control_video(
                control_video, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
            )
            if clip_feature is None or y is None:
                clip_feature = torch.zeros(
                    (1, 257, 1280), dtype=self.torch_dtype, device=self.device
                )
                y = torch.zeros(
                    (1, 16, (num_frames - 1) // 4 + 1, height // 8, width // 8),
                    dtype=self.torch_dtype,
                    device=self.device,
                )
            else:
                y = y[:, -16:]
            y = torch.concat([control_latents, y], dim=1)
        return {"clip_feature": clip_feature, "y": y}

    def tensor2video(self, frames):
        frames = rearrange(frames, "C T H W -> T H W C")
        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        frames = [Image.fromarray(frame) for frame in frames]
        return frames

    def prepare_extra_input(self, latents=None):
        return {}

    def encode_video(self, input_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        latents = self.vae.encode(
            input_video,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        return latents

    def mask_latents(self, latents, paint_mode, target_height, target_width, division=8):
        """
        Pads the latent tensor in the spatial dimensions using 0 so that the new spatial dimensions
        match target_height//division and target_width//division.

        Args:
          latents: Tensor of shape (B, C, T, H_in, W_in)
          target_height: Desired output height (e.g., 832)
          target_width: Desired output width (e.g., 832)
          division: Factor used in VAE encoding (typically 8)

        Returns:
          Padded latent tensor with shape (B, C, T, H_target, W_target)
          where H_target = target_height // division and W_target = target_width // division.
        """
        if paint_mode not in {"inpainting", "outpainting"}:
            raise ValueError("paint_mode must be inpainting or outpainting.")
        if target_height % division or target_width % division:
            raise ValueError("Target dimensions must be divisible by the VAE scale.")
        if paint_mode == "outpainting" and (
            target_height // division < latents.shape[-2]
            or target_width // division < latents.shape[-1]
        ):
            raise ValueError("The outpainting canvas cannot be smaller than the input.")
        # Current latent spatial dimensions
        H_in = latents.shape[-2]
        W_in = latents.shape[-1]
        # Target spatial dimensions in latent space
        H_target = target_height // division
        W_target = target_width // division

        B, C, T, _, _ = latents.shape
        # Pad using 0s
        padded = torch.zeros(
            (B, C, T, H_target, W_target), dtype=latents.dtype, device=latents.device
        )

        if paint_mode == "outpainting":
            pad_H_total = H_target - H_in
            pad_W_total = W_target - W_in
            pad_H_left = pad_H_total // 2
            pad_W_left = pad_W_total // 2
            # Copy the original latents into the center of the padded tensor.
            padded[..., pad_H_left : pad_H_left + H_in, pad_W_left : pad_W_left + W_in] = latents
        elif paint_mode == "inpainting":
            # For inpainting we simply work with the original latents.
            padded = latents.clone()
            # Mask the central square area using 0.
            # Determine the side of the square as half of the smaller spatial dimension.
            square_side = min(H_in, W_in) // 2  # e.g., if H_in=W_in=256, then square_side = 128
            top = (H_in - square_side) // 2
            left = (W_in - square_side) // 2
            # Zero-out the central square region.
            padded[..., top : top + square_side, left : left + square_side] = 0
        return padded

    def decode_video(self, latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        frames = self.vae.decode(
            latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
        )
        return frames

    def prepare_unified_sequence_parallel(self):
        return {"use_unified_sequence_parallel": self.use_unified_sequence_parallel}

    def prepare_motion_bucket_id(self, motion_bucket_id):
        motion_bucket_id = torch.Tensor((motion_bucket_id,)).to(
            dtype=self.torch_dtype, device=self.device
        )
        return {"motion_bucket_id": motion_bucket_id}

    def prepare_vace_kwargs(
        self,
        latents,
        vace_video=None,
        vace_mask=None,
        vace_reference_image=None,
        vace_scale=1.0,
        height=480,
        width=832,
        num_frames=81,
        seed=None,
        rand_device="cpu",
        tiled=True,
        tile_size=(34, 34),
        tile_stride=(18, 16),
    ):
        if vace_video is not None or vace_mask is not None or vace_reference_image is not None:
            self.load_models_to_device(["vae"])
            if vace_video is None:
                vace_video = torch.zeros(
                    (1, 3, num_frames, height, width), dtype=self.torch_dtype, device=self.device
                )
            else:
                vace_video = self.preprocess_images(vace_video)
                vace_video = torch.stack(vace_video, dim=2).to(
                    dtype=self.torch_dtype, device=self.device
                )

            if vace_mask is None:
                vace_mask = torch.ones_like(vace_video)
            else:
                vace_mask = self.preprocess_images(vace_mask)
                vace_mask = torch.stack(vace_mask, dim=2).to(
                    dtype=self.torch_dtype, device=self.device
                )

            inactive = vace_video * (1 - vace_mask) + 0 * vace_mask
            reactive = vace_video * vace_mask + 0 * (1 - vace_mask)
            inactive = self.encode_video(
                inactive, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
            ).to(dtype=self.torch_dtype, device=self.device)
            reactive = self.encode_video(
                reactive, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
            ).to(dtype=self.torch_dtype, device=self.device)
            vace_video_latents = torch.concat((inactive, reactive), dim=1)

            vace_mask_latents = rearrange(
                vace_mask[0, 0], "T (H P) (W Q) -> 1 (P Q) T H W", P=8, Q=8
            )
            vace_mask_latents = torch.nn.functional.interpolate(
                vace_mask_latents,
                size=(
                    (vace_mask_latents.shape[2] + 3) // 4,
                    vace_mask_latents.shape[3],
                    vace_mask_latents.shape[4],
                ),
                mode="nearest-exact",
            )

            if vace_reference_image is None:
                pass
            else:
                vace_reference_image = self.preprocess_images([vace_reference_image])
                vace_reference_image = torch.stack(vace_reference_image, dim=2).to(
                    dtype=self.torch_dtype, device=self.device
                )
                vace_reference_latents = self.encode_video(
                    vace_reference_image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
                ).to(dtype=self.torch_dtype, device=self.device)
                vace_reference_latents = torch.concat(
                    (vace_reference_latents, torch.zeros_like(vace_reference_latents)), dim=1
                )
                vace_video_latents = torch.concat(
                    (vace_reference_latents, vace_video_latents), dim=2
                )
                vace_mask_latents = torch.concat(
                    (torch.zeros_like(vace_mask_latents[:, :, :1]), vace_mask_latents), dim=2
                )

                noise = self.generate_noise(
                    (1, 16, 1, latents.shape[3], latents.shape[4]),
                    seed=seed,
                    device=rand_device,
                    dtype=torch.float32,
                )
                noise = noise.to(dtype=self.torch_dtype, device=self.device)
                latents = torch.concat((noise, latents), dim=2)

            vace_context = torch.concat((vace_video_latents, vace_mask_latents), dim=1)
            return latents, {"vace_context": vace_context, "vace_scale": vace_scale}
        else:
            return latents, {"vace_context": None, "vace_scale": vace_scale}

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        negative_prompt="",
        input_image=None,
        end_image=None,
        input_video=None,
        control_video=None,
        vace_video=None,
        vace_video_mask=None,
        vace_reference_image=None,
        vace_scale=1.0,
        denoising_strength=1.0,
        seed=None,
        rand_device="cpu",
        height=480,
        width=832,
        num_frames=81,
        cfg_scale=5.0,
        num_inference_steps=50,
        sigma_shift=5.0,
        motion_bucket_id=None,
        tiled=True,
        tile_size=(30, 52),
        tile_stride=(15, 26),
        tea_cache_l1_thresh=None,
        tea_cache_model_id="",
        progress_bar_cmd=tqdm,
        progress_bar_st=None,
        co_denoising=True,
        based_on_input_mask=False,
        outpaint_video=False,
        inpaint_video=False,
        inpaint_padding=False,
        denoising_batch_frames=81,
        overlap_factor=3,
        use_high_order_solver=True,
        input_mask=None,
    ):
        # Parameter check
        if num_frames < 1 or num_inference_steps < 1:
            raise ValueError("num_frames and num_inference_steps must be positive.")
        if not 0 < denoising_strength <= 1:
            raise ValueError("denoising_strength must lie in (0, 1].")
        if denoising_batch_frames < 4 or overlap_factor < 0:
            raise ValueError("Use at least 4 frames per window and a nonnegative overlap factor.")
        if inpaint_video and outpaint_video:
            raise ValueError("Choose either inpainting or outpainting.")
        if input_mask is not None and (not co_denoising or outpaint_video):
            raise ValueError("input_mask requires co-denoising inpainting.")
        height, width = self.check_resize_height_width(height, width)
        if num_frames % 4 != 1:
            num_frames = (num_frames + 2) // 4 * 4 + 1
            print(f"Only `num_frames % 4 == 1` is acceptable. We round it up to {num_frames}.")

        # Tiler parameters
        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}

        # Encode prompts
        self.load_models_to_device(["text_encoder"])
        prompt_emb_posi = self.encode_prompt(prompt, positive=True)
        prompt_emb_nega = (
            self.encode_prompt(negative_prompt, positive=False) if cfg_scale != 1.0 else None
        )

        # Encode image
        if input_image is not None and self.image_encoder is not None:
            self.load_models_to_device(["image_encoder", "vae"])
            image_emb = self.encode_image(
                input_image, end_image, num_frames, height, width, **tiler_kwargs
            )
        else:
            image_emb = {}

        # ControlNet
        if control_video is not None:
            self.load_models_to_device(["image_encoder", "vae"])
            image_emb = self.prepare_controlnet_kwargs(
                control_video, num_frames, height, width, **image_emb, **tiler_kwargs
            )

        # Motion Controller
        if self.motion_controller is not None and motion_bucket_id is not None:
            motion_kwargs = self.prepare_motion_bucket_id(motion_bucket_id)
        else:
            motion_kwargs = {}

        self.load_models_to_device(["dit", "vace", "vae"])

        # TeaCache
        tea_cache_posi = {
            "tea_cache": TeaCache(
                num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id
            )
            if tea_cache_l1_thresh is not None
            else None
        }
        tea_cache_nega = {
            "tea_cache": TeaCache(
                num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id
            )
            if tea_cache_l1_thresh is not None
            else None
        }

        # Unified Sequence Parallel
        usp_kwargs = self.prepare_unified_sequence_parallel()

        # if input_video is longer than one chunk, do co-denoising
        if input_video is not None and co_denoising:
            return self.co_denoising_gen_long_video(
                prompt_emb_posi=prompt_emb_posi,
                prompt_emb_nega=prompt_emb_nega,
                image_emb=image_emb,
                input_video=input_video,
                control_video=control_video,
                vace_video=vace_video,
                vace_video_mask=vace_video_mask,
                vace_reference_image=vace_reference_image,
                vace_scale=vace_scale,
                denoising_strength=denoising_strength,
                seed=seed,
                rand_device=rand_device,
                height=height,
                width=width,
                num_frames=num_frames,
                cfg_scale=cfg_scale,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                motion_bucket_id=motion_bucket_id,
                tiler_kwargs=tiler_kwargs,
                motion_kwargs=motion_kwargs,
                usp_kwargs=usp_kwargs,
                tea_cache_l1_thresh=tea_cache_l1_thresh,
                tea_cache_model_id=tea_cache_model_id,
                tea_cache_posi=tea_cache_posi,
                tea_cache_nega=tea_cache_nega,
                progress_bar_cmd=progress_bar_cmd,
                based_on_input_mask=based_on_input_mask,
                outpaint_video=outpaint_video,
                inpaint_video=inpaint_video,
                inpaint_padding=inpaint_padding,
                denoising_batch_frames=denoising_batch_frames,
                overlap_factor=overlap_factor,
                use_high_order_solver=use_high_order_solver,
                input_mask=input_mask,
            )

        # Scheduler
        self.scheduler.set_timesteps(
            num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift
        )

        # Initialize noise
        noise = self.generate_noise(
            (1, 16, (num_frames - 1) // 4 + 1, height // 8, width // 8),
            seed=seed,
            device=rand_device,
            dtype=torch.float32,
        )
        noise = noise.to(dtype=self.torch_dtype, device=self.device)
        if input_video is not None:
            self.load_models_to_device(["vae"])
            input_video = self.preprocess_images(input_video)
            input_video = torch.stack(input_video, dim=2).to(
                dtype=self.torch_dtype, device=self.device
            )
            latents = self.encode_video(input_video, **tiler_kwargs).to(
                dtype=self.torch_dtype, device=self.device
            )
            # [slyu]
            # Pad the latent tensor from input resolution (512x512 -> 64x64 in latent space)
            # to the target latent resolution corresponding to the output resolution (e.g., 1024x1024 -> 128x128).
            if outpaint_video:
                latents = self.mask_latents(
                    latents,
                    paint_mode="outpainting",
                    target_height=height,
                    target_width=width,
                    division=8,
                )
            elif inpaint_video and inpaint_padding:
                latents = self.mask_latents(
                    latents,
                    paint_mode="inpainting",
                    target_height=height,
                    target_width=width,
                    division=8,
                )
            # Generate noise with the same shape as latents
            noise = self.generate_noise(
                latents.shape, seed=seed, device=rand_device, dtype=torch.float32
            )
            noise = noise.to(dtype=self.torch_dtype, device=self.device)

            latents = self.scheduler.add_noise(latents, noise, timestep=self.scheduler.timesteps[0])

        else:
            latents = noise

        # Extra input
        extra_input = self.prepare_extra_input(latents)

        # VACE
        latents, vace_kwargs = self.prepare_vace_kwargs(
            latents,
            vace_video,
            vace_video_mask,
            vace_reference_image,
            vace_scale,
            height=height,
            width=width,
            num_frames=num_frames,
            seed=seed,
            rand_device=rand_device,
            **tiler_kwargs,
        )

        # Denoise - original logic
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)

            # Inference
            noise_pred_posi = model_fn_wan_video(
                self.dit,
                motion_controller=self.motion_controller,
                vace=self.vace,
                x=latents,
                timestep=timestep,
                **prompt_emb_posi,
                **image_emb,
                **extra_input,
                **tea_cache_posi,
                **usp_kwargs,
                **motion_kwargs,
                **vace_kwargs,
            )
            if cfg_scale != 1.0:
                noise_pred_nega = model_fn_wan_video(
                    self.dit,
                    motion_controller=self.motion_controller,
                    vace=self.vace,
                    x=latents,
                    timestep=timestep,
                    **prompt_emb_nega,
                    **image_emb,
                    **extra_input,
                    **tea_cache_nega,
                    **usp_kwargs,
                    **motion_kwargs,
                    **vace_kwargs,
                )
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # Scheduler
            latents = self.scheduler.step(
                noise_pred, self.scheduler.timesteps[progress_id], latents
            )

        if vace_reference_image is not None:
            latents = latents[:, :, 1:]

        # Decode
        self.load_models_to_device(["vae"])
        frames = self.decode_video(latents, **tiler_kwargs)
        self.load_models_to_device([])
        frames = self.tensor2video(frames[0])

        return frames

    def fuse_frames_with_optical_flow(self, frames):
        fused = []
        for i, frame in enumerate(frames):
            img = np.array(frame)
            if i > 0:
                prev = np.array(fused[-1])
                # compute dense flow from prev -> img
                flow = cv2.calcOpticalFlowFarneback(
                    cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY),
                    cv2.cvtColor(img, cv2.COLOR_RGB2GRAY),
                    None,
                    0.5,
                    3,
                    15,
                    3,
                    5,
                    1.2,
                    0,
                )
                h, w = flow.shape[:2]
                xx, yy = np.meshgrid(np.arange(w), np.arange(h))
                map_x = (xx + flow[..., 0]).astype(np.float32)
                map_y = (yy + flow[..., 1]).astype(np.float32)
                warped = cv2.remap(prev, map_x, map_y, interpolation=cv2.INTER_LINEAR)
                # blend warped previous with current
                blended = (0.8 * img + 0.2 * warped).astype(np.uint8)
                fused_frame = Image.fromarray(blended)
            else:
                fused_frame = frame
            fused.append(fused_frame)
        return fused

    def co_denoising_gen_long_video(
        self,
        prompt_emb_posi,
        prompt_emb_nega,
        image_emb,
        input_video,
        control_video,
        vace_video,
        vace_video_mask,
        vace_reference_image,
        vace_scale,
        denoising_strength,
        seed,
        rand_device,
        height,
        width,
        num_frames,
        cfg_scale,
        num_inference_steps,
        sigma_shift,
        motion_bucket_id,
        tiler_kwargs,
        motion_kwargs,
        usp_kwargs,
        tea_cache_l1_thresh,
        tea_cache_model_id,
        tea_cache_posi,
        tea_cache_nega,
        progress_bar_cmd,
        based_on_input_mask,
        outpaint_video,
        inpaint_video,
        inpaint_padding,
        denoising_batch_frames,
        overlap_factor,
        use_high_order_solver,
        input_mask=None,
    ):
        # Long-video editing currently supports the text-to-video backbone only.
        if image_emb or any(
            item is not None
            for item in (
                control_video,
                vace_video,
                vace_video_mask,
                vace_reference_image,
                motion_bucket_id,
            )
        ):
            raise ValueError(
                "Co-denoising supports T2V editing; use co_denoising=False for controls."
            )
        if tea_cache_l1_thresh is not None:
            raise ValueError("TeaCache is not supported by the co-denoising path.")

        frames = self.preprocess_images(input_video)
        video = torch.stack(frames, dim=2).to(device=self.device, dtype=self.torch_dtype)
        pixel_mask = None
        if input_mask is not None:
            if len(input_mask) != video.shape[2]:
                raise ValueError("The mask and input video must have the same frame count.")
            masks = np.stack([np.asarray(frame.convert("L")) >= 128 for frame in input_mask])
            if masks.shape[1:] != tuple(video.shape[-2:]):
                raise ValueError("The mask and input video must have the same spatial dimensions.")
            pixel_mask = torch.from_numpy(masks).unsqueeze(0).unsqueeze(0).to(video)
            video = video * (1 - pixel_mask)
        full_lat = self.encode_video(video, **tiler_kwargs)

        # Mask latent cells after encoding, matching the training convention.
        if pixel_mask is not None:
            mask_lat = torch.nn.functional.interpolate(
                pixel_mask, size=full_lat.shape[2:], mode="nearest"
            )
            full_lat = full_lat * (1 - mask_lat)
        elif based_on_input_mask:  # the input video already contains a gray overlay
            # 2b) Zero-out any latent cell whose receptive field saw pure‐128 pixels
            # -------------------------------------------------------------------
            # raw_frames: list of PIL.Image, length=T
            raw_frames = input_video.raw_data() if hasattr(input_video, "raw_data") else input_video

            # 1) make a grayscale stack of shape (T,1,H,W)
            gray_stack = (
                torch.stack([torch.from_numpy(np.array(f.convert("L"))) for f in raw_frames], dim=0)
                .unsqueeze(1)
                .to(device=self.device)
            )  # (T,1,H,W)

            tolerance = 2
            mask_value = 128  # grey
            # pixel_mask = (gray_stack == mask_value).float()
            pixel_mask = (
                (gray_stack.float() >= mask_value - tolerance)
                & (gray_stack.float() <= mask_value + tolerance)
            ).float()
            # now (T,1,H,W)

            # 3) reorder to (1,1,T,H,W)
            pixel_mask = pixel_mask.permute(1, 0, 2, 3).unsqueeze(0)  # (1,1,T,H,W)

            # downsample to latent grid via nearest-neighbor
            B, C_lat, T_lat, H_lat, W_lat = full_lat.shape
            mask_lat = torch.nn.functional.interpolate(
                pixel_mask,
                size=(T_lat, H_lat, W_lat),
                mode="nearest",
            )  # now correct 5D→5D
            mask_lat = mask_lat.to(device=full_lat.device, dtype=full_lat.dtype)

            # Apply the gray-overlay mask at latent resolution.
            full_lat = full_lat * (1.0 - mask_lat)
            # -------------------------------------------------------------------
        elif outpaint_video:
            full_lat = self.mask_latents(
                full_lat,
                paint_mode="outpainting",
                target_height=height,
                target_width=width,
                division=8,
            )
        elif inpaint_video and inpaint_padding:
            full_lat = self.mask_latents(
                full_lat,
                paint_mode="inpainting",
                target_height=height,
                target_width=width,
                division=8,
            )

        self.scheduler.set_timesteps(
            num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift
        )
        noise = self.generate_noise(full_lat.shape, seed, device=rand_device, dtype=torch.float32)
        full_lat = self.scheduler.add_noise(
            full_lat.to(self.device), noise.to(self.device), timestep=self.scheduler.timesteps[0]
        )

        # Retain the research code's frame-budget-to-latent-window convention.
        window_size = (
            min(denoising_batch_frames // 4, full_lat.shape[2])
            if overlap_factor > 0
            else full_lat.shape[2]
        )
        stride = max(1, window_size // overlap_factor) if overlap_factor > 0 else window_size
        views = get_views(full_lat.shape[2], window_size, stride)

        def predict(latents, timestep):
            timestep = torch.as_tensor(timestep, device=self.device, dtype=torch.float32).reshape(1)
            positive = self.dit(latents, timestep=timestep, **prompt_emb_posi)
            if cfg_scale == 1.0:
                return positive
            negative = self.dit(latents, timestep=timestep, **prompt_emb_nega)
            return negative + cfg_scale * (positive - negative)

        with torch.amp.autocast(dtype=self.torch_dtype, device_type=torch.device(self.device).type):
            for step_index, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
                full_lat = blend_windows(
                    full_lat,
                    views,
                    lambda window: denoise_window(
                        self.scheduler, predict, window, timestep, step_index, use_high_order_solver
                    ),
                )

        self.load_models_to_device(["vae"])
        decoded = self.decode_video(
            full_lat.to(device=self.device, dtype=self.torch_dtype), **tiler_kwargs
        )
        self.load_models_to_device([])
        return self.tensor2video(decoded[0])


class TeaCache:
    def __init__(self, num_inference_steps, rel_l1_thresh, model_id):
        self.num_inference_steps = num_inference_steps
        self.step = 0
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.rel_l1_thresh = rel_l1_thresh
        self.previous_residual = None
        self.previous_hidden_states = None

        self.coefficients_dict = {
            "Wan2.1-T2V-1.3B": [
                -5.21862437e04,
                9.23041404e03,
                -5.28275948e02,
                1.36987616e01,
                -4.99875664e-02,
            ],
            "Wan2.1-T2V-14B": [
                -3.03318725e05,
                4.90537029e04,
                -2.65530556e03,
                5.87365115e01,
                -3.15583525e-01,
            ],
            "Wan2.1-I2V-14B-480P": [
                2.57151496e05,
                -3.54229917e04,
                1.40286849e03,
                -1.35890334e01,
                1.32517977e-01,
            ],
            "Wan2.1-I2V-14B-720P": [
                8.10705460e03,
                2.13393892e03,
                -3.72934672e02,
                1.66203073e01,
                -4.17769401e-02,
            ],
        }
        if model_id not in self.coefficients_dict:
            supported_model_ids = ", ".join([i for i in self.coefficients_dict])
            raise ValueError(
                f"{model_id} is not a supported TeaCache model id. Please choose a valid model id in ({supported_model_ids})."
            )
        self.coefficients = self.coefficients_dict[model_id]

    def check(self, dit: WanModel, x, t_mod):
        modulated_inp = t_mod.clone()
        if self.step == 0 or self.step == self.num_inference_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = self.coefficients
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(
                (
                    (modulated_inp - self.previous_modulated_input).abs().mean()
                    / self.previous_modulated_input.abs().mean()
                )
                .cpu()
                .item()
            )
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.step += 1
        if self.step == self.num_inference_steps:
            self.step = 0
        if should_calc:
            self.previous_hidden_states = x.clone()
        return not should_calc

    def store(self, hidden_states):
        self.previous_residual = hidden_states - self.previous_hidden_states
        self.previous_hidden_states = None

    def update(self, hidden_states):
        hidden_states = hidden_states + self.previous_residual
        return hidden_states


def prepare_mask_and_masked_image(images, mask):
    # image: (B,C,T,H,W) or list of PIL images
    if not torch.is_tensor(images):
        frames = images
        arr = (
            torch.stack(
                [torch.from_numpy(np.array(f)).permute(2, 0, 1) for f in frames], dim=1
            ).float()
            / 127.5
            - 1
        )
        images = arr.to(device=mask.device)
    mask = mask.to(dtype=torch.float32)
    mask = mask.clamp(0, 1)
    masked = images * (mask < 0.5)
    return mask, masked


def model_fn_wan_video(
    dit: WanModel,
    motion_controller: WanMotionControllerModel = None,
    vace: VaceWanModel = None,
    x: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    vace_context=None,
    vace_scale=1.0,
    tea_cache: TeaCache = None,
    use_unified_sequence_parallel: bool = False,
    motion_bucket_id: Optional[torch.Tensor] = None,
    **kwargs,
):
    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (
            get_sequence_parallel_rank,
            get_sequence_parallel_world_size,
            get_sp_group,
        )

    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    if motion_bucket_id is not None and motion_controller is not None:
        t_mod = t_mod + motion_controller(motion_bucket_id).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)

    if dit.has_image_input:
        x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)

    x, (f, h, w) = dit.patchify(x)

    freqs = (
        torch.cat(
            [
                dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        )
        .reshape(f * h * w, 1, -1)
        .to(x.device)
    )

    # TeaCache
    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, x, t_mod)
        print(f"tea_cache_update: {tea_cache_update}")
    else:
        tea_cache_update = False
        print("tea_cache is None")

    if vace_context is not None:
        vace_hints = vace(x, vace_context, context, t_mod, freqs)

    # blocks
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)[
                get_sequence_parallel_rank()
            ]
    if tea_cache_update:
        x = tea_cache.update(x)
    else:
        for block_id, block in enumerate(dit.blocks):
            x = block(x, context, t_mod, freqs)
            if vace_context is not None and block_id in vace.vace_layers_mapping:
                x = x + vace_hints[vace.vace_layers_mapping[block_id]] * vace_scale
        if tea_cache is not None:
            tea_cache.store(x)

    x = dit.head(x, t)
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = get_sp_group().all_gather(x, dim=1)
    x = dit.unpatchify(x, (f, h, w))
    return x
