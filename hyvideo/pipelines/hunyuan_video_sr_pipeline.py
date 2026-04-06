# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.

from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any

import numpy as np
import torch
from einops import rearrange
from PIL import Image
from torch.nn import functional as F

from diffusers.models import AutoencoderKL
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import BaseOutput

from hyvideo.commons.parallel_states import get_parallel_state
from hyvideo.commons import auto_offload_model, get_rank
from hyvideo.models.text_encoders import TextEncoder
from hyvideo.models.transformers.hunyuanvideo_1_5_transformer import (
    HunyuanVideo_1_5_DiffusionTransformer,
)
from hyvideo.models.transformers.modules.upsample import SRTo720pUpsampler
from hyvideo.utils.data_utils import generate_crop_size_list

from .hunyuan_video_pipeline import HunyuanVideo_1_5_Pipeline
from .pipeline_utils import rescale_noise_cfg, retrieve_timesteps

def expand_dims(tensor: torch.Tensor, ndim: int):
    shape = tensor.shape + (1,) * (ndim - tensor.ndim)
    return tensor.reshape(shape)

class BucketMap:
    """Maps low-resolution bucket sizes to corresponding high-resolution bucket sizes."""

    def __init__(self, lr_base_size, hr_base_size, lr_patch_size, hr_patch_size):
        self.lr_buckets = generate_crop_size_list(base_size=lr_base_size, patch_size=lr_patch_size)
        self.hr_buckets = generate_crop_size_list(base_size=hr_base_size, patch_size=hr_patch_size)

        self.lr_aspect_ratios = np.array([float(w) / float(h) for w, h in self.lr_buckets])
        self.hr_aspect_ratios = np.array([float(w) / float(h) for w, h in self.hr_buckets])

        self.hr_bucket_map = {}
        for i, (lr_w, lr_h) in enumerate(self.lr_buckets):
            lr_ratio = self.lr_aspect_ratios[i]
            closest_hr_ratio_id = np.abs(self.hr_aspect_ratios - lr_ratio).argmin()
            self.hr_bucket_map[(lr_w, lr_h)] = self.hr_buckets[closest_hr_ratio_id]

    def __call__(self, lr_bucket):
        """
        Args:
            lr_bucket (tuple): Low-resolution bucket size as (width, height).

        Returns:
            tuple: High-resolution bucket size as (width, height).
        """
        if lr_bucket not in self.hr_bucket_map:
            raise ValueError(f"LR bucket {lr_bucket} not found in bucket map")
        return self.hr_bucket_map[lr_bucket]

SizeMap = {
    "480p": 640,
    "720p": 960,
    "1080p": 1440,
}

@dataclass
class HunyuanVideo_1_5_SR_PipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]


class HunyuanVideo_1_5_SR_Pipeline(HunyuanVideo_1_5_Pipeline):

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: TextEncoder,
        transformer: HunyuanVideo_1_5_DiffusionTransformer,
        scheduler: KarrasDiffusionSchedulers,
        upsampler: SRTo720pUpsampler,
        flow_shift: float=6.0,
        guidance_scale: float=6.0,
        num_inference_steps: int = 30,
        embedded_guidance_scale: Optional[float] = None,
        base_resolution: str = "480p",
        text_encoder_2: Optional[TextEncoder] = None,
        progress_bar_config: Dict[str, Any] = None,
        vision_num_semantic_tokens=729,
        vision_states_dim=1152,
        glyph_byT5_v2=True,
        byt5_model=None,
        byt5_tokenizer=None,
        byt5_max_length=256,
        prompt_format=None,
        execution_device=None,
        vision_encoder=None,
        enable_offloading=False,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            transformer=transformer,
            scheduler=scheduler,
            text_encoder_2=text_encoder_2,
            progress_bar_config=progress_bar_config,
            vision_num_semantic_tokens=vision_num_semantic_tokens,
            vision_states_dim=vision_states_dim,
            glyph_byT5_v2=glyph_byT5_v2,
            byt5_model=byt5_model,
            byt5_tokenizer=byt5_tokenizer,
            byt5_max_length=byt5_max_length,
            prompt_format=prompt_format,
            execution_device=execution_device,
            vision_encoder=vision_encoder,
            enable_offloading=enable_offloading,
        )
        assert upsampler is not None
        assert base_resolution is not None
        self.register_modules(upsampler=upsampler)
        self.register_to_config(
            base_resolution=base_resolution,
            flow_shift=flow_shift,
            guidance_scale=guidance_scale,
            embedded_guidance_scale=embedded_guidance_scale,
            num_inference_steps=num_inference_steps,
        )


    def add_noise_to_lq(self, lq_latents, strength=0.7):
        noise = torch.randn_like(lq_latents)
        timestep = torch.tensor([1000.0], device=self.execution_device) * strength
        t = expand_dims(timestep, lq_latents.ndim)
        return (1 - t / 1000.0) * lq_latents + (t / 1000.0) * noise

    def _prepare_lq_cond_latents(self, lq_latents):
        """
        Prepare conditional latents and mask for multitask training.

        Args:
            lq_latents: Low-resolution latent tensor.

        Returns:
            torch.Tensor: Low-resolution conditional latent tensor.
        """
        b, _, f, h, w = lq_latents.shape
        mask_ones = torch.ones(b, 1, f, h, w, device=lq_latents.device, dtype=lq_latents.dtype)
        cond_latents = torch.concat([lq_latents, mask_ones], dim=1)
        
        return cond_latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        video_length: int,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        seed: Optional[int] = None,
        embedded_guidance_scale: Optional[float] = None,
        reference_image=None,
        lq_latents=None,
        output_type: Optional[str] = "pt",
        return_dict: bool = True,
        enable_vae_tile_parallelism: bool = True,
        **kwargs,
    ):
        r"""
        Runs the super-resolution (SR) pipeline for video generation.

        Args:
            prompt (`str` or `List[str]`):
                Text prompt(s) that describe the desired video content.
            video_length (`int`):
                Number of frames in the video to generate.
            num_inference_steps (`int`, *optional*, defaults to value in config):
                Number of denoising steps during SR. Higher values may improve visual quality at the cost of slower inference.
            guidance_scale (`float`, *optional*, defaults to value in config):
                How closely to follow the prompt. `guidance_scale > 1` enables classifier-free guidance.
            negative_prompt (`str` or `List[str]`, *optional*):
                Prompt(s) of what should not appear in the generated video.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                PyTorch random generator(s) for deterministic and reproducible results.
            seed (`int`, *optional*):
                If specified, used to construct a generator for reproducibility.
            embedded_guidance_scale (`float`, *optional*):
                Additional guidance scale for enhanced control, if model supports it.
            reference_image (PIL.Image or `str`, *optional*):
                Reference image for image-to-video (i2v) tasks. Can be a PIL image or local file path. Set to `None` for text-to-video (t2v) mode.
            lq_latents (`torch.Tensor`, *optional*):
                Low-quality (LQ) video latents to use as input for SR upsampling step. Should have shape (B, C, F, H, W).
            output_type (`str`, *optional*, defaults to "pt"):
                Output format, either `"pt"` (PyTorch tensor) or `"np"` (NumPy array).
            return_dict (`bool`, *optional*, defaults to True):
                Whether to return a [`HunyuanVideo_1_5_SR_PipelineOutput`] or a tuple.
            **kwargs:
                Additional keyword arguments.
        """
        target_resolution = self.ideal_resolution

        if guidance_scale is None:
            guidance_scale = self.config.guidance_scale
        if embedded_guidance_scale is None:
            embedded_guidance_scale = self.config.embedded_guidance_scale
        if num_inference_steps is None:
            num_inference_steps = self.config.num_inference_steps

        if reference_image is not None:
            task_type = "i2v"
            if isinstance(reference_image, str):
                reference_image = Image.open(reference_image).convert('RGB')
            elif not isinstance(reference_image, Image.Image):
                raise ValueError("reference_image must be a PIL Image or path to image file")
            semantic_images_np = np.array(reference_image)
        else:
            task_type = "t2v"
            semantic_images_np = None

        self.scheduler = self._create_scheduler(self.config.flow_shift)

        if get_parallel_state().sp_enabled:
            assert seed is not None
        if generator is None and seed is not None:
            generator = torch.Generator(device=torch.device('cpu')).manual_seed(seed)

        sr_stride = 16
        base_size = SizeMap[self.config.base_resolution]
        sr_size = SizeMap[self.ideal_resolution]
        bucket_map = BucketMap(lr_base_size=base_size, hr_base_size=sr_size, lr_patch_size=16, hr_patch_size=sr_stride)
        lr_video_height, lr_video_width = [x * 16 for x in lq_latents.shape[-2:]]
        width, height = bucket_map((lr_video_width, lr_video_height))

        latent_target_length, latent_height, latent_width = self.get_latent_size(video_length, height, width)
        n_tokens = latent_target_length * latent_height * latent_width

        self._guidance_scale = guidance_scale
        self._guidance_rescale = kwargs.get("guidance_rescale", 0.0)
        self._clip_skip = kwargs.get("clip_skip", None)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = 1
        device = self.execution_device

        if get_rank() == 0:
            print(
                '\n'
                f"{'=' * 60}\n"
                f"🎬  HunyuanVideo SR Generation Task\n"
                f"{'-' * 60}\n"
                f"Prompt:                    {prompt}\n"
                f"Video Length:              {video_length}\n"
                f"Reference Image:           {reference_image}\n"
                f"Guidance Scale:            {guidance_scale}\n"
                f"Guidance Embedded Scale:   {embedded_guidance_scale}\n"
                f"Shift:                     {self.config.flow_shift}\n"
                f"Seed:                      {seed}\n"
                f"Video Resolution:          {width} x {height}\n"
                f"Transformer dtype:         {self.transformer.dtype}\n"
                f"Sampling Steps:            {num_inference_steps}\n"
                f"Use Meanflow:              {self.use_meanflow}\n"
                f"{'=' * 60}\n"
            )

        with auto_offload_model(self.text_encoder, self.execution_device, enabled=self.enable_offloading):
            (
                prompt_embeds,
                negative_prompt_embeds,
                prompt_mask,
                negative_prompt_mask,
            ) = self.encode_prompt(
                prompt,
                device,
                num_videos_per_prompt,
                self.do_classifier_free_guidance,
                negative_prompt,
                clip_skip=self.clip_skip,
                data_type="video",
            )


        extra_kwargs = {}
        if self.config.glyph_byT5_v2:
            with auto_offload_model(self.byt5_model, self.execution_device, enabled=self.enable_offloading):
                extra_kwargs = self._prepare_byt5_embeddings(prompt, device)

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            if prompt_mask is not None:
                prompt_mask = torch.cat([negative_prompt_mask, prompt_mask])

        extra_set_timesteps_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.set_timesteps, {"n_tokens": n_tokens}
        )

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            **extra_set_timesteps_kwargs,
        )

        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            32,
            latent_height,
            latent_width,
            latent_target_length,
            self.target_dtype,
            device,
            generator,
        )

        with auto_offload_model(self.vae, self.execution_device, enabled=self.enable_offloading):
            image_cond = self.get_image_condition_latents(task_type, reference_image, height, width)

        tgt_shape = latents.shape[-2:]  # (h w)
        bsz = lq_latents.shape[0]
        lq_latents = rearrange(lq_latents, "b c f h w -> (b f) c h w")
        lq_latents = F.interpolate(lq_latents, size=tgt_shape, mode="bilinear", align_corners=False)
        lq_latents = rearrange(lq_latents, "(b f) c h w -> b c f h w", b=bsz)
        with auto_offload_model(self.upsampler, self.execution_device, enabled=self.enable_offloading):
            lq_latents = self.upsampler(lq_latents.to(dtype=torch.float32, device=self.execution_device))
        lq_latents = lq_latents.to(dtype=latents.dtype)

        noise_scale = 0.7
        lq_latents = self.add_noise_to_lq(lq_latents, noise_scale)

        multitask_mask = self.get_task_mask(task_type, latent_target_length)
        cond_latents = self._prepare_cond_latents(task_type, image_cond, latents, multitask_mask)
        lq_cond_latents = self._prepare_lq_cond_latents(lq_latents)

        condition = torch.concat([cond_latents, lq_cond_latents], dim=1)

        c = lq_latents.shape[1]
        zero_lq_condition = condition.clone()
        zero_lq_condition[:, c + 1 : 2 * c + 1] = torch.zeros_like(lq_latents)
        zero_lq_condition[:, 2 * c + 1] = 0

        with auto_offload_model(self.vision_encoder, self.execution_device, enabled=self.enable_offloading):
            vision_states = self._prepare_vision_states(
                semantic_images_np, target_resolution, latents, device
            )

        extra_step_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.step, {"generator": generator, "eta": kwargs.get("eta", 0.0)},
        )

        self._num_timesteps = len(timesteps)
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        with self.progress_bar(total=num_inference_steps) as progress_bar, auto_offload_model(self.transformer, self.execution_device, enabled=self.enable_offloading):
            for i, t in enumerate(timesteps):
                if t < 1000 * noise_scale:
                    condition = zero_lq_condition

                latents_concat = torch.concat([latents, condition], dim=1)
                latent_model_input = torch.cat([latents_concat] * 2) if self.do_classifier_free_guidance else latents_concat

                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                t_expand = t.expand(latent_model_input.shape[0])
                if not self.use_meanflow:
                    timesteps_r = None
                else:
                    if i == len(timesteps) - 1:
                        timesteps_r = torch.tensor([0.0], device=self.execution_device)
                    else:
                        timesteps_r = timesteps[i + 1]
                    timesteps_r = timesteps_r.expand(latent_model_input.shape[0])

                guidance_expand = (
                    torch.full(
                        (latent_model_input.shape[0],),
                        embedded_guidance_scale * 1000.0,
                        dtype=self.target_dtype,
                        device=device,
                    )
                    if embedded_guidance_scale is not None
                    else None
                )

                with torch.autocast(device_type="cuda", dtype=self.target_dtype, enabled=self.autocast_enabled):
                    output = self.transformer(
                        latent_model_input,
                        t_expand,
                        prompt_embeds,
                        None,
                        prompt_mask,
                        timestep_r=timesteps_r,
                        vision_states=vision_states,
                        mask_type=task_type,
                        guidance=guidance_expand,
                        return_dict=False,
                        extra_kwargs=extra_kwargs,
                    )
                    noise_pred = output[0]

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.guidance_rescale > 0.0 and self.do_classifier_free_guidance:
                    noise_pred = rescale_noise_cfg(
                        noise_pred,
                        noise_pred_text,
                        guidance_rescale=self.guidance_rescale,
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # Update progress bar
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    if progress_bar is not None:
                        progress_bar.update()

        if output_type == "latent":
            video_frames = latents
        else:
            if len(latents.shape) == 4:
                latents = latents.unsqueeze(2)
            elif len(latents.shape) != 5:
                raise ValueError(
                    f"Only support latents with shape (b, c, h, w) or (b, c, f, h, w), but got {latents.shape}."
                )

            if hasattr(self.vae.config, "shift_factor") and self.vae.config.shift_factor:
                latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
            else:
                latents = latents / self.vae.config.scaling_factor

            if enable_vae_tile_parallelism and hasattr(self.vae, 'enable_tile_parallelism'):
                self.vae.enable_tile_parallelism()
            with torch.autocast(device_type="cuda", dtype=self.vae_dtype, enabled=self.vae_autocast_enabled), auto_offload_model(self.vae, self.execution_device, enabled=self.enable_offloading):
                self.vae.enable_tiling()
                video_frames = self.vae.decode(latents, return_dict=False, generator=generator)[0]
                self.vae.disable_tiling()

            if video_frames is not None:
                video_frames = (video_frames / 2 + 0.5).clamp(0, 1).cpu().float()

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return video_frames

        return HunyuanVideo_1_5_SR_PipelineOutput(videos=video_frames)
