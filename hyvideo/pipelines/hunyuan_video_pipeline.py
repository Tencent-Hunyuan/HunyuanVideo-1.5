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


import psutil
import inspect
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import loguru

import numpy as np
import torch
from einops import rearrange
from PIL import Image
import torchvision.transforms as transforms
from torch import distributed as dist

from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import BaseOutput, deprecate, logging

from hyvideo.commons import (
    PIPELINE_CONFIGS,
    SR_PIPELINE_CONFIGS,
    TRANSFORMER_VERSION_TO_SR_VERSION,
    auto_offload_model,
    get_gpu_memory,
    get_rank,
    is_flash3_available,
    is_angelslim_available,
)
from hyvideo.commons.parallel_states import get_parallel_state

from hyvideo.models.autoencoders import hunyuanvideo_15_vae
from hyvideo.models.text_encoders import PROMPT_TEMPLATE, TextEncoder
from hyvideo.models.text_encoders.byT5 import load_glyph_byT5_v2
from hyvideo.models.text_encoders.byT5.format_prompt import MultilingualPromptFormat
from hyvideo.models.transformers.hunyuanvideo_1_5_transformer import (
    HunyuanVideo_1_5_DiffusionTransformer,
)
from hyvideo.models.transformers.modules.upsample import SRTo720pUpsampler, SRTo1080pUpsampler
from hyvideo.models.vision_encoder import VisionEncoder

from hyvideo.schedulers.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler

from hyvideo.utils.data_utils import (
    generate_crop_size_list,
    get_closest_ratio,
    resize_and_center_crop,
)
from hyvideo.utils.multitask_utils import (
    merge_tensor_by_mask,
)
from hyvideo.commons.infer_state import InferState

from .pipeline_utils import retrieve_timesteps, rescale_noise_cfg


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

@dataclass
class HunyuanVideoPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]
    sr_videos: Optional[Union[torch.Tensor, np.ndarray]] = None


class HunyuanVideo_1_5_Pipeline(DiffusionPipeline):

    model_cpu_offload_seq = "text_encoder->text_encoder_2->byt5_model->transformer->vae"
    _optional_components = ["text_encoder_2"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: TextEncoder,
        transformer: HunyuanVideo_1_5_DiffusionTransformer,
        scheduler: KarrasDiffusionSchedulers,
        text_encoder_2: Optional[TextEncoder] = None,
        flow_shift: float = 7.0,
        guidance_scale: float = 6.0,
        num_inference_steps: int = 50,
        embedded_guidance_scale: Optional[float] = None,
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
        super().__init__()

        self.register_to_config(
            glyph_byT5_v2=glyph_byT5_v2,
            byt5_max_length=byt5_max_length,
            vision_num_semantic_tokens=vision_num_semantic_tokens,
            vision_states_dim=vision_states_dim,
            flow_shift=flow_shift,
            guidance_scale=guidance_scale,
            embedded_guidance_scale=embedded_guidance_scale,
            num_inference_steps=num_inference_steps,
        )

        if progress_bar_config is None:
            progress_bar_config = {}
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        self._progress_bar_config.update(progress_bar_config)

        if glyph_byT5_v2:
            self.byt5_max_length = byt5_max_length
            self.register_modules(
                vae=vae,
                text_encoder=text_encoder,
                transformer=transformer,
                scheduler=scheduler,
                text_encoder_2=text_encoder_2,
                byt5_model=byt5_model,
                byt5_tokenizer=byt5_tokenizer,
            )
            self.prompt_format = prompt_format
        else:
            self.register_modules(
                vae=vae,
                text_encoder=text_encoder,
                transformer=transformer,
                scheduler=scheduler,
                text_encoder_2=text_encoder_2,
            )
            self.byt5_model = None
            self.byt5_tokenizer = None


        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.text_len = text_encoder.max_length
        self.target_dtype = torch.bfloat16
        self.vae_dtype = torch.float16
        self.autocast_enabled = True
        self.vae_autocast_enabled = True
        self.enable_offloading = enable_offloading
        self.execution_device = torch.device(execution_device)

        if vision_encoder:
            self.register_modules(vision_encoder=vision_encoder)
        else:
            self.vision_encoder = None

        # Default i2v target size configurations
        self.target_size_config = {
            "360p": {"bucket_hw_base_size": 480, "bucket_hw_bucket_stride": 16},
            "480p": {"bucket_hw_base_size": 640, "bucket_hw_bucket_stride": 16},
            "720p": {"bucket_hw_base_size": 960, "bucket_hw_bucket_stride": 16},
            "1080p": {"bucket_hw_base_size": 1440, "bucket_hw_bucket_stride": 16},
        }


    @classmethod
    def _create_scheduler(cls, flow_shift):
        scheduler = FlowMatchDiscreteScheduler(
            shift=flow_shift,
            reverse=True,
            solver="euler",
        )
        return scheduler

    @classmethod
    def _load_byt5(cls, cached_folder, glyph_byT5_v2, byt5_max_length, device):
        if not glyph_byT5_v2:
            byt5_kwargs = None
            prompt_format = None
            return byt5_kwargs, prompt_format
        try:
            load_from = os.path.join(cached_folder, "text_encoder")
            glyph_root = os.path.join(load_from, "Glyph-SDXL-v2")
            if not os.path.exists(glyph_root):
                raise RuntimeError(
                    f"Glyph checkpoint not found from '{glyph_root}'. \n"
                    "Please download from https://modelscope.cn/models/AI-ModelScope/Glyph-SDXL-v2/files.\n\n"
                    "- Required files:\n"
                    "    Glyph-SDXL-v2\n"
                    "    ├── assets\n"
                    "    │   ├── color_idx.json\n"
                    "    │   └── multilingual_10-lang_idx.json\n"
                    "    └── checkpoints\n"
                    "        └── byt5_model.pt\n"
                )

            byT5_google_path = os.path.join(load_from, "byt5-small")
            if not os.path.exists(byT5_google_path):
                loguru.logger.warning(f"ByT5 google path not found from: {byT5_google_path}. Try downloading from https://huggingface.co/google/byt5-small.")
                byT5_google_path = "google/byt5-small"


            multilingual_prompt_format_color_path = os.path.join(glyph_root, "assets/color_idx.json")
            multilingual_prompt_format_font_path = os.path.join(glyph_root, "assets/multilingual_10-lang_idx.json")

            byt5_args = dict(
                byT5_google_path=byT5_google_path,
                byT5_ckpt_path=os.path.join(glyph_root, "checkpoints/byt5_model.pt"),
                multilingual_prompt_format_color_path=multilingual_prompt_format_color_path,
                multilingual_prompt_format_font_path=multilingual_prompt_format_font_path,
                byt5_max_length=byt5_max_length
            )

            byt5_kwargs = load_glyph_byT5_v2(byt5_args, device=device)
            prompt_format = MultilingualPromptFormat(
                font_path=multilingual_prompt_format_font_path,
                color_path=multilingual_prompt_format_color_path
            )
            return byt5_kwargs, prompt_format
        except Exception as e:
            raise RuntimeError("Error loading byT5 glyph processor") from e


    def encode_prompt(
        self,
        prompt,
        device,
        num_videos_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        clip_skip: Optional[int] = None,
        text_encoder: Optional[TextEncoder] = None,
        data_type: Optional[str] = "image",
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_videos_per_prompt (`int`):
                number of videos that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the video generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            attention_mask (`torch.Tensor`, *optional*):
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            negative_attention_mask (`torch.Tensor`, *optional*):
                Attention mask for negative prompt embeddings.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            text_encoder (TextEncoder, *optional*):
                Text encoder to use. If None, uses the pipeline's default text encoder.
            data_type (`str`, *optional*):
                Type of data being encoded. Defaults to "image".
        """
        if text_encoder is None:
            text_encoder = self.text_encoder

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:

            text_inputs = text_encoder.text2tokens(prompt, data_type=data_type, max_length=self.text_len)
            if clip_skip is None:
                prompt_outputs = text_encoder.encode(
                    text_inputs, data_type=data_type, device=device
                )
                prompt_embeds = prompt_outputs.hidden_state
            else:
                prompt_outputs = text_encoder.encode(
                    text_inputs,
                    output_hidden_states=True,
                    data_type=data_type,
                    device=device,
                )
                # Access the `hidden_states` first, that contains a tuple of
                # all the hidden states from the encoder layers. Then index into
                # the tuple to access the hidden states from the desired layer.
                prompt_embeds = prompt_outputs.hidden_states_list[-(clip_skip + 1)]
                # We also need to apply the final LayerNorm here to not mess with the
                # representations. The `last_hidden_states` that we typically use for
                # obtaining the final prompt representations passes through the LayerNorm
                # layer.
                prompt_embeds = text_encoder.model.text_model.final_layer_norm(
                    prompt_embeds
                )

            attention_mask = prompt_outputs.attention_mask
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
                bs_embed, seq_len = attention_mask.shape
                attention_mask = attention_mask.repeat(1, num_videos_per_prompt)
                attention_mask = attention_mask.view(
                    bs_embed * num_videos_per_prompt, seq_len
                )

        if text_encoder is not None:
            prompt_embeds_dtype = text_encoder.dtype
        elif self.transformer is not None:
            prompt_embeds_dtype = self.transformer.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        if prompt_embeds.ndim == 2:
            bs_embed, _ = prompt_embeds.shape
            # duplicate text embeddings for each generation per prompt, using mps friendly method
            prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt)
            prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, -1)
        else:
            bs_embed, seq_len, _ = prompt_embeds.shape
            # duplicate text embeddings for each generation per prompt, using mps friendly method
            prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(
                bs_embed * num_videos_per_prompt, seq_len, -1
            )

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            uncond_input = text_encoder.text2tokens(uncond_tokens, data_type=data_type, max_length=self.text_len)

            negative_prompt_outputs = text_encoder.encode(uncond_input, data_type=data_type, is_uncond=True)
            negative_prompt_embeds = negative_prompt_outputs.hidden_state

            negative_attention_mask = negative_prompt_outputs.attention_mask
            if negative_attention_mask is not None:
                negative_attention_mask = negative_attention_mask.to(device)
                _, seq_len = negative_attention_mask.shape
                negative_attention_mask = negative_attention_mask.repeat(1, num_videos_per_prompt)
                negative_attention_mask = negative_attention_mask.view(batch_size * num_videos_per_prompt, seq_len)

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            if negative_prompt_embeds.ndim == 2:
                negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_videos_per_prompt)
                negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_videos_per_prompt, -1)
            else:
                negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_videos_per_prompt, 1)
                negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return (
            prompt_embeds,
            negative_prompt_embeds,
            attention_mask,
            negative_attention_mask,
        )


    def prepare_extra_func_kwargs(self, func, kwargs):
        """
        Prepare extra keyword arguments for scheduler functions.

        Filters kwargs to only include parameters that the function accepts.
        This is useful since not all schedulers have the same signature.
        """
        extra_step_kwargs = {}

        for k, v in kwargs.items():
            accepts = k in set(inspect.signature(func).parameters.keys())
            if accepts:
                extra_step_kwargs[k] = v
        return extra_step_kwargs


    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        latent_height,
        latent_width,
        video_length,
        dtype,
        device,
        generator,
        latents=None,
    ):
        """
        Prepare latents for video generation.

        Args:
            batch_size: Batch size for generation.
            num_channels_latents: Number of channels in latent space.
            latent_height: Height of latent tensors.
            latent_width: Width of latent tensors.
            video_length: Number of frames in the video.
            dtype: Data type for latents.
            device: Target device for latents.
            generator: Random number generator.
            latents: Pre-computed latents. If None, random latents are generated.

        Returns:
            torch.Tensor: Prepared latents tensor.
        """
        shape = (
            batch_size,
            num_channels_latents,
            video_length,
            latent_height,
            latent_width,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = torch.randn(shape, generator=generator, device=torch.device('cpu'), dtype=dtype).to(device)
        else:
            latents = latents.to(device)

        # Check existence to make it compatible with FlowMatchEulerDiscreteScheduler
        if hasattr(self.scheduler, "init_noise_sigma"):
            # scale the initial noise by the standard deviation required by the scheduler
            latents = latents * self.scheduler.init_noise_sigma
        return latents

    # Copied from diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img.LatentConsistencyModelPipeline.get_guidance_scale_embedding
    def get_guidance_scale_embedding(
        self,
        w: torch.Tensor,
        embedding_dim: int = 512,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            w (`torch.Tensor`):
                Generate embedding vectors with a specified guidance scale to subsequently enrich timestep embeddings.
            embedding_dim (`int`, *optional*, defaults to 512):
                Dimension of the embeddings to generate.
            dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                Data type of the generated embeddings.

        Returns:
            `torch.Tensor`: Embedding vectors with shape `(len(w), embedding_dim)`.
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    @property
    def clip_skip(self):
        return self._clip_skip

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale is not None and self._guidance_scale > 1

    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    @staticmethod
    def get_byt5_text_tokens(byt5_tokenizer, byt5_max_length, text_prompt):
        """
        Tokenize text prompt for byT5 model.

        Args:
            byt5_tokenizer: The byT5 tokenizer.
            byt5_max_length: Maximum sequence length for tokenization.
            text_prompt: Text prompt string to tokenize.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - input_ids: Tokenized input IDs.
                - attention_mask: Attention mask tensor.
        """
        byt5_text_inputs = byt5_tokenizer(
            text_prompt,
            padding="max_length",
            max_length=byt5_max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        return byt5_text_inputs.input_ids, byt5_text_inputs.attention_mask

    def _extract_glyph_texts(self, prompt):
        """
        Extract glyph texts from prompt using regex pattern.

        Args:
            prompt: Input prompt string containing quoted text.

        Returns:
            List[str]: List of extracted glyph texts (deduplicated if multiple).
        """
        pattern = r'\"(.*?)\"|“(.*?)”'
        matches = re.findall(pattern, prompt)
        result = [match[0] or match[1] for match in matches]
        result = list(dict.fromkeys(result)) if len(result) > 1 else result
        return result

    def _process_single_byt5_prompt(self, prompt_text, device):
        """
        Process a single prompt for byT5 encoding.

        Args:
            prompt_text: The prompt text to process.
            device: Target device for tensors.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - byt5_embeddings: Encoded embeddings tensor.
                - byt5_mask: Attention mask tensor.
        """
        byt5_embeddings = torch.zeros((1, self.byt5_max_length, 1472), device=device)
        byt5_mask = torch.zeros((1, self.byt5_max_length), device=device, dtype=torch.int64)
        
        glyph_texts = self._extract_glyph_texts(prompt_text)
        
        if len(glyph_texts) > 0:
            text_styles = [{'color': None, 'font-family': None} for _ in range(len(glyph_texts))]
            formatted_text = self.prompt_format.format_prompt(glyph_texts, text_styles)
            
            text_ids, text_mask = self.get_byt5_text_tokens(
                self.byt5_tokenizer, self.byt5_max_length, formatted_text
            )
            text_ids = text_ids.to(device=device)
            text_mask = text_mask.to(device=device)
            
            byt5_outputs = self.byt5_model(text_ids, attention_mask=text_mask.float())
            byt5_embeddings = byt5_outputs[0]
            byt5_mask = text_mask
            
        return byt5_embeddings, byt5_mask

    def _prepare_byt5_embeddings(self, prompts, device):
        """
        Prepare byT5 embeddings for both positive and negative prompts.

        Args:
            prompts: List of prompt strings or single prompt string.
            device: Target device for tensors.

        Returns:
            dict: Dictionary containing:
                - "byt5_text_states": Combined embeddings tensor.
                - "byt5_text_mask": Combined attention mask tensor.
                Returns empty dict if glyph_byT5_v2 is disabled.
        """
        if not self.config.glyph_byT5_v2:
            return {}
            
        if isinstance(prompts, str):
            prompt_list = [prompts]
        elif isinstance(prompts, list):
            prompt_list = prompts
        else:
            raise ValueError("prompts must be str or list of str")

        positive_embeddings = []
        positive_masks = []
        negative_embeddings = []
        negative_masks = []

        for prompt in prompt_list:
            pos_emb, pos_mask = self._process_single_byt5_prompt(prompt, device)
            positive_embeddings.append(pos_emb)
            positive_masks.append(pos_mask)

            if self.do_classifier_free_guidance:
                neg_emb, neg_mask = self._process_single_byt5_prompt("", device)
                negative_embeddings.append(neg_emb)
                negative_masks.append(neg_mask)

        byt5_positive = torch.cat(positive_embeddings, dim=0)
        byt5_positive_mask = torch.cat(positive_masks, dim=0)
        
        if self.do_classifier_free_guidance:
            byt5_negative = torch.cat(negative_embeddings, dim=0)
            byt5_negative_mask = torch.cat(negative_masks, dim=0)
            
            byt5_embeddings = torch.cat([byt5_negative, byt5_positive], dim=0)
            byt5_masks = torch.cat([byt5_negative_mask, byt5_positive_mask], dim=0)
        else:
            byt5_embeddings = byt5_positive
            byt5_masks = byt5_positive_mask

        return {
            "byt5_text_states": byt5_embeddings,
            "byt5_text_mask": byt5_masks
        }

    def extract_image_features(self, reference_image):
        """
        Extract features from a reference image using VisionEncoder.

        Args:
            reference_image: numpy array of shape (H, W, 3) with dtype uint8.

        Returns:
            VisionEncoderModelOutput: Encoded image features.
        """
        assert isinstance(reference_image, np.ndarray)
        assert reference_image.ndim == 3 and reference_image.shape[2] == 3
        assert reference_image.dtype == np.uint8

        image_encoder_output = self.vision_encoder.encode_images(reference_image)

        return image_encoder_output

    def _prepare_vision_states(self, reference_image, target_resolution, latents, device):
        """
        Prepare vision states for multitask training.

        Args:
            reference_image: Reference image for i2v tasks (None for t2v tasks).
            target_resolution: Target size for i2v tasks.
            latents: Latent tensors.
            device: Target device.

        Returns:
            torch.Tensor or None: Vision states tensor or None if vision encoder is unavailable.
        """
        if reference_image is None:
            vision_states = torch.zeros(latents.shape[0], self.config.vision_num_semantic_tokens, self.config.vision_states_dim).to(latents.device)
        else:
            reference_image = np.array(reference_image) if isinstance(reference_image, Image.Image) else reference_image
            if len(reference_image.shape) == 4:
                reference_image = reference_image[0]

            height, width = self.get_closest_resolution_given_reference_image(reference_image, target_resolution)

            # Encode reference image to vision states
            if self.vision_encoder is not None:
                input_image_np = resize_and_center_crop(reference_image, target_width=width, target_height=height)
                vision_states = self.vision_encoder.encode_images(input_image_np)
                vision_states = vision_states.last_hidden_state.to(device=device, dtype=self.target_dtype)
            else:
                vision_states = None
        
        # Repeat image features for batch size if needed (for classifier-free guidance)
        if self.do_classifier_free_guidance and vision_states is not None:
            vision_states = vision_states.repeat(2, 1, 1)
        
        return vision_states

    def _prepare_cond_latents(self, task_type, cond_latents, latents, multitask_mask):
        """
        Prepare conditional latents and mask for multitask training.

        Args:
            task_type: Type of task ("i2v" or "t2v").
            cond_latents: Conditional latents tensor.
            latents: Main latents tensor.
            multitask_mask: Multitask mask tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - latents_concat: Concatenated conditional latents.
                - mask_concat: Concatenated mask tensor.
        """
        latents_concat = None
        mask_concat = None
        
        if cond_latents is not None and task_type == 'i2v':
            latents_concat = cond_latents.repeat(1, 1, latents.shape[2], 1, 1)
            latents_concat[:, :, 1:, :, :] = 0.0
        else:
            latents_concat = torch.zeros(latents.shape[0], latents.shape[1], latents.shape[2], latents.shape[3], latents.shape[4]).to(latents.device)
        
        mask_zeros = torch.zeros(latents.shape[0], 1, latents.shape[2], latents.shape[3], latents.shape[4])
        mask_ones = torch.ones(latents.shape[0], 1, latents.shape[2], latents.shape[3], latents.shape[4])
        mask_concat = merge_tensor_by_mask(mask_zeros.cpu(), mask_ones.cpu(), mask=multitask_mask.cpu(), dim=2).to(device=latents.device)

        cond_latents = torch.concat([latents_concat, mask_concat], dim=1)
        
        return cond_latents

    def get_task_mask(self, task_type, latent_target_length):
        if task_type == "t2v":
            mask = torch.zeros(latent_target_length)
        elif task_type == "i2v":
            mask = torch.zeros(latent_target_length)
            mask[0] = 1.0
        else:
            raise ValueError(f"{task_type} is not supported !")
        return mask

    def get_closest_resolution_given_reference_image(self, reference_image, target_resolution):
        """
        Get closest supported resolution for a reference image.

        Args:
            reference_image: PIL Image or numpy array.
            target_resolution: Target resolution string (e.g., "720p", "1080p").

        Returns:
            tuple[int, int]: (height, width) of closest supported resolution.
        """
        assert reference_image is not None

        if isinstance(reference_image, Image.Image):
            origin_size = reference_image.size
        elif isinstance(reference_image, np.ndarray):
            H, W, C = reference_image.shape
            origin_size = (W, H)
        else:
            raise ValueError(f"Unsupported reference_image type: {type(reference_image)}. Must be PIL Image or numpy array")

        return self.get_closest_resolution_given_original_size(origin_size, target_resolution)


    def get_closest_resolution_given_original_size(self, origin_size, target_size):
        """
        Get closest supported resolution for given original size and target resolution.

        Args:
            origin_size: Tuple of (width, height) of original image.
            target_size: Target resolution string (e.g., "720p", "1080p").

        Returns:
            tuple[int, int]: (height, width) of closest supported resolution.
        """
        bucket_hw_base_size = self.target_size_config[target_size]["bucket_hw_base_size"]
        bucket_hw_bucket_stride = self.target_size_config[target_size]["bucket_hw_bucket_stride"]

        assert bucket_hw_base_size in [128, 256, 480, 512, 640, 720, 960, 1440], \
            f"bucket_hw_base_size must be in [128, 256, 480, 512, 640, 720, 960, 1440], but got {bucket_hw_base_size}"

        crop_size_list = generate_crop_size_list(bucket_hw_base_size, bucket_hw_bucket_stride)
        aspect_ratios = np.array([round(float(h) / float(w), 5) for h, w in crop_size_list])
        closest_size, closest_ratio = get_closest_ratio(origin_size[1], origin_size[0], aspect_ratios, crop_size_list)

        height = closest_size[0]
        width = closest_size[1]

        return height, width

    def get_image_condition_latents(self, task_type, reference_image, height, width):

        if task_type == "t2v":
            cond_latents = None

        elif task_type == "i2v":
            origin_size = reference_image.size
            
            target_height, target_width = height, width
            original_width, original_height = origin_size
            
            scale_factor = max(target_width / original_width, target_height / original_height)
            resize_width = int(round(original_width * scale_factor))
            resize_height = int(round(original_height * scale_factor))
            
            ref_image_transform = transforms.Compose([
                transforms.Resize((resize_height, resize_width),
                                  interpolation=transforms.InterpolationMode.LANCZOS),
                transforms.CenterCrop((target_height, target_width)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
            
            ref_images_pixel_values = ref_image_transform(reference_image).unsqueeze(0).unsqueeze(2).to(self.execution_device)
            
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                cond_latents = self.vae.encode(ref_images_pixel_values).latent_dist.mode()
                cond_latents.mul_(self.vae.config.scaling_factor)
            
        else:
            raise ValueError(f"Unsupported task_type: {task_type}. Must be 't2v' or 'i2v'")
        
        return cond_latents

    @property
    def vae_spatial_compression_ratio(self):
        if hasattr(self.vae.config, "ffactor_spatial"):
            return self.vae.config.ffactor_spatial
        else:
            return 16

    @property
    def vae_temporal_compression_ratio(self):
        if hasattr(self.vae.config, "ffactor_temporal"):
            return self.vae.config.ffactor_temporal
        else:
            return 4

    def get_latent_size(self, video_length, height, width):
        spatial_compression_ratio = self.vae_spatial_compression_ratio
        temporal_compression_ratio = self.vae_temporal_compression_ratio
        video_length = (video_length - 1) // temporal_compression_ratio + 1
        height, width = height // spatial_compression_ratio, width // spatial_compression_ratio

        assert height > 0 and width > 0 and video_length > 0, f"height: {height}, width: {width}, video_length: {video_length}"

        return video_length, height, width


    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        aspect_ratio: str,
        video_length: int,
        prompt_rewrite: bool = True,
        num_inference_steps: int = None,
        guidance_scale: Optional[float] = None,
        enable_sr: bool = True,
        sr_num_inference_steps: Optional[int] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        seed: Optional[int] = None,
        flow_shift: Optional[float] = None,
        embedded_guidance_scale: Optional[float] = None,
        reference_image=None,  # For i2v tasks: PIL Image or path to image file
        output_type: Optional[str] = "pt",
        return_dict: bool = True,
        return_pre_sr_video: bool = False,
        enable_vae_tile_parallelism: bool = True,
        **kwargs,
    ):
        r"""
        Generates a video (or videos) based on text (and optionally image) conditions.

        Args:
            prompt (`str` or `List[str]`):
                Text prompt(s) to guide video generation.
            aspect_ratio (`str`):
                Output video aspect ratio as a string formatted like "720:1280" or "16:9". Required for text-to-video tasks.
            video_length (`int`):
                Number of frames in the generated video.
            num_inference_steps (`int`, *optional*, defaults to 50):
                Number of denoising steps during generation. Larger values may improve video quality at the expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to value in config):
                Scale to encourage the model to better follow the prompt. `guidance_scale > 1` enables classifier-free guidance.
            enable_sr (`bool`, *optional*, defaults to True):
                Whether to apply super-resolution to the generated video.
            sr_num_inference_steps (`int`, *optional*, defaults to 30):
                Number of inference steps in the super-resolution module (if enabled).
            negative_prompt (`str` or `List[str]`, *optional*):
                Negative prompt(s) that describe what should NOT be shown in the generated video.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                PyTorch random generator(s) for deterministic results.
            seed (`int`, *optional*):
                If specified, used to create the generator for reproducible sampling.
            flow_shift (`float`, *optional*):
                Flow shift parameter for the scheduler. Overrides the default pipeline configuration if provided.
            embedded_guidance_scale (`float`, *optional*):
                Additional control guidance scale, if supported.
            reference_image (PIL.Image or `str`, *optional*):
                Reference image for image-to-video (i2v) tasks. Can be a PIL image or a path to an image file. Set to `None` for text-to-video (t2v) generation.
            output_type (`str`, *optional*, defaults to "pt"):
                Output format of the returned video(s). Accepted values: `"pt"` for torch.Tensor or `"np"` for numpy.ndarray.
            return_dict (`bool`, *optional*, defaults to True):
                Whether to return a [`HunyuanVideoPipelineOutput`] or a tuple.
            **kwargs:
                Additional keyword arguments.

        Returns:
            HunyuanVideoPipelineOutput or `tuple`:
                If `return_dict` is True, returns a [`HunyuanVideoPipelineOutput`] with fields:
                    - `videos`: Generated video(s) as a tensor or numpy array.
                    - `sr_videos`: Super-resolved video(s) if `enable_sr` is True, else None.
                Otherwise, returns a tuple containing the outputs as above.

        Example:
            ```python
            pipe = HunyuanVideoPipeline.from_pretrained("your_model_dir")
            # Text-to-video
            video = pipe(prompt="A dog surfing on the beach", aspect_ratio="9:16", video_length=32).videos
            # Image-to-video
            video = pipe(prompt="Make this image move", reference_image="img.jpg", aspect_ratio="16:9", video_length=24).videos
            ```
        """
        num_videos_per_prompt = 1
        target_resolution = self.ideal_resolution

        if guidance_scale is None:
            guidance_scale = self.config.guidance_scale
        if embedded_guidance_scale is None:
            embedded_guidance_scale = self.config.embedded_guidance_scale
        if flow_shift is None:
            flow_shift = self.config.flow_shift
        if num_inference_steps is None:
            num_inference_steps = self.config.num_inference_steps

        if embedded_guidance_scale is not None:
            assert not self.do_classifier_free_guidance
            assert self.transformer.config.guidance_embed
        else:
            assert not self.transformer.config.guidance_embed

        user_reference_image = reference_image
        user_prompt = prompt

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

        if prompt_rewrite:
            from hyvideo.utils.rewrite.rewrite_utils import run_prompt_rewrite

            if not dist.is_initialized() or get_parallel_state().sp_rank == 0:
                try:
                    prompt = run_prompt_rewrite(user_prompt, reference_image, task_type)
                except Exception as e:
                    loguru.logger.warning(f"Failed to rewrite prompt: {e}")
                    prompt = user_prompt
                
            if dist.is_initialized() and get_parallel_state().sp_enabled:
                obj_list = [prompt]
                # not use group_src to support old PyTorch
                group_src_rank = dist.get_global_rank(get_parallel_state().sp_group, 0)
                dist.broadcast_object_list(obj_list, src=group_src_rank, group=get_parallel_state().sp_group)
                prompt = obj_list[0]

        if self.ideal_task is not None and self.ideal_task != task_type:
            raise ValueError(
                f"The loaded pipeline is trained for '{self.ideal_task}' task, but received input for '{task_type}' task. "
                "Please load a pipeline trained for the correct task, or check and update your arguments accordingly."
            )

        if flow_shift is None:
            self.scheduler = self._create_scheduler(self.config.flow_shift)
        else:
            self.scheduler = self._create_scheduler(flow_shift)

        if seed is None or seed == -1:
            seed = random.randint(100000, 999999)

        if get_parallel_state().sp_enabled:
            assert seed is not None
            if dist.is_initialized():
                obj_list = [seed]
                group_src_rank = dist.get_global_rank(get_parallel_state().sp_group, 0)
                dist.broadcast_object_list(obj_list, src=group_src_rank, group=get_parallel_state().sp_group)
                seed = obj_list[0]

        if generator is None and seed is not None:
            generator = torch.Generator(device=torch.device('cpu')).manual_seed(seed)

        if reference_image is not None:
            if self.ideal_resolution is not None and target_resolution != self.ideal_resolution:
                raise ValueError(
                    f'The loaded pipeline is trained for {self.ideal_resolution} resolution, but received input for {target_resolution} resolution. '
                )
            height, width = self.get_closest_resolution_given_reference_image(reference_image, target_resolution)
        else:
            if self.ideal_resolution is not None:
                if ":" not in aspect_ratio:
                    raise ValueError("aspect_ratio must be separated by a colon")
                width, height = aspect_ratio.split(":")
                # check if width and height are integers
                if not width.isdigit() or not height.isdigit() or int(width) <= 0 or int(height) <= 0:
                    raise ValueError("width and height must be positive integers and separated by a colon in aspect_ratio")
                width = int(width)
                height = int(height)
                height, width = self.get_closest_resolution_given_original_size((width, height), self.ideal_resolution)
            else:
                raise ValueError("ideal_resolution is not set")

        latent_target_length, latent_height, latent_width = self.get_latent_size(video_length, height, width)
        n_tokens = latent_target_length * latent_height * latent_width
        multitask_mask = self.get_task_mask(task_type, latent_target_length)


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
                f"🎬  HunyuanVideo Generation Task\n"
                f"{'-' * 60}\n"
                f"User Prompt:               {user_prompt}\n"
                f"Rewritten Prompt:          {prompt if prompt_rewrite else '<disabled>'}\n"
                f"Aspect Ratio:              {aspect_ratio if task_type == 't2v' else f'{width}:{height}'}\n"
                f"Video Length:              {video_length}\n"
                f"Reference Image:           {user_reference_image} {reference_image.size if reference_image is not None else ''}\n"
                f"Guidance Scale:            {guidance_scale}\n"
                f"Guidance Embedded Scale:   {embedded_guidance_scale}\n"
                f"Shift:                     {flow_shift}\n"
                f"Seed:                      {seed}\n"
                f"Video Resolution:          {width} x {height}\n"
                f'Attn mode:                 {self.transformer.attn_mode}\n'
                f"Transformer dtype:         {self.transformer.dtype}\n"
                f"Sampling Steps:            {num_inference_steps}\n"
                f"Use Meanflow:              {self.use_meanflow}\n"
                f"{'=' * 60}"
                '\n'
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


        # Encode prompts with second encoder if available
        if self.text_encoder_2 is not None:
            with auto_offload_model(self.text_encoder_2, self.execution_device, enabled=self.enable_offloading):
                (
                    prompt_embeds_2,
                    negative_prompt_embeds_2,
                    prompt_mask_2,
                    negative_prompt_mask_2,
                ) = self.encode_prompt(
                    prompt,
                    device,
                    num_videos_per_prompt,
                    self.do_classifier_free_guidance,
                    negative_prompt,
                    clip_skip=self.clip_skip,
                    text_encoder=self.text_encoder_2,
                    data_type="video",
                )
        else:
            prompt_embeds_2 = None
            negative_prompt_embeds_2 = None
            prompt_mask_2 = None
            negative_prompt_mask_2 = None

        extra_kwargs = {}
        if self.config.glyph_byT5_v2:
            with auto_offload_model(self.byt5_model, self.execution_device, enabled=self.enable_offloading):
                extra_kwargs = self._prepare_byt5_embeddings(prompt, device)

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            if prompt_mask is not None:
                prompt_mask = torch.cat([negative_prompt_mask, prompt_mask])
            if prompt_embeds_2 is not None:
                prompt_embeds_2 = torch.cat([negative_prompt_embeds_2, prompt_embeds_2])
            if prompt_mask_2 is not None:
                prompt_mask_2 = torch.cat([negative_prompt_mask_2, prompt_mask_2])

        extra_set_timesteps_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.set_timesteps, {"n_tokens": n_tokens}
        )

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            **extra_set_timesteps_kwargs,
        )

        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            latent_height,
            latent_width,
            latent_target_length,
            self.target_dtype,
            device,
            generator,
        )

        with auto_offload_model(self.vae, self.execution_device, enabled=self.enable_offloading):
            image_cond = self.get_image_condition_latents(task_type, reference_image, height, width)

        cond_latents = self._prepare_cond_latents(
            task_type, image_cond, latents, multitask_mask
        )
        with auto_offload_model(self.vision_encoder, self.execution_device, enabled=self.enable_offloading):
            vision_states = self._prepare_vision_states(
                semantic_images_np, target_resolution, latents, device
            )

        extra_step_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.step, {"generator": generator, "eta": kwargs.get("eta", 0.0)},
        )

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        cache_helper = getattr(self, 'cache_helper', None)
        if cache_helper is not None:
            cache_helper.clear_states()

        with self.progress_bar(total=num_inference_steps) as progress_bar, auto_offload_model(self.transformer, self.execution_device, enabled=self.enable_offloading):
            for i, t in enumerate(timesteps):
                if cache_helper is not None:
                    cache_helper.cur_timestep = i
                latents_concat = torch.concat([latents, cond_latents], dim=1)
                latent_model_input = torch.cat([latents_concat] * 2) if self.do_classifier_free_guidance else latents_concat

                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                t_expand = t.repeat(latent_model_input.shape[0])
                if self.use_meanflow:
                    if i == len(timesteps) - 1:
                        timesteps_r = torch.tensor([0.0], device=self.execution_device)
                    else:
                        timesteps_r = timesteps[i + 1]
                    timesteps_r = timesteps_r.repeat(latent_model_input.shape[0])
                else:
                    timesteps_r = None

                guidance_expand = (
                    torch.tensor(
                        [embedded_guidance_scale] * latent_model_input.shape[0],
                        dtype=torch.float32,
                        device=device,
                    ).to(self.target_dtype)
                    * 1000.0
                    if embedded_guidance_scale is not None
                    else None
                )

                with torch.autocast(device_type="cuda", dtype=self.target_dtype, enabled=self.autocast_enabled):
                    output = self.transformer(
                        latent_model_input,
                        t_expand,
                        prompt_embeds,
                        prompt_embeds_2,
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

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
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

        if enable_sr:
            assert hasattr(self, 'sr_pipeline')
            sr_out = self.sr_pipeline(
                prompt=prompt,
                num_inference_steps=sr_num_inference_steps,

                video_length=video_length,
                negative_prompt="",
                num_videos_per_prompt=num_videos_per_prompt,
                seed=seed,
                output_type=output_type,

                lq_latents=latents,
                reference_image=user_reference_image,
                enable_vae_tile_parallelism=enable_vae_tile_parallelism,
            )

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

            if return_pre_sr_video or not enable_sr:
                with torch.autocast(device_type="cuda", dtype=self.vae_dtype, enabled=self.vae_autocast_enabled), auto_offload_model(self.vae, self.execution_device, enabled=self.enable_offloading):
                    self.vae.enable_tiling()
                    video_frames = self.vae.decode(latents, return_dict=False, generator=generator)[0]
                    self.vae.disable_tiling()

                if video_frames is not None:
                    video_frames = (video_frames / 2 + 0.5).clamp(0, 1).cpu().float()
                    
            else:
                video_frames = sr_out.videos

        # Offload all models
        self.maybe_free_model_hooks()
        if enable_sr:
            sr_video_frames = sr_out.videos

        if not return_dict:
            ret = video_frames
            if enable_sr:
                ret = (video_frames, sr_video_frames)
            return ret

        if enable_sr:
            return HunyuanVideoPipelineOutput(videos=video_frames, sr_videos=sr_video_frames)
        else:
            return HunyuanVideoPipelineOutput(videos=video_frames)

    @property
    def ideal_resolution(self):
        return self.transformer.config.ideal_resolution

    @property
    def ideal_task(self):
        return self.transformer.config.ideal_task

    @property
    def use_meanflow(self):
        return self.transformer.config.use_meanflow

    def apply_infer_optimization(
        self,
        infer_state: Optional[InferState] = None,
        enable_offloading: bool = False,
        enable_group_offloading: bool = False,
        overlap_group_offloading: bool = True,
    ):
        """
        Apply inference optimizations to transformer based on infer_state.
        
        Args:
            infer_state: Optional InferState object containing optimization settings.
            enable_offloading: Whether to enable CPU offloading.
            enable_group_offloading: Whether to enable group offloading.
            overlap_group_offloading: Whether to use overlapping group offloading.
        """
        if infer_state is not None:

            if infer_state.enable_torch_compile:
                # block-wise compile
                for block in self.transformer.double_blocks:
                    block.forward = torch.compile(block.forward)

            # Apply sageattn if enabled
            if infer_state.enable_sageattn:
                self.transformer.set_attn_mode('sageattn')
            
            # Apply cache if enabled
            if infer_state.enable_cache:
                if not is_angelslim_available():
                    raise RuntimeError("Please install angelslim==0.2.1 via `pip install angelslim==0.2.1` to enable cache.")
                from angelslim.compressor.diffusion import DeepCacheHelper, TeaCacheHelper, TaylorCacheHelper
                no_cache_steps = list(range(0, infer_state.cache_start_step)) + list(range(infer_state.cache_start_step, infer_state.cache_end_step, infer_state.cache_step_interval)) + list(range(infer_state.cache_end_step, infer_state.total_steps))
                cache_type = infer_state.cache_type
                if cache_type == 'deepcache':
                    no_cache_block_id = {"double_blocks": infer_state.no_cache_block_id}
                    self.cache_helper = DeepCacheHelper(
                        double_blocks=self.transformer.double_blocks,
                        no_cache_steps=no_cache_steps,
                        no_cache_block_id=no_cache_block_id,
                    )
                elif cache_type == 'teacache':
                    self.cache_helper = TeaCacheHelper(
                        double_blocks=self.transformer.double_blocks,
                        no_cache_steps=no_cache_steps,
                    )
                elif cache_type == 'taylorcache':
                    self.cache_helper = TaylorCacheHelper(
                        double_blocks=self.transformer.double_blocks,
                        no_cache_steps=no_cache_steps,
                    )
                else:
                    raise ValueError(f"Unknown cache type: {cache_type}. Only 'deepcache', 'teacache', 'taylorcache' are supported.")
                self.cache_helper.enable()
            else:
                self.cache_helper = None
        else:
            self.cache_helper = None
        
        # Set enable_offloading
        self.enable_offloading = enable_offloading
        
        # Apply group offloading if enabled
        if enable_group_offloading:
            assert enable_offloading, "enable_group_offloading requires enable_offloading to be True"
            group_offloading_kwargs = {
                'onload_device': torch.device('cuda'),
                'num_blocks_per_group': 1 if overlap_group_offloading else 4,
            }
            if overlap_group_offloading:
                group_offloading_kwargs['use_stream'] = True
            self.transformer.enable_group_offload(**group_offloading_kwargs)

    @classmethod
    def load_sr_transformer_upsampler(cls, cached_folder, sr_version, transformer_dtype=torch.bfloat16, device=None):
        transformer = HunyuanVideo_1_5_DiffusionTransformer.from_pretrained(
            os.path.join(cached_folder, "transformer", sr_version), 
            low_cpu_mem_usage=True, 
            torch_dtype=transformer_dtype
        ).to(device)
        upsampler_cls = SRTo720pUpsampler if "720p" in sr_version else SRTo1080pUpsampler
        upsampler = upsampler_cls.from_pretrained(os.path.join(cached_folder, "upsampler", sr_version)).to(device)
        return transformer, upsampler

    def create_sr_pipeline(self, cached_folder, sr_version, transformer_dtype=torch.bfloat16, device=None):
        from .hunyuan_video_sr_pipeline import HunyuanVideo_1_5_SR_Pipeline


        transformer, upsampler = self.load_sr_transformer_upsampler(cached_folder, sr_version, transformer_dtype=transformer_dtype, device=device)

        return HunyuanVideo_1_5_SR_Pipeline(
            vae=self.vae,
            transformer=transformer,
            text_encoder=self.text_encoder,
            scheduler=self.scheduler,
            upsampler=upsampler,
            text_encoder_2=self.text_encoder_2,
            progress_bar_config=None,
            glyph_byT5_v2=self.config.glyph_byT5_v2,
            byt5_model=self.byt5_model,
            byt5_tokenizer=self.byt5_tokenizer,
            byt5_max_length=self.byt5_max_length,
            prompt_format=self.prompt_format,
            execution_device='cuda',
            vision_encoder=self.vision_encoder,
            enable_offloading=self.enable_offloading,
            **SR_PIPELINE_CONFIGS[sr_version],
        )

    @staticmethod
    def get_transformer_version(resolution, task, cfg_distilled=False, step_distilled=False, sparse_attn=False):
        # Build transformer_version based on flags
        # Note: sparse attention requires distilled model, so if sparse_attn is enabled,
        # we automatically include distilled in the version string
        transformer_version = f'{resolution}_{task}'
        if step_distilled:
            transformer_version += '_step_distilled'
        else:
            if cfg_distilled or sparse_attn:
                transformer_version += '_distilled'
            if sparse_attn:
                transformer_version += '_sparse'
        return transformer_version

    @classmethod
    def create_pipeline(cls, pretrained_model_name_or_path, transformer_version, create_sr_pipeline=False, transformer_dtype=torch.bfloat16, device=None, transformer_init_device=None, **kwargs):
        # use snapshot download here to get it working from from_pretrained

        if not os.path.isdir(pretrained_model_name_or_path):
            if pretrained_model_name_or_path.count("/") > 1:
                raise ValueError(
                    f'The provided pretrained_model_name_or_path "{pretrained_model_name_or_path}"'
                    " is neither a valid local path nor a valid repo id. Please check the parameter."
                )
            cached_folder = cls.download(
                pretrained_model_name_or_path,
                **kwargs,
            )
        else:
            cached_folder = pretrained_model_name_or_path

        if device is None:
            device = torch.device('cuda')
        
        if transformer_init_device is None:
            transformer_init_device = device

        supported_transformer_version = os.listdir(os.path.join(cached_folder, "transformer"))
        if transformer_version not in supported_transformer_version:
            error_msg = (
                f"Could not find {transformer_version} in {cached_folder}. "
                f"Only {supported_transformer_version} are available.\n"
                f"Trying to download from https://huggingface.co/tencent/HunyuanVideo-1.5. "
            )
            loguru.logger.warning(error_msg)
            from_pretrain_kwargs ={
                'pretrained_model_name_or_path': 'tencent/HunyuanVideo-1.5',
                'subfolder': f'transformer/{transformer_version}',
            }
        else:
            from_pretrain_kwargs ={
                'pretrained_model_name_or_path': os.path.join(cached_folder, "transformer", transformer_version),
            }

        vae_inference_config = cls.get_vae_inference_config()
        transformer = HunyuanVideo_1_5_DiffusionTransformer.from_pretrained(
            **from_pretrain_kwargs,
            torch_dtype=transformer_dtype, 
            low_cpu_mem_usage=True,
        ).to(transformer_init_device)

        vae = hunyuanvideo_15_vae.AutoencoderKLConv3D.from_pretrained(
            os.path.join(cached_folder, "vae"), 
            torch_dtype=vae_inference_config['dtype']
        ).to(device)
        vae.set_tile_sample_min_size(vae_inference_config['sample_size'], vae_inference_config['tile_overlap_factor'])
        scheduler = FlowMatchDiscreteScheduler.from_pretrained(os.path.join(cached_folder, "scheduler"))

        byt5_kwargs, prompt_format = cls._load_byt5(cached_folder, True, 256, device=device)
        text_encoder, text_encoder_2 = cls._load_text_encoders(cached_folder, device=device)
        vision_encoder = cls._load_vision_encoder(cached_folder, device=device)

        pipeline = cls(
            vae=vae,
            text_encoder=text_encoder,
            transformer=transformer,
            scheduler=scheduler,
            text_encoder_2=text_encoder_2,
            progress_bar_config=None,
            byt5_model=byt5_kwargs["byt5_model"],
            byt5_tokenizer=byt5_kwargs["byt5_tokenizer"],
            byt5_max_length=byt5_kwargs["byt5_max_length"],
            prompt_format=prompt_format,
            execution_device='cuda',
            vision_encoder=vision_encoder,
            enable_offloading=False,
            **PIPELINE_CONFIGS[transformer_version],
        )

        if create_sr_pipeline:
            sr_version = TRANSFORMER_VERSION_TO_SR_VERSION[transformer_version]
            sr_pipeline = pipeline.create_sr_pipeline(cached_folder, sr_version, transformer_dtype=transformer_dtype, device=device)
            pipeline.sr_pipeline = sr_pipeline


        return pipeline

    @staticmethod
    def get_offloading_config(memory_limitation=None):
        if memory_limitation is None:
            memory_limitation = get_gpu_memory()
        GB = 1024 * 1024 * 1024
        if memory_limitation < 60 * GB:
            return {
                'enable_offloading': True,
                'enable_group_offloading': True,
            }
        else:
            return {
                'enable_offloading': True,
                'enable_group_offloading': False,
            }

    @staticmethod
    def get_vae_inference_config(memory_limitation=None):
        if memory_limitation is None:
            memory_limitation = get_gpu_memory()
        GB = 1024 * 1024 * 1024
        if memory_limitation > 21 * GB:
            sample_size = 256
            tile_overlap_factor = 0.125
            dtype = torch.float16
        else:
            sample_size = 128
            tile_overlap_factor = 0.25
            dtype = torch.float16
        return {'sample_size': sample_size, 'tile_overlap_factor': tile_overlap_factor, 'dtype': dtype}


    @classmethod
    def _load_text_encoders(cls, pretrained_model_path, device):
        text_encoder_path = f'{pretrained_model_path}/text_encoder/llm'
        if not os.path.exists(text_encoder_path):
            msg = f"{text_encoder_path} not found. Please refer to checkpoints-download.md to download the text encoder checkpoints."
            loguru.logger.error(msg)
            raise FileNotFoundError(msg)
        text_encoder = TextEncoder(
            text_encoder_type='llm',
            tokenizer_type='llm',
            text_encoder_path=text_encoder_path,
            max_length=1000,
            text_encoder_precision="fp16",
            prompt_template=PROMPT_TEMPLATE['li-dit-encode-image-json'],
            prompt_template_video=PROMPT_TEMPLATE['li-dit-encode-video-json'],
            hidden_state_skip_layer=2,
            apply_final_norm=False,
            reproduce=False,
            logger=loguru.logger,
            device=device,
        )
        text_encoder_2 = None

        return text_encoder, text_encoder_2

    @classmethod
    def _load_vision_encoder(cls, pretrained_model_name_or_path, device):
        vision_encoder_path = f'{pretrained_model_name_or_path}/vision_encoder/siglip'
        if not os.path.exists(vision_encoder_path):
            msg = f"{vision_encoder_path} not found. Please refer to checkpoints-download.md to download the vision encoder checkpoints."
            loguru.logger.error(msg)
            raise FileNotFoundError(msg)
        vision_encoder = VisionEncoder(
            vision_encoder_type="siglip",
            vision_encoder_precision="fp16",
            vision_encoder_path=vision_encoder_path,
            processor_type=None,
            processor_path=None,
            output_key=None,
            logger=logger,
            device=device,
        )
        return vision_encoder
