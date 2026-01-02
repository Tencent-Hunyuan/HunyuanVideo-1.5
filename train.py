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

"""
HunyuanVideo-1.5 Training Script

This script provides a complete training pipeline for HunyuanVideo-1.5 model.

Quick Start:
1. Implement your own dataloader:
   - Replace the `create_dummy_dataloader()` function with your own implementation
   - Your dataset's __getitem__ method should return a single sample:
     * "pixel_values": torch.Tensor - Video: [C, F, H, W] or Image: [C, H, W]
       Pixel values must be in range [-1, 1] 
       Note: For video data, temporal dimension F must be 4n+1 (e.g., 1, 5, 9, 13, 17, ...)
     * "text": str - Text prompt for this sample
     * "data_type": str - "video" or "image"
     * Optional: "latents" - Pre-encoded VAE latents for faster training
     * Optional: "byt5_text_ids" and "byt5_text_mask" - Pre-tokenized byT5 inputs
   - See `create_dummy_dataloader()` function for detailed format documentation

2. Configure training parameters:
   - Set `--pretrained_model_root` to your pretrained model path
   - Adjust training hyperparameters (learning_rate, batch_size, etc.)
   - Configure distributed training settings (sp_size, enable_fsdp, etc.)

3. Run training:
   - Single GPU: python train.py --pretrained_model_root <path> [other args]
   - Multi-GPU: torchrun --nproc_per_node=N train.py --pretrained_model_root <path> [other args]

4. Monitor training:
   - Checkpoints are saved to `output_dir` at intervals specified by `--save_interval`
   - Validation videos are generated at intervals specified by `--validation_interval`
   - Training logs are printed to console at intervals specified by `--log_interval`

5. Resume training:
   - Use `--resume_from_checkpoint <checkpoint_dir>` to resume from a saved checkpoint

For detailed format requirements, see the docstring of `create_dummy_dataloader()` function.
"""

import os
import random
import math
import argparse
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from enum import Enum

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
)
from diffusers.optimization import get_scheduler
from loguru import logger
import einops
import imageio

from hyvideo.pipelines.hunyuan_video_pipeline import HunyuanVideo_1_5_Pipeline
from hyvideo.commons.parallel_states import get_parallel_state, initialize_parallel_state
from hyvideo.optim.muon import get_muon_optimizer

from torch.distributed._composable.fsdp import (
    MixedPrecisionPolicy,
    fully_shard,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)


class SNRType(str, Enum):
    UNIFORM = "uniform"
    LOGNORM = "lognorm"
    MIX = "mix"
    MODE = "mode"


def str_to_bool(value):
    """Convert string to boolean, supporting true/false, 1/0, yes/no.
    If value is None (when flag is provided without value), returns True."""
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = value.lower().strip()
        if value in ('true', '1', 'yes', 'on'):
            return True
        elif value in ('false', '0', 'no', 'off'):
            return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {value}")


def save_video(video: torch.Tensor, path: str):
    if video.ndim == 5:
        assert video.shape[0] == 1, f"Expected batch size 1, got {video.shape[0]}"
        video = video[0]
    vid = (video * 255).clamp(0, 255).to(torch.uint8)
    vid = einops.rearrange(vid, 'c f h w -> f h w c')
    imageio.mimwrite(path, vid.cpu().numpy(), fps=24)


@dataclass
class TrainingConfig:
    # Model paths
    pretrained_model_root: str
    pretrained_transformer_version: str = "720p_t2v"
    
    # Training parameters
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    max_steps: int = 10000
    warmup_steps: int = 500
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    use_muon: bool = True
    
    # Diffusion parameters
    num_train_timesteps: int = 1000
    train_timestep_shift: float = 3.0
    validation_timestep_shift: float = 5.0
    snr_type: SNRType = SNRType.LOGNORM  # Timestep sampling strategy: uniform, lognorm, mix, or mode
    
    # Task configuration
    task_type: str = "t2v"  # "t2v" or "i2v"
    i2v_prob: float = 0.3  # Probability of using i2v task when data_type is video (default: 0.3 for video training)
    
    # FSDP configuration
    enable_fsdp: bool = True  # Enable FSDP for distributed training
    enable_gradient_checkpointing: bool = True  # Enable gradient checkpointing
    sp_size: int = 8  # Sequence parallelism size (must divide world_size evenly)
    dp_replicate: int = 1  # Data parallelism replicate size (must divide world_size evenly)
    
    # Data configuration
    batch_size: int = 1
    num_workers: int = 4
    
    # Output configuration
    output_dir: str = "./outputs"
    save_interval: int = 1000
    log_interval: int = 10
    
    # Device configuration
    dtype: str = "bf16"  # "bf16" or "fp32"
    
    # Seed
    seed: int = 42
    
    # Validation configuration
    validation_interval: int = 100  # Run validation every N steps
    validation_prompts: Optional[List[str]] = None  # Prompts for validation (default: single prompt)
    validate_video_length: int = 121  # Video length (number of frames) for validation
    
    # Resume training configuration
    resume_from_checkpoint: Optional[str] = None  # Path to checkpoint directory to resume from
    
    # LoRA configuration
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: Optional[List[str]] = None  # Target modules for LoRA (default: all Linear layers)
    pretrained_lora_path: Optional[str] = None


class LinearInterpolationSchedule:
    """Simple linear interpolation schedule for flow matching"""
    def __init__(self, T: int = 1000):
        self.T = T
    
    def forward(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Linear interpolation: x_t = (1 - t/T) * x0 + (t/T) * x1
        Args:
            x0: starting point (clean latents)
            x1: ending point (noise)
            t: timesteps
        """
        t_normalized = t / self.T
        t_normalized = t_normalized.view(-1, *([1] * (x0.ndim - 1)))
        return (1 - t_normalized) * x0 + t_normalized * x1


class TimestepSampler:

    TRAIN_EPS = 1e-5
    SAMPLE_EPS = 1e-3
    
    def __init__(
        self, 
        T: int = 1000, 
        device: torch.device = None,
        snr_type: SNRType = SNRType.LOGNORM,
    ):
        self.T = T
        self.device = device
        self.snr_type = SNRType(snr_type) if isinstance(snr_type, str) else snr_type
    
    def _check_interval(self, eval: bool = False):
        # For ICPlan-like path with velocity model, use [eps, 1-eps]
        eps = self.SAMPLE_EPS if eval else self.TRAIN_EPS
        t0 = eps
        t1 = 1.0 - eps
        return t0, t1
    
    def sample(self, batch_size: int, device: torch.device = None) -> torch.Tensor:
        if device is None:
            device = self.device if self.device is not None else torch.device("cuda")
        
        t0, t1 = self._check_interval(eval=False)
        
        if self.snr_type == SNRType.UNIFORM:
            # Uniform sampling: t = rand() * (t1 - t0) + t0
            t = torch.rand((batch_size,), device=device) * (t1 - t0) + t0
            
        elif self.snr_type == SNRType.LOGNORM:
            # Log-normal sampling: t = 1 / (1 + exp(-u)) * (t1 - t0) + t0
            u = torch.normal(mean=0.0, std=1.0, size=(batch_size,), device=device)
            t = 1.0 / (1.0 + torch.exp(-u)) * (t1 - t0) + t0
            
        elif self.snr_type == SNRType.MIX:
            # Mix sampling: 30% lognorm + 70% clipped uniform
            u = torch.normal(mean=0.0, std=1.0, size=(batch_size,), device=device)
            t_lognorm = 1.0 / (1.0 + torch.exp(-u)) * (t1 - t0) + t0
            
            # Clipped uniform: delta = 0.0 (0.0~0.01 clip)
            delta = 0.0
            t0_clip = t0 + delta
            t1_clip = t1 - delta
            t_clip_uniform = torch.rand((batch_size,), device=device) * (t1_clip - t0_clip) + t0_clip
            
            # Mix with 30% lognorm, 70% uniform
            mask = (torch.rand((batch_size,), device=device) > 0.3).float()
            t = mask * t_lognorm + (1 - mask) * t_clip_uniform
            
        elif self.snr_type == SNRType.MODE:
            # Mode sampling: t = 1 - u - mode_scale * (cos(pi * u / 2)^2 - 1 + u)
            mode_scale = 1.29
            u = torch.rand(size=(batch_size,), device=device)
            t = 1.0 - u - mode_scale * (torch.cos(math.pi * u / 2.0) ** 2 - 1.0 + u)
            # Scale to [t0, t1] range
            t = t * (t1 - t0) + t0
        else:
            raise ValueError(f"Unknown SNR type: {self.snr_type}")
        
        # Scale to [0, T] range
        timesteps = t * self.T
        return timesteps


def timestep_transform(timesteps: torch.Tensor, T: int, shift: float = 1.0) -> torch.Tensor:
    """Transform timesteps with shift"""
    if shift == 1.0:
        return timesteps
    timesteps_normalized = timesteps / T
    timesteps_transformed = shift * timesteps_normalized / (1 + (shift - 1) * timesteps_normalized)
    return timesteps_transformed * T


def is_src(src, group_src, group):
    assert src is not None or group_src is not None
    assert src is None or group_src is None
    if src is not None:
        return dist.get_rank() == src
    if group_src is not None:
        return dist.get_rank() == dist.get_global_rank(group, group_src)
    raise RuntimeError("src and group_src cannot be both None")

def broadcast_object(
        obj,
        src = None,
        group = None,
        device = None,
        group_src = None,
):
    kwargs = dict(
        src=src,
        group_src=group_src,
        group=group,
        device=device,
    )
    buffer = [obj] if is_src(src, group_src, group) else [None]

    dist.broadcast_object_list(buffer, **kwargs)
    return buffer[0]

def broadcast_tensor(
        tensor,
        src  = None,
        group = None,
        async_op: bool = False,
        group_src = None,
):
    """shape and dtype safe broadcast of tensor"""
    kwargs = dict(
        src=src,
        group_src=group_src,
        group=group,
        async_op=async_op,
    )
    if is_src(src, group_src, group):
        tensor = tensor.cuda().contiguous()
    if is_src(src, group_src, group):
        shape, dtype = tensor.shape, tensor.dtype
    else:
        shape, dtype = None, None
    shape = broadcast_object(shape, src=src, group_src=group_src, group=group)
    dtype = broadcast_object(dtype, src=src, group_src=group_src, group=group)

    buffer = tensor if is_src(src, group_src, group) else torch.empty(shape, device='cuda', dtype=dtype)
    dist.broadcast(buffer, **kwargs)
    return buffer


def sync_tensor_for_sp(tensor: torch.Tensor, sp_group) -> torch.Tensor:
    """
    Sync tensor within sequence parallel group.
    Ensures all ranks in the SP group have the same tensor values.
    """
    if sp_group is None:
        return tensor
    if not isinstance(tensor, torch.Tensor):
        obj_list = [tensor]
        dist.broadcast_object_list(obj_list, group_src=0, group=sp_group)
        return obj_list[0]
    return broadcast_tensor(tensor, group_src=0, group=sp_group)


class HunyuanVideoTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if "RANK" in os.environ:
            self.rank = int(os.environ["RANK"])
            self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
            self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            self.device = torch.device(f"cuda:{self.local_rank}")
            self.is_main_process = self.rank == 0
        else:
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0
            self.is_main_process = True
        
        if config.sp_size > self.world_size:
            raise ValueError(
                f"sp_size ({config.sp_size}) cannot be greater than world_size ({self.world_size})"
            )
        if self.world_size % config.sp_size != 0:
            raise ValueError(
                f"sp_size ({config.sp_size}) must evenly divide world_size ({self.world_size}). "
                f"world_size % sp_size = {self.world_size % config.sp_size}"
            )
        
        initialize_parallel_state(sp=config.sp_size, dp_replicate=config.dp_replicate)
        torch.cuda.set_device(self.local_rank)
        self.parallel_state = get_parallel_state()
        self.dp_rank = self.parallel_state.world_mesh['dp'].get_local_rank()
        self.dp_size = self.parallel_state.world_mesh['dp'].size()
        self.sp_enabled = self.parallel_state.sp_enabled
        self.sp_group = self.parallel_state.sp_group if self.sp_enabled else None

        self._set_seed(config.seed + self.dp_rank)
        self._build_models()
        self._build_optimizer()
        
        self.noise_schedule = LinearInterpolationSchedule(T=config.num_train_timesteps)
        self.timestep_sampler = TimestepSampler(
            T=config.num_train_timesteps, 
            device=self.device,
            snr_type=config.snr_type,
        )
        
        self.global_step = 0
        self.current_epoch = 0
        
        if self.is_main_process:
            os.makedirs(config.output_dir, exist_ok=True)
        
        self.validation_output_dir = os.path.join(config.output_dir, "samples")
        if self.is_main_process:
            os.makedirs(self.validation_output_dir, exist_ok=True)
        
        if config.validation_prompts is None:
            config.validation_prompts = ["A beautiful sunset over the ocean with waves gently crashing on the shore"]
    
    def _set_seed(self, seed: int):
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def _build_models(self):
        if self.config.dtype == "bf16":
            transformer_dtype = torch.bfloat16
        elif self.config.dtype == "fp32":
            transformer_dtype = torch.float32
        else:
            raise ValueError(f"Unsupported dtype: {self.config.dtype}")
        
        # Don't create SR pipeline for training (validation uses enable_sr=False)
        self.pipeline = HunyuanVideo_1_5_Pipeline.create_pipeline(
            pretrained_model_name_or_path=self.config.pretrained_model_root,
            transformer_version=self.config.pretrained_transformer_version,
            transformer_dtype=transformer_dtype,
            enable_offloading=False,
            enable_group_offloading=False,
            overlap_group_offloading=False,
            create_sr_pipeline=False,
            flow_shift=self.config.validation_timestep_shift,
            device=self.device,
        )
        
        self.transformer = self.pipeline.transformer
        self.vae = self.pipeline.vae
        self.text_encoder = self.pipeline.text_encoder
        self.text_encoder_2 = self.pipeline.text_encoder_2
        self.vision_encoder = self.pipeline.vision_encoder
        self.byt5_kwargs = {
            "byt5_model": self.pipeline.byt5_model,
            "byt5_tokenizer": self.pipeline.byt5_tokenizer,
        }
        
        self.transformer.train()

        if self.config.use_lora:
            self._apply_lora()
        
        if self.config.enable_gradient_checkpointing:
            self._apply_gradient_checkpointing()
        
        if self.config.enable_fsdp and self.world_size > 1:
            self._apply_fsdp()
        
        if self.is_main_process:
            logger.info(f"Models loaded. Transformer dtype: {transformer_dtype}")
            total_params = sum(p.numel() for p in self.transformer.parameters())
            trainable_params = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
            logger.info(f"Transformer parameters: {total_params:,} (trainable: {trainable_params:,})")
            logger.info(f"LoRA enabled: {self.config.use_lora}")
            logger.info(f"FSDP enabled: {self.config.enable_fsdp and self.world_size > 1}")
            logger.info(f"Gradient checkpointing enabled: {self.config.enable_gradient_checkpointing}")
            logger.info(f"Timestep sampling strategy: {self.config.snr_type.value}")
    
    def _apply_lora(self):
        if self.is_main_process:
            logger.info("Applying LoRA to transformer using PeftAdapterMixin...")
        
        if self.config.pretrained_lora_path is not None:
            if self.is_main_process:
                logger.info(f"Loading pretrained LoRA from {self.config.pretrained_lora_path}")
            self.load_pretrained_lora(self.config.pretrained_lora_path)
        else:
            from peft import LoraConfig
            
            if self.config.lora_target_modules is None:
                target_modules = "all-linear"
            else:
                target_modules = self.config.lora_target_modules
            
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=target_modules,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type="FEATURE_EXTRACTION",
            )
            
            self.transformer.add_adapter(lora_config, adapter_name="default")

        
        if self.is_main_process:
            trainable_params = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.transformer.parameters())
            logger.info(f"LoRA applied successfully. Trainable parameters: {trainable_params:,} / {total_params:,} "
                       f"({100 * trainable_params / total_params:.2f}%)")
    
    def _apply_fsdp(self):
        if self.is_main_process:
            logger.info("Applying FSDP2 to transformer...")
        
        param_dtype = torch.bfloat16
        reduce_dtype = torch.float32  # Reduce in float32 for stability

        self.transformer = self.transformer.to(dtype=param_dtype)
        
        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
        )
        
        fsdp_config = {"mp_policy": mp_policy}
        if self.world_size > 1:
            try:
                fsdp_config["mesh"] = get_parallel_state().fsdp_mesh
            except Exception as e:
                if self.is_main_process:
                    logger.warning(f"Could not create DeviceMesh: {e}. FSDP will use process group instead.")
        
        for block in list(self.transformer.double_blocks) + list(self.transformer.single_blocks):
            if block is not None:
                fully_shard(block, **fsdp_config)
        
        fully_shard(self.transformer, **fsdp_config)
        
        if self.is_main_process:
            logger.info("FSDP2 applied successfully")
    
    def _apply_gradient_checkpointing(self):
        if self.is_main_process:
            logger.info("Applying gradient checkpointing to transformer blocks...")
        
        no_split_module_type = None
        for block in self.transformer.double_blocks:
            if block is not None:
                no_split_module_type = type(block)
                break
        
        if no_split_module_type is None:
            for block in self.transformer.single_blocks:
                if block is not None:
                    no_split_module_type = type(block)
                    break
        
        if no_split_module_type is None:
            logger.warning("Could not find block type for gradient checkpointing. Using fallback.")
            if hasattr(self.transformer, "gradient_checkpointing_enable"):
                self.transformer.gradient_checkpointing_enable()
            return
        
        def non_reentrant_wrapper(module):
            return checkpoint_wrapper(
                module,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            )
        
        def selective_checkpointing(submodule):
            return isinstance(submodule, no_split_module_type)
        
        apply_activation_checkpointing(
            self.transformer,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=selective_checkpointing,
        )
        
        if self.is_main_process:
            logger.info("Gradient checkpointing applied successfully")
    
    def _build_optimizer(self):
        if self.config.use_muon:
            self.optimizer = get_muon_optimizer(
                model=self.transformer,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        else:
            trainable_params = list(self.transformer.parameters())
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.config.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=self.config.weight_decay,
            )
        
        self.lr_scheduler = get_scheduler(
            "constant",
            optimizer=self.optimizer,
            num_warmup_steps=self.config.warmup_steps * self.world_size,
            num_training_steps=self.config.max_steps * self.world_size,
        )
        
        if self.is_main_process:
            logger.info(f"Optimizer and scheduler initialized")
    
    def encode_text(self, prompts, data_type: str = "image"):
        text_inputs = self.text_encoder.text2tokens(prompts, data_type=data_type)
        text_outputs = self.text_encoder.encode(text_inputs, data_type=data_type, device=self.device)
        text_emb = text_outputs.hidden_state
        text_mask = text_outputs.attention_mask
        
        text_emb_2 = None
        text_mask_2 = None
        if self.text_encoder_2 is not None:
            text_inputs_2 = self.text_encoder_2.text2tokens(prompts)
            text_outputs_2 = self.text_encoder_2.encode(text_inputs_2, device=self.device)
            text_emb_2 = text_outputs_2.hidden_state
            text_mask_2 = text_outputs_2.attention_mask
        
        return text_emb, text_mask, text_emb_2, text_mask_2
    
    def encode_byt5(self, text_ids: torch.Tensor, attention_mask: torch.Tensor):
        if self.byt5_kwargs["byt5_model"] is None:
            return None, None
        byt5_outputs = self.byt5_kwargs["byt5_model"](text_ids, attention_mask=attention_mask.float())
        byt5_emb = byt5_outputs[0]
        return byt5_emb, attention_mask
    
    def encode_images(self, images):
        """Encode images to vision states (for i2v)"""
        if self.vision_encoder is None:
            return None
        assert images.max() <= 1.0 and images.min() >= -1.0, f"Images must be in the range [-1, 1], but got {images.min()} {images.max()}"
        images = (images + 1) / 2 # [-1, 1] -> [0, 1]
        images_np = (images.cpu().permute(0, 2, 3, 1).numpy() * 255).clip(0, 255).astype("uint8")
        vision_states = self.vision_encoder.encode_images(images_np)
        return vision_states.last_hidden_state.to(device=self.device, dtype=self.transformer.dtype)
    
    def encode_vae(self, images: torch.Tensor) -> torch.Tensor:
        if images.max() > 1.0 or images.min() < -1.0:
            raise ValueError(f"Images must be in the range [-1, 1], but got {images.min()} {images.max()}")
        
        if images.ndim == 4:
            images = images.unsqueeze(2)
        
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16), self.vae.memory_efficient_context():
            latents = self.vae.encode(images).latent_dist.sample()
            if hasattr(self.vae.config, "shift_factor") and self.vae.config.shift_factor:
                latents = (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
            else:
                latents = latents * self.vae.config.scaling_factor
        
        return latents
    
    def get_condition(self, latents: torch.Tensor, task_type: str) -> torch.Tensor:
        b, c, f, h, w = latents.shape
        cond = torch.zeros([b, c + 1, f, h, w], device=latents.device, dtype=latents.dtype)
        
        if task_type == "t2v":
            return cond
        elif task_type == "i2v":
            cond[:, :-1, :1] = latents[:, :, :1]
            cond[:, -1, 0] = 1
            return cond
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    
    def sample_task(self, data_type: str) -> str:
        """
        Sample task type based on data type and configuration.
        
        For video data: samples between t2v and i2v based on i2v_prob
        For image data: always returns t2v (image-to-video generation)
        """
        if data_type == "image":
            return "t2v"
        elif data_type == "video":
            if random.random() < self.config.i2v_prob:
                return "i2v"
            else:
                return "t2v"
        else:
            return "t2v"
    
    def prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare batch for training.
        
        Expected batch format:
        {
            "pixel_values": torch.Tensor, # [B, C, F, H, W] for video or [B, C, H, W] for image
                                          # Pixel values must be in range [-1, 1] 
            "text": List[str],
            "data_type": str,  # "image" or "video"
            "byt5_text_ids": Optional[torch.Tensor],
            "byt5_text_mask": Optional[torch.Tensor],
        }
        
        Note: For video data, the temporal dimension F must be 4n+1 (e.g., 1, 5, 9, 13, 17, ...)
        to satisfy VAE requirements. The dataset should ensure this before returning data.
        
        """
        pixel_values = batch.get("pixel_values", None)
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.device)
        if 'latents' in batch:
            latents = batch['latents'].to(self.device)
        else:
            latents = self.encode_vae(pixel_values)
        
        if self.sp_enabled:
            latents = sync_tensor_for_sp(latents, self.sp_group)
            if pixel_values is not None:
                pixel_values = sync_tensor_for_sp(pixel_values, self.sp_group)
        
        data_type_raw = batch.get("data_type", "image")
        if isinstance(data_type_raw, list):
            data_type = data_type_raw[0]
        elif isinstance(data_type_raw, str):
            data_type = data_type_raw
        else:
            data_type = str(data_type_raw) if data_type_raw is not None else "image"
        task_type = self.sample_task(data_type)

        if self.sp_enabled:
            task_type = sync_tensor_for_sp(task_type, self.sp_group)
        
        cond_latents = self.get_condition(latents, task_type)
        prompts = batch["text"]
        if self.sp_enabled:
            prompts = sync_tensor_for_sp(prompts, self.sp_group)
        text_emb, text_mask, text_emb_2, text_mask_2 = self.encode_text(prompts, data_type=data_type)
        
        byt5_text_states = None
        byt5_text_mask = None
        if self.byt5_kwargs["byt5_model"] is not None:
            if "byt5_text_ids" in batch and batch["byt5_text_ids"] is not None:
                byt5_text_ids = batch["byt5_text_ids"].to(self.device)
                byt5_text_mask = batch["byt5_text_mask"].to(self.device)
                if self.sp_enabled:
                    byt5_text_ids = sync_tensor_for_sp(byt5_text_ids, self.sp_group)
                    byt5_text_mask = sync_tensor_for_sp(byt5_text_mask, self.sp_group)
                byt5_text_states, byt5_text_mask = self.encode_byt5(byt5_text_ids, byt5_text_mask)
            else:
                byt5_embeddings_list = []
                byt5_mask_list = []
                for prompt in prompts:
                    emb, mask = self.pipeline._process_single_byt5_prompt(prompt, self.device)
                    byt5_embeddings_list.append(emb)
                    byt5_mask_list.append(mask)
                
                byt5_text_states = torch.cat(byt5_embeddings_list, dim=0)
                byt5_text_mask = torch.cat(byt5_mask_list, dim=0)
        
        vision_states = None
        if task_type == "i2v":
            assert pixel_values is not None, '`pixel_values` must be provided for i2v task'
            if pixel_values.ndim == 5:
                first_frame = pixel_values[:, :, 0, :, :]
            else:
                first_frame = pixel_values
            vision_states = self.encode_images(first_frame)
        
        noise = torch.randn_like(latents)
        timesteps = self.timestep_sampler.sample(latents.shape[0], device=self.device)
        timesteps = timestep_transform(timesteps, self.config.num_train_timesteps, self.config.train_timestep_shift)
        
        latents_noised = self.noise_schedule.forward(latents, noise, timesteps)
        target = noise - latents
        
        if self.sp_enabled:
            target = sync_tensor_for_sp(target, self.sp_group)
        
        return {
            "latents_noised": latents_noised,
            "cond_latents": cond_latents,
            "timesteps": timesteps,
            "target": target,
            "text_emb": text_emb,
            "text_emb_2": text_emb_2,
            "text_mask": text_mask,
            "text_mask_2": text_mask_2,
            "byt5_text_states": byt5_text_states,
            "byt5_text_mask": byt5_text_mask,
            "vision_states": vision_states,
            "task_type": task_type,
            "data_type": data_type,
        }
    
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        inputs = self.prepare_batch(batch)
        latents_input = torch.cat([inputs["latents_noised"], inputs["cond_latents"]], dim=1)
        model_dtype = torch.bfloat16 if self.config.dtype == "bf16" else torch.float32
        
        extra_kwargs = {}
        if inputs["byt5_text_states"] is not None:
            extra_kwargs["byt5_text_states"] = inputs["byt5_text_states"].to(dtype=model_dtype)
            extra_kwargs["byt5_text_mask"] = inputs["byt5_text_mask"]
        
        with torch.autocast(device_type="cuda", dtype=model_dtype, enabled=(model_dtype == torch.bfloat16)):
            model_pred = self.transformer(
                latents_input.to(dtype=model_dtype),
                inputs["timesteps"],
                text_states=inputs["text_emb"].to(dtype=model_dtype),
                text_states_2=inputs["text_emb_2"].to(dtype=model_dtype) if inputs["text_emb_2"] is not None else None,
                encoder_attention_mask=inputs["text_mask"].to(dtype=model_dtype),
                vision_states=inputs["vision_states"].to(dtype=model_dtype) if inputs["vision_states"] is not None else None,
                mask_type=inputs["task_type"],
                extra_kwargs=extra_kwargs if extra_kwargs else None,
                return_dict=False,
            )[0]
        
        target = inputs["target"].to(dtype=model_pred.dtype)
        loss = nn.functional.mse_loss(model_pred, target)
        
        loss = loss / self.config.gradient_accumulation_steps
        loss.backward()
        
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            if self.config.max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.transformer.parameters(),
                    self.config.max_grad_norm
                )
            else:
                grad_norm = torch.tensor(0.0)
            
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
        else:
            grad_norm = torch.tensor(0.0)
        
        metrics = {
            "loss": loss.item() * self.config.gradient_accumulation_steps,
            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            "lr": self.lr_scheduler.get_last_lr()[0] if hasattr(self.lr_scheduler, "get_last_lr") else self.config.learning_rate,
        }
        
        return metrics
    
    def save_checkpoint(self, step: int):
        checkpoint_dir = os.path.join(self.config.output_dir, f"checkpoint-{step}")
        transformer_dir = os.path.join(checkpoint_dir, "transformer")
        
        if self.is_main_process:
            os.makedirs(checkpoint_dir, exist_ok=True)
        if self.world_size > 1:
            dist.barrier()
        
        if self.config.use_lora and hasattr(self.transformer, "save_lora_adapter"):
            lora_dir = os.path.join(checkpoint_dir, "lora")
            os.makedirs(lora_dir, exist_ok=True)
            
            if hasattr(self.transformer, "peft_config") and self.transformer.peft_config:
                adapter_names = list(self.transformer.peft_config.keys())
                if self.is_main_process:
                    logger.info(f"Saving {len(adapter_names)} LoRA adapter(s): {adapter_names}")
                
                for adapter_name in adapter_names:
                    adapter_dir = os.path.join(lora_dir, adapter_name)
                    os.makedirs(adapter_dir, exist_ok=True)
                    self.transformer.save_lora_adapter(
                        save_directory=adapter_dir,
                        adapter_name=adapter_name,
                        safe_serialization=True,
                    )
                    if self.is_main_process:
                        logger.info(f"LoRA adapter '{adapter_name}' saved to {adapter_dir}")
            else:
                raise RuntimeError("No LoRA adapter found in the model")
            
            if self.world_size > 1:
                dist.barrier()
        
        # Save full model state dict
        model_state_dict = get_model_state_dict(self.transformer)
        dcp.save(
            state_dict={"model": model_state_dict},
            checkpoint_id=transformer_dir,
        )

        optimizer_state_dict = get_optimizer_state_dict(
            self.transformer,
            self.optimizer,
        )
        optimizer_dir = os.path.join(checkpoint_dir, "optimizer")
        dcp.save(
            state_dict={"optimizer": optimizer_state_dict},
            checkpoint_id=optimizer_dir,
        )
        
        if self.is_main_process:
            training_state_path = os.path.join(checkpoint_dir, "training_state.pt")
            torch.save({
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "global_step": step,
            }, training_state_path)
        
        if self.world_size > 1:
            dist.barrier()
        
        if self.is_main_process:
            logger.info(f"Checkpoint saved at step {step} to {checkpoint_dir}")

    def load_pretrained_lora(self, lora_dir: str):
        self.transformer.load_lora_adapter(
            pretrained_model_name_or_path_or_dict=lora_dir,
            prefix=None,
            adapter_name="default",
            use_safetensors=True,
            hotswap=False,
        )
    
    def load_checkpoint(self, checkpoint_path: str):
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
        
        if self.is_main_process:
            logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        if self.world_size > 1:
            dist.barrier()
        
        
        transformer_dir = os.path.join(checkpoint_path, "transformer")
        if os.path.exists(transformer_dir):
            model_state_dict = get_model_state_dict(self.transformer)
            dcp.load(
                state_dict={"model": model_state_dict},
                checkpoint_id=transformer_dir,
            )
            if self.is_main_process:
                logger.info("Transformer model state loaded")
        else:
            logger.warning(f"Transformer dcp checkpoint not found from {checkpoint_path}")

        optimizer_dir = os.path.join(checkpoint_path, "optimizer")
        if os.path.exists(optimizer_dir):
            optimizer_state_dict = get_optimizer_state_dict(
                self.transformer,
                self.optimizer,
            )
            dcp.load(
                state_dict={"optimizer": optimizer_state_dict},
                checkpoint_id=optimizer_dir,
            )
            if self.is_main_process:
                logger.info("Optimizer state loaded")
        
        training_state_path = os.path.join(checkpoint_path, "training_state.pt")
        if os.path.exists(training_state_path):
            if self.is_main_process:
                training_state = torch.load(training_state_path, map_location=self.device)
                self.lr_scheduler.load_state_dict(training_state["lr_scheduler"])
                self.global_step = training_state.get("global_step", 0)
                logger.info(f"Training state loaded: global_step={self.global_step}")
            else:
                # Non-main processes will get global_step via broadcast
                self.global_step = 0
        
        if self.world_size > 1:
            global_step_tensor = torch.tensor(self.global_step, device=self.device)
            dist.broadcast(global_step_tensor, src=0)
            self.global_step = global_step_tensor.item()
        
        if self.world_size > 1:
            dist.barrier()
        
        if self.is_main_process:
            logger.info(f"Checkpoint loaded successfully. Resuming from step {self.global_step}")
    
    def train(self, dataloader):
        if self.is_main_process:
            logger.info("Starting training...")
            logger.info(f"Max steps: {self.config.max_steps}")
            logger.info(f"Batch size: {self.config.batch_size}")
            logger.info(f"Learning rate: {self.config.learning_rate}")
        
        if self.config.resume_from_checkpoint is not None:
            self.load_checkpoint(self.config.resume_from_checkpoint)
        
        self.transformer.train()
        
        while self.global_step < self.config.max_steps:
            for batch in dataloader:
                if self.global_step >= self.config.max_steps:
                    break

                metrics = self.train_step(batch)
                
                if self.global_step % self.config.log_interval == 0 and self.is_main_process:
                    logger.info(
                        f"Step {self.global_step}/{self.config.max_steps} | "
                        f"Loss: {metrics['loss']:.6f} | "
                        f"Grad Norm: {metrics['grad_norm']:.4f} | "
                        f"LR: {metrics['lr']:.2e}"
                    )
                
                if self.global_step >= 0 and self.global_step % self.config.validation_interval == 0:
                    self.validate(self.global_step)
                
                if (self.global_step + 1) % self.config.save_interval == 0:
                    self.save_checkpoint(self.global_step + 1)
                    if self.world_size > 1:
                        dist.barrier()
                
                self.global_step += 1
        
        if self.is_main_process:
            self.save_checkpoint(self.global_step)
            logger.info("Training completed!")
        
        if self.world_size > 1:
            dist.barrier()
            dist.destroy_process_group()
    
    def validate(self, step: int):
        """
        Implement your own validation logic here
        An example:


        logger.info(f"Running validation at step {step}...")
        
        self.transformer.eval()
        
        try:
            for idx, prompt in enumerate(self.config.validation_prompts):
                logger.info(f"Generating validation video {idx+1}/{len(self.config.validation_prompts)}: {prompt[:50]}...")
                
                with torch.no_grad():
                    output = self.pipeline(
                        prompt=prompt,
                        aspect_ratio="16:9",
                        video_length=self.config.validate_video_length,
                        enable_sr=False,  # Disable SR for faster validation
                        prompt_rewrite=False,  # Disable prompt rewrite for faster validation
                        output_type="pt",
                        seed=42,
                    )
                    
                    video_path = os.path.join(
                        self.validation_output_dir,
                        f"step_{step:06d}_prompt_{idx:02d}.mp4"
                    )
                    print(f"Prompt: {prompt}")
                    video_to_save = output.videos
                    if dist.get_rank() == 0:
                        save_video(video_to_save, video_path)
                        logger.info(f"Validation video saved to {video_path}")
        
        except Exception as e:
            logger.error(f"Error during validation: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        finally:
            self.transformer.train()
        pass
        """


def create_dummy_dataloader(config: TrainingConfig):
    """
    Create a dummy dataloader for testing.
    
    Note: This is a placeholder - users should implement their own dataset and dataloader
    that loads actual video/image data.
    
    Required fields for Dataset __getitem__:
    - "pixel_values": torch.Tensor
        * For video: shape [C, F, H, W] where F is the number of frames
        * For image: shape [C, H, W]
        * Pixel values must be in range [-1, 1]
        * Data type: torch.float32
        * Note: For video data, temporal dimension F must be 4n+1 (e.g., 1, 5, 9, 13, 17, 21, ...)
          to satisfy VAE requirements. The dataset should ensure this before returning data.
    
    - "text": str
        * Text prompt for this sample
    
    - "data_type": str
        * "video" for video data (supports both t2v and i2v tasks based on i2v_prob)
        * "image" for image data (always uses t2v task)
    
    Optional fields (for performance optimization):
    - "latents": torch.Tensor, shape [C_latent, F, H_latent, W_latent]
        * Pre-encoded VAE latents. If provided, pixel_values will be ignored and VAE encoding
          will be skipped, significantly speeding up training.
        * Should be in the same format as VAE encoder output (after scaling_factor applied)
        * Temporal dimension F must still be 4n+1 for video data
    
    Optional fields (for byT5 text encoding):
    - "byt5_text_ids": Optional[torch.Tensor], shape [seq_len]
        * Pre-tokenized byT5 token IDs. If provided, will be used directly.
        * If not provided, text will be tokenized on-the-fly.
    
    - "byt5_text_mask": Optional[torch.Tensor], shape [seq_len]
        * Attention mask for byT5 tokens (1 for valid tokens, 0 for padding)
        * Required if byt5_text_ids is provided
    
    Task type selection (automatic based on data_type and config.i2v_prob):
    - For "video" data: randomly samples between t2v (text-to-video) and i2v (image-to-video)
      based on config.i2v_prob probability
    - For "image" data: always uses t2v task
    
    Example sample format (what dataset __getitem__ should return):
    {
        "pixel_values": torch.Tensor([3, 121, 480, 848]),  # Video example
        "text": "A cat playing",
        "data_type": "video",
        "byt5_text_ids": torch.Tensor([256]),  # Optional
        "byt5_text_mask": torch.Tensor([256]),  # Optional
    }
    
    Or with pre-encoded latents (faster):
    {
        "latents": torch.Tensor([32, 31, 30, 53]),  # Pre-encoded VAE latents
        "text": "A cat playing",
        "data_type": "video",
    }
    """
    # This is a placeholder - users should implement their own dataloader
    class DummyDataset:
        def __init__(self, size=100):
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            # Video: temporal dimension must be 4n+1, using 17 frames
            # Generate data in range [-1, 1]

            resolution = (121, 480, 848)
            latent_resolution = [(resolution[0] - 1) // 4 + 1, resolution[1] // 16, resolution[2] // 16]

            data = torch.rand(3, *resolution) * 2.0 - 1.0  # [0, 1] -> [-1, 1]
            data_type = "video"

            return {
                "pixel_values": data,
                "text": "A sample prompt",
                "data_type": data_type,
                "latents": torch.randn(32, *latent_resolution),
                # "byt5_text_ids": torch.zeros((256), dtype=torch.int64),
                # "byt5_text_mask": torch.zeros((256), dtype=torch.int64),
            }
    
    dataset = DummyDataset()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    return dataloader


def main():
    parser = argparse.ArgumentParser(description="Train HunyuanVideo-1.5 on video data")
    
    # Model paths
    parser.add_argument("--pretrained_model_root", type=str, default='ckpts', help="Path to pretrained model")
    parser.add_argument("--pretrained_transformer_version", type=str, default="480p_t2v", help="Transformer version")
    
    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_steps", type=int, default=10000, help="Maximum training steps")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--train_timestep_shift", type=float, default=3.0, help="Train Timestep shift")
    parser.add_argument("--flow_snr_type", type=str, default="lognorm", 
                        choices=["uniform", "lognorm", "mix", "mode"],
                        help="SNR type for flow matching: uniform, lognorm, mix, or mode (default: lognorm)")
    
    # Data parameters
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--save_interval", type=int, default=1000, help="Checkpoint save interval")
    parser.add_argument("--log_interval", type=int, default=10, help="Logging interval")
    
    # Other parameters
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"], help="Data type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--i2v_prob", type=float, default=0.3, help="Probability of i2v task for video data (default: 0.3)")
    parser.add_argument("--use_muon", type=str_to_bool, nargs='?', const=True, default=True,
        help="Use Muon optimizer for training (default: true). "
             "Use --use_muon or --use_muon true/1 to enable, --use_muon false/0 to disable"
    )
    # FSDP and gradient checkpointing
    parser.add_argument(
        "--enable_fsdp", type=str_to_bool, nargs='?', const=True, default=True,
        help="Enable FSDP for distributed training (default: true). "
             "Use --enable_fsdp or --enable_fsdp true/1 to enable, --enable_fsdp false/0 to disable"
    )
    parser.add_argument(
        "--enable_gradient_checkpointing", type=str_to_bool, nargs='?', const=True, default=True,
        help="Enable gradient checkpointing (default: true). "
             "Use --enable_gradient_checkpointing or --enable_gradient_checkpointing true/1 to enable, "
             "--enable_gradient_checkpointing false/0 to disable"
    )
    parser.add_argument(
        "--sp_size", type=int, default=8,
        help="Sequence parallelism size (default: 1). Must evenly divide world_size. "
             "For example, if world_size=8, valid sp_size values are 1, 2, 4, 8."
    )
    parser.add_argument(
        "--dp_replicate", type=int, default=1,
        help="Data parallelism replicate size (default: 1). "
    )
    
    # Validation parameters
    parser.add_argument("--validation_interval", type=int, default=100, help="Run validation every N steps (default: 100)")
    parser.add_argument("--validation_prompts", type=str, nargs="+", default=None, 
                        help="Prompts for validation (default: single default prompt). Can specify multiple prompts.")
    parser.add_argument("--validation_timestep_shift", type=float, default=5.0, help="Validation Timestep shift")
    parser.add_argument("--validate_video_length", type=int, default=241, help="Video length (number of frames) for validation (default: 241)")
    
    # Resume training parameters
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint directory to resume training from (e.g., ./outputs/checkpoint-1000)")
    
    # LoRA parameters
    parser.add_argument("--use_lora", type=str_to_bool, nargs='?', const=True, default=False,
                        help="Enable LoRA training (default: false). "
                             "Use --use_lora or --use_lora true/1 to enable, --use_lora false/0 to disable")
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA rank (default: 8)")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha scaling parameter (default: 16)")
    parser.add_argument("--lora_dropout", type=float, default=0.0,
                        help="LoRA dropout rate (default: 0.0)")
    parser.add_argument("--lora_target_modules", type=str, nargs="+", default=None,
                        help="Target modules for LoRA (default: all Linear layers). "
                             "Example: --lora_target_modules img_attn_q img_attn_v img_mlp.fc1")
    parser.add_argument("--pretrained_lora_path", type=str, default=None,
                        help="Path to pretrained LoRA adapter to load. If provided, will load this adapter instead of creating a new one.")
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        pretrained_model_root=args.pretrained_model_root,
        pretrained_transformer_version=args.pretrained_transformer_version,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        dtype=args.dtype,
        seed=args.seed,
        i2v_prob=args.i2v_prob,
        enable_fsdp=args.enable_fsdp,
        enable_gradient_checkpointing=args.enable_gradient_checkpointing,
        sp_size=args.sp_size,
        use_muon=args.use_muon,
        dp_replicate=args.dp_replicate,
        validation_interval=args.validation_interval,
        validation_prompts=args.validation_prompts,
        train_timestep_shift=args.train_timestep_shift,
        validation_timestep_shift=args.validation_timestep_shift,
        snr_type=SNRType(args.flow_snr_type),
        validate_video_length=args.validate_video_length,
        resume_from_checkpoint=args.resume_from_checkpoint,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        pretrained_lora_path=args.pretrained_lora_path,
    )
    
    trainer = HunyuanVideoTrainer(config)
    dataloader = create_dummy_dataloader(config)
    trainer.train(dataloader)


if __name__ == "__main__":
    main()

