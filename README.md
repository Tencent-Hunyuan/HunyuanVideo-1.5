[中文文档](./README_CN.md)

# HunyuanVideo-1.5

<div align="center">

<img src="./assets/logo.png" alt="HunyuanVideo-1.5 Logo" width="80%">

# 🎬 HunyuanVideo-1.5: A leading lightweight video generation model

</div>


<div align="center">
<!-- <img src="./assets/banner.png" alt="HunyuanVideo-1.5 Banner" width="800"> -->

</div>


HunyuanVideo-1.5 is a video generation model that delivers top-tier quality with only 8.3B parameters, significantly lowering the barrier to usage. It runs smoothly on consumer-grade GPUs, making it accessible for every developer and creator. This repository provides the implementation and tools needed to generate creative videos.


<div align="center">
  <a href="https://hunyuan.tencent.com/video/zh?tabIndex=0" target="_blank"><img src=https://img.shields.io/badge/Official%20Site-333399.svg?logo=homepage height=22px></a>
  <a href=https://huggingface.co/tencent/HunyuanVideo-1.5 target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20Models-d96902.svg height=22px></a>
  <a href=https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5 target="_blank"><img src= https://img.shields.io/badge/Page-bb8a2e.svg?logo=github height=22px></a>
  <a href="https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/report/HunyuanVideo_1_5.pdf" target="_blank"><img src=https://img.shields.io/badge/Report-b5212f.svg?logo=arxiv height=22px></a>
  <a href=https://x.com/TencentHunyuan target="_blank"><img src=https://img.shields.io/badge/Hunyuan-black.svg?logo=x height=22px></a>
  <a href="https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/assets/HunyuanVideo_1_5_Prompt_Handbook_EN.md" target="_blank"><img src=https://img.shields.io/badge/📚-PromptHandBook-blue.svg?logo=book height=22px></a> <br/>
  <a href="./ComfyUI/README.md" target="_blank"><img src=https://img.shields.io/badge/ComfyUI-blue.svg?logo=book height=22px></a>
  <a href="https://github.com/ModelTC/LightX2V" target="_blank"><img src=https://img.shields.io/badge/LightX2V-yellow.svg?logo=book height=22px></a>
  <a href="https://tusi.cn/models/933574988890423836" target="_blank"><img src=https://img.shields.io/badge/吐司-purple.svg?logo=book height=22px></a>
  <a href="https://tensor.art/models/933574988890423836" target="_blank"><img src=https://img.shields.io/badge/TensorArt-cyan.svg?logo=book height=22px></a>

</div>


<p align="center">
    👏 Join our <a href="./assets/wechat.png" target="_blank">WeChat</a> and <a href="https://discord.gg/ehjWMqF5wY">Discord</a> | 
💻 <a href="https://hunyuan.tencent.com/video/zh?tabIndex=0">Official website Try our model!</a>&nbsp&nbsp
</p>

## 🔥🔥🔥 News
* 🚀 Nov 24, 2025: We now support cache inference, achieving approximately 2x speedup! Pull the latest code to try it.
* 👋 Nov 20, 2025: We release the inference code and model weights of HunyuanVideo-1.5.


## 🎥 Demo
<div align="center">
  <video src="https://github.com/user-attachments/assets/d45ec78e-ea40-47f1-8d4d-f4d9a0682e2d" width="60%"> </video>
</div>

## 🧩 Community Contributions

If you develop/use HunyuanVideo-1.5 in your projects, welcome to let us know.

- **ComfyUI** - [ComfyUI](https://github.com/comfyanonymous/ComfyUI): A powerful and modular diffusion model GUI with a graph/nodes interface. ComfyUI supports HunyuanVideo-1.5 with various engineering optimizations for fast inference. We provide a [ComfyUI Usage Guide](./ComfyUI/README.md) for HunyuanVideo-1.5.

- **Community-implemented ComfyUI Plugin** - [comfyui_hunyuanvideo_1.5_plugin](https://github.com/yuanyuan-spec/comfyui_hunyuanvideo_1.5_plugin): A community-implemented ComfyUI plugin for HunyuanVideo-1.5, offering both simplified and complete node sets for quick usage or deep workflow customization, with built-in automatic model download support.

- **LightX2V** - [LightX2V](https://github.com/ModelTC/LightX2V): A lightweight and efficient video generation framework that integrates HunyuanVideo-1.5, supporting multiple engineering acceleration techniques for fast inference.

## 📑 Open-source Plan
- HunyuanVideo-1.5 (T2V/I2V)
  - [x] Inference Code and checkpoints
  - [x] ComfyUI Support
  - [x] LightX2V Support
  - [ ] Diffusers Support
  - [ ] Release all model weights (Sparse attention, distill model, and SR models)

## 📋 Table of Contents
- [🔥🔥🔥 News](#-news)
- [🎥 Demo](#-demo)
- [🧩 Community Contributions](#-community-contributions)
- [📑 Open-source Plan](#-open-source-plan)
- [📖 Introduction](#-introduction)
- [✨ Key Features](#-key-features)
- [📜 System Requirements](#-system-requirements)
- [🛠️ Dependencies and Installation](#️-dependencies-and-installation)
- [🧱 Download Pretrained Models](#-download-pretrained-models)
- [📝 Prompt Guide](#-prompt-guide)
- [🔑 Usage](#-usage)
  - [Prompt Enhancement](#prompt-enhancement)
  - [Text to Video](#text-to-video)
  - [Image to Video](#image-to-video)
  - [Command Line Arguments](#command-line-arguments)
  - [Optimal Inference Configurations](#optimal-inference-configurations)
- [🧱 Models Cards](#-models-cards)
- [🎬 More Examples](#-more-examples)
- [📊 Evaluation](#-evaluation)
- [📚 Citation](#-citation)
- [🙏 Acknowledgements](#-acknowledgements)
- [🌟 Github Star History](#-github-star-history)


## 📖 Introduction
We present HunyuanVideo-1.5, a lightweight yet powerful video generation model that achieves state-of-the-art visual quality and motion coherence with only 8.3 billion parameters, enabling efficient inference on consumer-grade GPUs. This achievement is built upon several key components, including meticulous data curation, an advanced DiT architecture with selective and sliding tile attention(SSTA), enhanced bilingual understanding through glyph-aware text encoding , progressive pre-training and post-training, and an efficient video super-resolution network. Leveraging these designs, we developed a unified framework capable of high-quality text-to-video and image-to-video generation across multiple durations and resolutions. Extensive experiments demonstrate that this compact and proficient model establishes a new state-of-the-art among open-source models. By releasing the code and weights of HunyuanVideo-1.5, we provide the community with a high-performance foundation that significantly lowers the cost of video creation and research, making advanced video generation more accessible to all.


## ✨ Key Features
- **Lightweight High-Performance Architecture**: We propose an efficient architecture that integrates an 8.3B-parameter Diffusion Transformer (DiT) with a 3D causal VAE, achieving compression ratios of 16× in spatial dimensions and 4× along the temporal axis. Additionally, the innovative SSTA (Selective and Sliding Tile Attention) mechanism prunes redundant spatiotemporal kv blocks, significantly reduces computational overhead for long video sequences and accelerates inference, achieving an end-to-end speedup of $1.87 \times$ in 10-second 720p video synthesis compared to FlashAttention-3.

<div align="center">
<img src="./assets/hy_video_1_5_dit.png" alt="HunyuanVideo-1.5 DiT" width="600">
</div> 


- **Video Super-Resolution Enhancement**: We develop an efficient few-step super-resolution network that upscales outputs to 1080p. It enhances sharpness while correcting distortions, thereby refining details and overall visual texture.

<div align="center">
<img src="./assets/hy_video_1_5_vsr.png" alt="HunyuanVideo-1.5 VSR" width="600">
</div> 

- **End-to-End Training Optimization**: This work employs a multi-stage, progressive training strategy covering the entire pipeline from pre-training to post-training. Combined with the Muon optimizer to accelerate convergence, this approach holistically refines motion coherence, aesthetic quality, and human preference alignment, achieving professional-grade content generation.

## 📜 System Requirements

### Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA support
- **Minimum GPU Memory**: 14 GB (with model offloading enabled)
  
  > **Note:** The memory requirements above are measured with model offloading enabled. If your GPU has sufficient memory, you may disable offloading for improved inference speed.

### Software Requirements

- **Operating System**: Linux
- **Python**: Python 3.10 or higher
- **CUDA**: Compatible CUDA version for your PyTorch installation

## 🛠️ Dependencies and Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5.git
cd HunyuanVideo-1.5
```

### Step 2: Install Basic Dependencies

```bash
pip install -r requirements.txt
pip install -i https://mirrors.tencent.com/pypi/simple/ --upgrade tencentcloud-sdk-python
```

### Step 3: Install Attention Libraries

* Flash Attention: 
  Install Flash Attention for faster inference and reduced GPU memory consumption.
  Detailed installation instructions are available at [Flash Attention](https://github.com/Dao-AILab/flash-attention).

* Flex-Block-Attention: 
  flex-block-attn is only required for sparse attention to achieve faster inference and can be installed by the following command:
  ```bash
  git clone https://github.com/Tencent-Hunyuan/flex-block-attn.git
  cd flex-block-attn
  git submodule update --init --recursive
  python3 setup.py install
  ```

* SageAttention: 
  To enable SageAttention for faster inference, you need to install it by the following command:
  > **Note**: Enabling SageAttention will automatically disable Flex-Block-Attention.
  ```bash
  git clone https://github.com/cooper1637/SageAttention.git
  cd SageAttention 
  export EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=32 # Optional
  python3 setup.py install
  ```

## 🧱 Download Pretrained Models

> 💡 Distillation models and sparse attention models are still coming soon. Please stay tuned for the latest updates on the Hugging Face Model Card.

Download the pretrained models before generating videos. Detailed instructions are available at [checkpoints-download.md](checkpoints-download.md).

## 📝 Prompt Guide
### Prompt Writing Handbook
Prompt enhancement plays a crucial role in enabling our model to generate high-quality videos. By writing longer and more detailed prompts, the generated video will be significantly improved. We encourage you to craft comprehensive and descriptive prompts to achieve the best possible video quality. we recommend community partners consulting our official guide on how to write effective prompts. 

**Reference:** **[HunyuanVideo-1.5 Prompt Handbook](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/assets/HunyuanVideo_1_5_Prompt_Handbook_EN.md)**

### System Prompts for Automatic Prompt Enhancement
For users seeking to optimize prompts for other large models, it is recommended to consult the definition of `t2v_rewrite_system_prompt` in the file `hyvideo/utils/rewrite/t2v_prompt.py` to guide text-to-video rewriting. Similarly, for image-to-video rewriting, refer to the definition of `i2v_rewrite_system_prompt` in `hyvideo/utils/rewrite/i2v_prompt.py`.

## 🔑 Usage
### Video Generation

For prompt rewriting, we recommend using Gemini or models deployed via vLLM. This codebase currently only supports models compatible with the vLLM API. If you wish to use Gemini, you will need to implement your own interface calls.

For models with a vLLM API, note that T2V (text-to-video) and I2V (image-to-video) have different recommended models and environment variables:

- T2V: use [Qwen3-235B-A22B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507), configure `T2V_REWRITE_BASE_URL` and `T2V_REWRITE_MODEL_NAME`
- I2V: use [Qwen3-VL-235B-A22B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct), configure `I2V_REWRITE_BASE_URL` and `I2V_REWRITE_MODEL_NAME`

> You may set the above model names to any other vLLM-compatible models you have deployed (including HuggingFace models).  
> Rewriting is enabled by default (`--rewrite` defaults to `true`); to disable it explicitly, use `--rewrite false` or `--rewrite 0`. If no vLLM endpoint is configured, the pipeline runs without remote rewriting.

Example: Generate a video (works for both T2V and I2V; set `IMAGE_PATH=none` for T2V or provide an image path for I2V)

```bash
export T2V_REWRITE_BASE_URL="<your_vllm_server_base_url>"
export T2V_REWRITE_MODEL_NAME="<your_model_name>"
export I2V_REWRITE_BASE_URL="<your_vllm_server_base_url>"
export I2V_REWRITE_MODEL_NAME="<your_model_name>"

PROMPT='A girl holding a paper with words "Hello, world!"'

IMAGE_PATH=./data/reference_image.png # Optional, 'none' or <image path>
SEED=1
ASPECT_RATIO=16:9
RESOLUTION=480p
OUTPUT_PATH=./outputs/output.mp4

# Configuration
N_INFERENCE_GPU=8 # Parallel inference GPU count
CFG_DISTILLED=true # Inference with CFG distilled model, 2x speedup
SPARSE_ATTN=false # Inference with sparse attention (only 720p models are equipped with sparse attention). Please ensure flex-block-attn is installed
SAGE_ATTN=true # Inference with SageAttention
REWRITE=true # Enable prompt rewriting. Please ensure rewrite vLLM server is deployed and configured.
OVERLAP_GROUP_OFFLOADING=true # Only valid when group offloading is enabled, significantly increases CPU memory usage but speeds up inference
ENABLE_CACHE=true # Enable feature cache during inference. Significantly speeds up inference.
MODEL_PATH=ckpts # Path to pretrained model

torchrun --nproc_per_node=$N_INFERENCE_GPU generate.py \
  --prompt "$PROMPT" \
  --image_path $IMAGE_PATH \
  --resolution $RESOLUTION \
  --aspect_ratio $ASPECT_RATIO \
  --seed $SEED \
  --cfg_distilled $CFG_DISTILLED \
  --sparse_attn $SPARSE_ATTN \
  --use_sageattn $SAGE_ATTN \
  --enable_cache $ENABLE_CACHE \
  --rewrite $REWRITE \
  --output_path $OUTPUT_PATH \
  --overlap_group_offloading $OVERLAP_GROUP_OFFLOADING \
  --save_pre_sr_video \
  --model_path $MODEL_PATH
```

> **Tips:** If your GPU memory is > 14GB but you encounter OOM (Out of Memory) errors during generation, you can try setting the following environment variable before running:
> ```bash
> export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
> ```
> 
> **Tips:** If you have limited CPU memory and encounter OOM during inference, you can try disable overlapped group offloading by adding the following argument:
> ```bash
> --overlap_group_offloading false
> ```


### Command Line Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--prompt` | str | Yes | - | Text prompt for video generation |
| `--negative_prompt` | str | No | `''` | Negative prompt for video generation |
| `--resolution` | str | Yes | - | Video resolution: `480p` or `720p` |
| `--model_path` | str | Yes | - | Path to pretrained model directory |
| `--aspect_ratio` | str | No | `16:9` | Aspect ratio of the output video |
| `--num_inference_steps` | int | No | `50` | Number of inference steps |
| `--video_length` | int | No | `121` | Number of frames to generate |
| `--seed` | int | No | `123` | Random seed for reproducibility |
| `--image_path` | str | No | `None` | Path to reference image (enables i2v mode). Use `none` or `None` to explicitly use text-to-video mode |
| `--output_path` | str | No | `None` | Output file path (if not provided, saves to `./outputs/output_{transformer_version}_{timestamp}.mp4`) |
| `--sr` | bool | No | `true` | Enable super resolution (use `--sr false` or `--sr 0` to disable) |
| `--save_pre_sr_video` | bool | No | `false` | Save original video before super resolution (use `--save_pre_sr_video` or `--save_pre_sr_video true` to enable, only effective when super resolution is enabled) |
| `--rewrite` | bool | No | `true` | Enable prompt rewriting (use `--rewrite false` or `--rewrite 0` to disable, may result in lower quality video generation) |
| `--cfg_distilled` | bool | No | `false` | Enable CFG distilled model for faster inference (~2x speedup, use `--cfg_distilled` or `--cfg_distilled true` to enable) |
| `--sparse_attn` | bool | No | `false` | Enable sparse attention for faster inference (~1.5-2x speedup, requires H-series GPUs, auto-enables CFG distilled, use `--sparse_attn` or `--sparse_attn true` to enable) |
| `--offloading` | bool | No | `true` | Enable CPU offloading (use `--offloading false` or `--offloading 0` to disable for faster inference if GPU memory allows) |
| `--group_offloading` | bool | No | `None` | Enable group offloading (default: None, automatically enabled if offloading is enabled. Use `--group_offloading` or `--group_offloading true/1` to enable, `--group_offloading false/0` to disable) |
| `--overlap_group_offloading` | bool | No | `true` | Enable overlap group offloading (default: true). Significantly increases CPU memory usage but speeds up inference. Use `--overlap_group_offloading` or `--overlap_group_offloading true/1` to enable, `--overlap_group_offloading false/0` to disable |
| `--dtype` | str | No | `bf16` | Data type for transformer: `bf16` (faster, lower memory) or `fp32` (better quality, slower, higher memory) |
| `--use_sageattn` | bool | No | `false` | Enable SageAttention (use `--use_sageattn` or `--use_sageattn true/1` to enable, `--use_sageattn false/0` to disable) |
| `--sage_blocks_range` | str | No | `0-53` | SageAttention blocks range (e.g., `0-5` or `0,1,2,3,4,5`) |
| `--enable_torch_compile` | bool | No | `false` | Enable torch compile for transformer (use `--enable_torch_compile` or `--enable_torch_compile true/1` to enable, `--enable_torch_compile false/0` to disable) |
| `--enable_cache` | bool | No | `false` | Enable cache for transformer (use `--enable_cache` or `--enable_cache true/1` to enable, `--enable_cache false/0` to disable) |
| `--cache_start_step` | int | No | `11` | Start step to skip when using cache |
| `--cache_end_step` | int | No | `45` | End step to skip when using cache |
| `--total_steps` | int | No | `50` | Total inference steps |
| `--cache_step_interval` | int | No | `4` | Step interval to skip when using cache |

**Note:** Use `--nproc_per_node` to specify the number of GPUs. For example, `--nproc_per_node=8` uses 8 GPUs.

### Optimal Inference Configurations

The following table provides the optimal inference configurations (CFG scale, embedded CFG scale, flow shift, and inference steps) for each model to achieve the best generation quality:

| Model | CFG Scale | Embedded CFG Scale | Flow Shift | Inference Steps |
|-------|-----------|-------------------|------------|-----------------|
| 480p T2V | 6 | None | 5 | 50 |
| 480p I2V | 6 | None | 5 | 50 |
| 720p T2V | 6 | None | 9 | 50 |
| 720p I2V | 6 | None | 7 | 50 |
| 480p T2V CFG Distilled | 1 | None | 5 | 50 |
| 480p I2V CFG Distilled | 1 | None | 5 | 50 |
| 720p T2V CFG Distilled | 1 | None | 9 | 50 |
| 720p I2V CFG Distilled | 1 | None | 7 | 50 |
| 720p T2V CFG Distilled Sparse | 1 | None | 9 | 50 |
| 720p I2V CFG Distilled Sparse | 1 | None | 7 | 50 |
| 480→720 SR Step Distilled | 1 | None | 2 | 6 |
| 720→1080 SR Step Distilled | 1 | None | 2 | 8 |

**Please note that the cfg distilled model we provided, must use 50 steps to generate correct results.**


## 🧱 Models Cards
|ModelName| Download                     |
|-|---------------------------| 
|HunyuanVideo-1.5-480P-T2V|[480P-T2V](https://huggingface.co/tencent/HunyuanVideo-1.5/tree/main/transformer/480p_t2v) |
|HunyuanVideo-1.5-480P-I2V |[480P-I2V](https://huggingface.co/tencent/HunyuanVideo-1.5/tree/main/transformer/480p_i2v) |
|HunyuanVideo-1.5-480P-T2V-cfg-distill | [480P-T2V-cfg-distill](https://huggingface.co/tencent/HunyuanVideo-1.5/tree/main/transformer/480p_t2v_distilled) |
|HunyuanVideo-1.5-480P-I2V-cfg-distill |[480P-I2V-cfg-distill](https://huggingface.co/tencent/HunyuanVideo-1.5/tree/main/transformer/480p_i2v_distilled) |
|HunyuanVideo-1.5-720P-T2V|[720P-T2V](https://huggingface.co/tencent/HunyuanVideo-1.5/tree/main/transformer/720p_t2v) |
|HunyuanVideo-1.5-720P-I2V |[720P-I2V](https://huggingface.co/tencent/HunyuanVideo-1.5/tree/main/transformer/720p_i2v) |
|HunyuanVideo-1.5-720P-T2V-cfg-distill| Comming soon |
|HunyuanVideo-1.5-720P-I2V-cfg-distill |[720P-I2V-cfg-distill](https://huggingface.co/tencent/HunyuanVideo-1.5/tree/main/transformer/720p_i2v_distilled) |
|HunyuanVideo-1.5-720P-T2V-sparse-cfg-distill| Comming soon |
|HunyuanVideo-1.5-720P-I2V-sparse-cfg-distill |[720P-I2V-sparse-cfg-distill](https://huggingface.co/tencent/HunyuanVideo-1.5/tree/main/transformer/720p_i2v_distilled_sparse) |
|HunyuanVideo-1.5-720P-sr-step-distill |[720P-sr](https://huggingface.co/tencent/HunyuanVideo-1.5/tree/main/transformer/720p_sr_distilled) |
|HunyuanVideo-1.5-1080P-sr-step-distill |[1080P-sr](https://huggingface.co/tencent/HunyuanVideo-1.5/tree/main/transformer/1080p_sr_distilled) |



## 🎬 More Examples
|Features|Demo1|Demo2|
|------|------|------|
|Strong Instruction Following|<video src="https://github.com/user-attachments/assets/fdc3c27b-69f5-46a1-b707-0b57510fa32f" width="600"> </video> <details><summary>📋 Show input prompt</summary> ```一名哀伤的黑发中国女子凝望天空，复古胶片风格烘托出怀旧戏剧氛围``` </details> <details><summary>📋 Show rewrite prompt</summary> ```俯视角度，一位有着深色，略带凌乱的长卷发的年轻中国女性，佩戴着闪耀的珍珠项链和圆形金色耳环，她凌乱的头发被风吹散，她微微抬头，望向天空，神情十分哀伤，眼中含着泪水。嘴唇涂着红色口红。背景是带有华丽红色花纹的图案。画面呈现复古电影风格，色调低饱和，带着轻微柔焦，烘托情绪氛围，质感仿佛20世纪90年代的经典胶片风格，营造出怀旧且富有戏剧性的感觉。``` </details>|<video src="https://github.com/user-attachments/assets/3fcb42cc-cdd3-4651-86a6-645a858561c4" width="600"> </video> <details><summary>📋 Show input prompt</summary> ```建筑蓝图上的线条化为实体，瞬间生长出一个完整的复古工业风办公空间。``` </details> <details><summary>📋 Show rewrite prompt</summary> ```一座空旷的现代阁楼里，有一张铺展在地板中央的建筑蓝图。忽然间，图纸上的线条泛起微光，仿佛被某种无形的力量唤醒。紧接着，那些发光的线条开始向上延伸，从平面中挣脱，勾勒出立体的轮廓——就像在空中进行一场无声的3D打印。随后，奇迹在加速发生：极简的橡木办公桌、优雅的伊姆斯风格皮质椅、高挑的工业风金属书架，还有几盏爱迪生灯泡，以光纹为骨架迅速“生长”出来。转瞬间，线条被真实的材质填充——木材的温润、皮革的质感、金属的冷静，都在眨眼间完整呈现。最终，所有家具稳固落地，蓝图的光芒悄然褪去。一个完整的办公空间，就这样从二维的图纸中诞生。``` </details>|
|Smooth Motion Generation|<video src="https://github.com/user-attachments/assets/447847f0-490a-45f9-a86d-a67ab1ff4231" width="600"> </video> <details><summary>📋 Show input prompt</summary> ```A DJ is immersed in his musical world. He wears a pair of professional, matte-black headphones, revealing a focused expression. He wears a black bomber jacket, zipped open to reveal a T-shirt underneath. His upper body sways back and forth rhythmically to the throbbing electronic beats, his head moving with precise movement. The mixing console in front of him serves as the primary source of light. In the distance, the cool white glow of several stadium floodlights casts a deep, dark haze across the vast field, casting long shadows across the emerald green grass, creating a stark contrast to the brightly lit area surrounding the DJ booth. His hands danced swiftly and precisely across the equipment. The entire scene was filled with high-tech dynamics and the solitary creative passion. Against the backdrop of the vast and silent night stadium, it created an atmosphere of high focus, energy, and a slightly surreal feeling.``` </details> <details><summary>📋 Show rewrite prompt</summary> ```slowly advancing medium shot, shot from a level angle, focuses on the center of an empty football field, where a DJ is immersed in his musical world. He wears a pair of professional, matte-black headphones, one earcup slightly removed, revealing a focused expression and a brow beaded with sweat from his intense concentration. He wears a black bomber jacket, zipped open to reveal a T-shirt underneath. His upper body sways back and forth rhythmically to the throbbing electronic beats, his head moving with precise movement. The mixing console in front of him serves as the primary source of light. In the distance, the cool white glow of several stadium floodlights casts a deep, dark haze across the vast field, casting long shadows across the emerald green grass, creating a stark contrast to the brightly lit area surrounding the DJ booth. His hands danced swiftly and precisely across the equipment, one hand steadily pushing and pulling a long volume fader, while the fingers of the other nimbly jumped between the illuminated knobs and pads, sometimes decisively cutting a bass line, sometimes triggering an echo effect. The entire scene was filled with high-tech dynamics and the solitary creative passion. Against the backdrop of the vast and silent night stadium, it created an atmosphere of high focus, energy, and a slightly surreal feeling.``` </details>|<video src="https://github.com/user-attachments/assets/49057fe8-a102-4fd7-bd92-e9561abb9f45" width="600"> </video> <details><summary>📋 Show input prompt</summary> ```A figure skater performs a rapid, graceful Biellmann spin, captured from all angles.``` </details> <details><summary>📋 Show rewrite prompt</summary> ```The video captures a figure skater performing a Biellmann spin on ice. The subject is a female skater in a glittering costume. Initially, she spins on one leg. Then, she reaches back and pulls her free leg up. Next, she spins rapidly, becoming a blur of motion, with ice shavings spraying from her skate blade. The background is an ice rink with blurred advertising boards. The camera circles around the subject to capture the spin from all angles. The lighting is spotlit, creating lens flares and sparkles on her costume. The overall video presents a graceful artistic sports style.``` </details>|
|Cinematic Aesthetics|<video src="https://github.com/user-attachments/assets/4098cf72-357d-4b81-97df-6752064ce0c3" width="600"> </video> <details><summary>📋 Show input prompt</summary> ```固定镜头,焦点在图片里的挂钟上，镜头轻微摇晃营造手持摄影感，​wjw,filmphotos,Film Grain,Reversal film photography，Wong Kar-wai movies,cinematic photography, HK film style,neon lighting, in the style of Wong Kar Wai film``` </details> <details><summary>📋 Show rewrite prompt</summary> ```Handheld lens shooting, the camera focuses on the wall clock hanging on the green-toned wall, shaking slightly. The second hand sweeps steadily across the clock face, and the shadow of the clock cast on the wall shifts subtly with the movement of the lens.``` </details>|<video src="https://github.com/user-attachments/assets/2b4575e5-79f1-4011-bed0-e8380198f7c9" width="600"> </video> <details><summary>📋 Show input prompt</summary> ```The leaves of calamus shine in the sunlight, dotted with dewdrops that trickle down to the ground with the breeze.``` </details> <details><summary>📋 Show rewrite prompt</summary> ```A macro shot focuses on long, slender calamus leaves, rendered in a cinematic photography realistic style. The main leaf, a vibrant, deep green, is positioned diagonally across the frame. Its surface is covered in tiny, glistening spherical dewdrops that catch and refract the bright morning sunlight, creating sparkling highlights. Initially, a larger, perfectly round dewdrop clings to the upper section of the leaf, its surface tension holding it in place. Then, as the leaf sways almost imperceptibly, the dewdrop begins to slowly dislodge. Next, it starts to trickle down the central vein of the leaf, its shape elongating slightly as it moves, leaving a subtle, glistening wet trail in its path. Finally, it reaches the pointed tip of the leaf, hangs for a brief moment, and falls out of the bottom of the frame. In the background, other leaves and blades of grass are softly blurred, creating a beautiful bokeh effect with soft, out-of-focus circles of light. The environment is bathed in the warm, golden glow of early morning sunlight, which streams in from behind the leaves, backlighting them and causing their wet edges to shine brilliantly. The overall impression is one of serene, natural beauty, captured in a highly realistic and detailed manner. This is a macro shot. The camera tilts down very slowly, following the path of the main dewdrop as it travels down the leaf. The lighting is soft and natural, with strong backlighting to create a radiant, glowing effect on the dewdrops and leaf edges, characteristic of professional nature photography. The atmosphere is peaceful and serene. The overall video presents a cinematic photography realistic style.``` </details>|
|Text Rendering|<video src="https://github.com/user-attachments/assets/7c964fc5-c27e-4bd0-bf3f-eb8fca2caef6" width="600"> </video> <details><summary>📋 Show input prompt</summary> ```赛博朋克风格的夜晚街角，一个巨大的招牌上， “Hunyuan Video 1.5”的霓虹灯管轮廓已经安装好。镜头推进，霓虹灯从“H”开始，伴随着‘滋滋’的电流声，每个字母依次亮起粉紫色的光芒，直到全部点亮，照亮了潮湿的街道。赛博朋克，城市美学``` </details> <details><summary>📋 Show rewrite prompt</summary> ```On a wet street corner in a cyberpunk city at night, a large neon sign reading "Hunyuan Video 1.5" lights up sequentially, illuminating the dark, rainy environment with a pinkish-purple glow. he scene is a dark, rain-slicked street corner in a futuristic, cinematic cyberpunk city. Mounted on the metallic, weathered facade of a building is a massive, unlit neon sign. The sign's glass tube framework clearly spells out the words "Hunyuan Video 1.5". Initially, the street is dimly lit, with ambient light from distant skyscrapers creating shimmering reflections on the wet asphalt below. Then, the camera zooms in slowly toward the sign. As it moves, a low electrical sizzling sound begins. In the background, the dense urban landscape of the cyberpunk metropolis is visible through a light atmospheric haze, with towering structures adorned with their own flickering advertisements. A complex web of cables and pipes crisscrosses between the buildings. The shot is at a low angle, looking up at the sign to emphasize its grand scale. The lighting is high-contrast and dramatic, dominated by the neon glow which creates sharp, specular reflections and deep shadows. The atmosphere is moody and tech-noir. The overall video presents a cinematic photography realistic style.,``` </details>|<video src="https://github.com/user-attachments/assets/73e8b741-baec-4a40-9d36-a1435172ab64" width="600"> </video> <details><summary>📋 Show input prompt</summary> ```一张铺开的中国宣纸上，浓墨滴入水中，晕染出壮丽的山水画轮廓。山峰、云雾、孤舟在墨色中自然形成。随后，这些水墨元素巧妙地流动、重组，在画面的留白处汇聚成"Hunyuan Video 1.5"的书法字体。优雅，诗意，文化底蕴``` </details> <details><summary>📋 Show rewrite prompt</summary> ```A drop of black ink blooms on wet Chinese Xuan paper, forming a landscape painting before the ink elements fluidly reassemble into the calligraphic text "Hunyuan Video 1.5". On a flat, laid-out sheet of off-white Chinese Xuan paper with a subtle, fibrous texture, the scene unfolds. Initially, a single, concentrated drop of deep black ink falls into a clear, wet area at the center of the paper. Then, the ink instantly begins to bloom outwards in intricate, flowing tendrils of varying shades from jet-black to smoky grey. As it spreads, the ink wash naturally and rapidly forms the silhouette of a majestic mountain range with sharp, defined peaks. Next, softer, diluted grey tones billow around the mountains, creating layers of atmospheric mist and clouds, while a simple, dark stroke materializes as a lone boat on a tranquil, watery expanse at the base. As the landscape is formed, the ink elements—the lines of the mountains, wisps of cloud, and the shape of the boat—begin to deconstruct, dissolving into flowing streams of liquid ink. Finally, these streams move gracefully across the paper's empty white space, converging and elegantly reorganizing to form the text "Hunyuan Video 1.5" in a fluid, semi-cursive calligraphic style. The background is the minimalist expanse of the Xuan paper itself, its texture providing a subtle depth. The entire process is lit by soft, even, diffused light from above, which enhances the rich tonal variations of the ink and the delicate texture of the paper without creating harsh shadows. Bird's-eye view. The camera is positioned directly above the subject, capturing the entire process. The camera remains static. The aesthetic is a high-quality, dynamic Chinese ink wash animation style, perfectly simulating the real-world physics of ink spreading on wet paper. The entire sheet of paper and the final text are kept fully within the frame. Poetic, elegant, artistic. The overall video presents a dynamic Chinese ink wash animation style.``` </details>|
|Physics Compliance|<video src="https://github.com/user-attachments/assets/f1d74e48-cc03-415d-b75f-f7186a4fb41d" width="600"> </video> <details><summary>📋 Show input prompt</summary> ```In a sleek museum gallery, a woman pauses before a gilded oil painting. The painted man inside slowly comes alive, lifting a bottle and pouring real wine straight from the canvas into her glass. Surrounded by stylish art critics moving naturally through the hall, she accepts the pour with calm elegance, as if the impossible were routine. ``` </details> <details><summary>📋 Show rewrite prompt</summary> ```In a sleek museum gallery, a woman receives a glass of wine poured directly from an animated oil painting. A sophisticated woman with dark hair tied back elegantly stands in the mid-ground. She is wearing a simple, black silk sleeveless dress and holds a clear, crystal wine glass in her right hand. She is positioned before a large, baroque-style oil painting in an ornate, gilded frame. Inside the painting, an aristocratic man with a mustache, dressed in a dark velvet doublet with a white lace collar, is depicted. His form is defined by visible, impasto oil brushstrokes. Initially, the woman watches the painting with calm poise. Then, the painted man's arm slowly animates, his painted texture retained as he lifts a dark bottle. Next, a photorealistic stream of red wine emerges directly from the flat canvas surface, arcing through the air and splashing gently into the real crystal glass she holds. She remains perfectly still, accepting the impossible pour with a subtle, knowing smile. The setting is a modern art gallery with high white walls and polished dark concrete floors that reflect the ambient light. Focused track lighting from the high ceiling casts a warm, dramatic spotlight on the woman and the painting, creating soft shadows. In the background, two other gallery patrons, a man and a woman in stylish, modern attire, stroll slowly from right to left, their figures slightly blurred by a shallow depth of field, moving naturally through the hall. The shot is at an eye-level angle with the woman. The camera remains static, capturing the surreal event in a steady medium shot. The lighting is high-contrast and dramatic, reminiscent of a cinematic photography realistic style, using soft side lighting to accentuate the woman's features and the texture of the painting. The mood is surreal, elegant, and mysterious. The overall video presents a cinematic photography realistic style.``` </details>|<video src="https://github.com/user-attachments/assets/07bcce06-ff4f-4688-8c60-c02f600635ea" width="600"> </video> <details><summary>📋 Show input prompt</summary> ```An intact soda can is slowly crushed by a hand.``` </details> <details><summary>📋 Show rewrite prompt</summary> ```In a medium close-up, a hand slowly crushes an intact red and white soda can on a wooden table. A male hand with visible, realistic skin texture is wrapped firmly around the middle of an intact, pristine red and white aluminum soda can. The can, covered in glistening condensation droplets, rests on a dark, polished wooden surface. The cinematic realism captures every minute detail of the scene. Initially, the hand's grip is steady, with the can's cylindrical shape perfectly preserved. Then, the fingers begin to tighten slowly, the knuckles whitening slightly from the exertion. Next, the smooth aluminum surface starts to buckle under the controlled pressure, a sharp crease forming vertically down its side as the metallic sheen distorts. As the hand continues its deliberate squeeze, the can collapses inward progressively, the vibrant red paint wrinkling as the metal structure crumples. Finally, the can is left significantly crushed, its form now an irregular, crumpled shape held tightly in the fist. The scene takes place on a dark, polished wooden tabletop that catches soft, diffuse reflections. The grain of the wood is faintly discernible, adding a layer of texture to the foreground. The background is completely out of focus, rendered as a soft, dark, and non-descript blur, which isolates the main action and enhances the photorealistic quality of the shot. The shot is a medium close-up, presented in a cinematic photography realistic style. The camera remains static at a slightly high angle, looking down to provide a clear and unobstructed view of the can's deformation. Soft side lighting creates high contrast, sculpting the muscles and tendons of the hand while casting specular highlights on the metallic can and the water droplets. The atmosphere is focused and intense. The overall video presents a cinematic photography realistic style.``` </details>|
|Camera Movement|<video src="https://github.com/user-attachments/assets/6deacbfe-4cca-48d7-a2be-cb638a3e01cb" width="600"> </video> <details><summary>📋 Show input prompt</summary> ```圣诞节的家中，小女孩靠着妈妈听妈妈读书，背景是下着雪的窗外，镜头缓慢下移，一只可爱的长毛小白猫戴着圣诞帽趴在温暖的地摊上``` </details> <details><summary>📋 Show rewrite prompt</summary> ```In a cozy home on Christmas, a young girl leans against her mother as they read a book, and the camera moves down to reveal a fluffy white cat in a Santa hat resting on a warm rug. In a warmly lit living room on a snowy Christmas evening, a young mother and her little daughter are sitting together on a comfortable sofa. The mother, with a gentle expression and wearing a cream-colored knitted sweater, holds an open storybook with colorful illustrations. Her daughter, a small girl with brown hair in pigtails and a red pajama set, leans her head affectionately on her mother's shoulder, her eyes fixed on the book. On the floor below them, a fluffy, long-haired white cat is curled up on a plush, beige wool rug. The cat wears a tiny red and white Santa hat perched between its ears. Initially, the shot focuses on the mother and daughter, capturing their quiet, shared moment. The mother’s finger gently rests on the page of the book. Then, the camera slowly moves downward, gliding past the book and their laps. Finally, the camera settles at a low angle, bringing the adorable white cat into sharp focus as the primary subject. The cat's chest gently rises and falls with each breath, its eyes peacefully closed. Through a large window in the background, large, soft snowflakes can be seen falling silently against the dark blue twilight sky, creating a peaceful and serene backdrop. Faint, out-of-focus golden Christmas lights twinkle in the corner of the room, adding to the warm, festive atmosphere. The scene is imbued with a sense of comfort and holiday warmth, creating a beautiful cinematic photography realistic image. The camera slowly moves downward. The shot uses soft, warm interior lighting that casts gentle shadows, creating a high-contrast, cinematic look. A shallow depth of field keeps the focus on the subjects while beautifully blurring the background elements. The mood is heartwarming, peaceful, and festive. The overall video presents a cinematic photography realistic style.``` </details>|<video src="https://github.com/user-attachments/assets/8e72ed0f-f8ac-445b-97e5-eb4b16fbc121" width="600"> </video> <details><summary>📋 Show input prompt</summary> ```The hiker begins walking forward along the trail, causing the water bottle to swing rhythmically with each step. The camera gradually pulls back and rises to reveal a vast desert landscape stretching out ahead.``` </details> <details><summary>📋 Show rewrite prompt</summary> ```The hiker begins walking forward along the trail, causing the water bottle to swing rhythmically with each step. The camera gradually pulls back and rises to reveal a vast desert landscape stretching out ahead, while the sun position shifts from afternoon to dusk, casting increasingly longer shadows across the terrain as the figure becomes smaller in the frame.``` </details>|
|Multi-Style Support|<video src="https://github.com/user-attachments/assets/65b2c5a5-e6ba-43be-9462-a98b03b675f1" width="600"> </video> <details><summary>📋 Show input prompt</summary> ```Have the cake man begin to take chunks out of himself and eat it.``` </details> <details><summary>📋 Show rewrite prompt</summary> ```The cake man sits on the chair, with his hands resting on his knees. Then, he slowly raises his right hand and breaks off a piece of cake from his left shoulder. Next, he brings the piece of cake to his mouth and begins to chew. At the same time, his eyes widen slightly, and his mouth parts gently. After that, he raises his right hand again, breaks off another piece of cake from his right arm, and repeats the action of bringing it to his mouth to chew.``` </details>|<video src="https://github.com/user-attachments/assets/de5f7480-b79c-4fc1-b345-c5880a3b5f9e" width="600"> </video> <details><summary>📋 Show input prompt</summary> ```A little girl, carrying a colorful handbag, skips through the garden.  The video uses claymation style.``` </details> <details><summary>📋 Show rewrite prompt</summary> ```A little girl with a colorful handbag skips through a whimsical claymation garden. In a vibrant garden constructed entirely from clay, a young girl, meticulously crafted in a claymation style, skips joyfully. She has chunky, sculpted yellow clay hair tied in pigtails that bounce with a slight stiffness, simple black button eyes, and a wide, permanently etched smile. She wears a simple pink clay dress with a white collar. In her left hand, she carries a small handbag molded from bright red and blue clay, which swings in a slightly jerky arc as she moves. Initially, the girl lifts her right leg high, her body momentarily suspended in a classic stop-motion pose. Then, she hops forward, landing lightly as her left leg swings through for the next skip. Her arms move in an exaggerated, back-and-forth rhythm, characteristic of stop-motion animation. Her movements are intentionally not perfectly fluid, highlighting the frame-by-frame nature of the claymation technique. The garden around her is a whimsical, textured world. In the foreground and mid-ground, oversized flowers with swirled purple and orange petals stand on thick green stems. The ground is a textured mat of green clay, showing subtle fingerprints and tool marks that add to the handmade charm. In the background, a pale blue clay backdrop features a simplified, smiling sun molded from yellow clay. The shot is at an eye-level angle with the main subject. The camera follows the subject, moving smoothly to the right to keep her in the frame. The lighting is bright and even, casting soft shadows that emphasize the rounded, three-dimensional forms of the clay models. The overall video presents a charming and detailed claymation style.``` </details>|
|High Image-Video Consistency|<img src="https://github.com/user-attachments/assets/3bc8e55d-c211-454e-8067-128c0e215eb6"> <video src="https://github.com/user-attachments/assets/3e6b7ee9-ec66-4e46-a446-801b1c1a1c81" width="600"> </video> <details><summary>📋 Show input prompt</summary> ```女孩放下书，站起身，转身向屋内走去。镜头拉远。``` </details> <details><summary>📋 Show rewrite prompt</summary> ```女孩合上手中的书，将书放在身侧的窗台上。随后，她缓缓站起身，转身向屋内走去，身影逐渐没入门后的阴影中。镜头缓缓拉远，露出更多被绿植覆盖的屋檐和墙体。``` </details>|<img src="https://github.com/user-attachments/assets/7657ce60-90b5-4fdc-b713-0eaa55829b09"> <video src="https://github.com/user-attachments/assets/9ca24021-2353-40d5-8a4d-0f8e67d51826" width="600"> </video> <details><summary>📋 Show input prompt</summary> ```女人手上的鸟亲了女人一口``` </details> <details><summary>📋 Show rewrite prompt</summary> ```女人手臂上的白色鹦鹉缓缓转过头，将喙轻轻触碰女人的脸颊，随后收回头部。女人嘴角微微上扬，目光温柔地注视着鹦鹉。背景中的绿植保持静止。``` </details>|




## 📊 Evaluation

### Rating
We assess text-to-video generation using a comprehensive rating methodology that considers five key dimensions: text-video consistency, visual quality, structural stability, motion effects, and the aesthetic quality of individual frames. For image-to-video generation, the evaluation encompasses image-video consistency, instruction responsiveness, visual quality, structural stability, and motion effects.

<div align="center">
<img src="./assets/T2V_Rating.png" alt="rating result of t2v" width="800">
</div> 

---

<div align="center">
<img src="./assets/I2V_Rating.png" alt="rating result of i2v" width="800">
</div> 


### GSB
The GSB(Good/Same/Bad) approach is widely used to evaluate the relative performance of two models based on overall video perception quality.We carefully construct 300 diverse text prompts and 300 image samples to cover balanced application scenarios for both text-to-video and image-to-video tasks. For each prompt or image input, an equal number of video samples are generated by each model in a single run to ensure comparability. To maintain fairness, inference is performed only once per input without any cherry-picking of results. All competing models are evaluated using their default configurations. The evaluation is conducted by over 100 professional assessors

<div align="center">
<img src="./assets/T2V_GSB.png" alt="gsb result of t2v" width="800">
</div>

---

<div align="center">
<img src="./assets/I2V_GSB.png" alt="gsb result of i2v" width="800">
</div> 


### Inference speed
We report inference speed with basic engineering-level acceleration techniques enabled on 8 H800 GPUs to demonstrate practical performance achievable in real-world deployment scenarios.
Please note that in this experiment, we do not pursue the most extreme acceleration at the cost of generation quality, but rather to achieve notable speed improvements while maintaining nearly identical output quality.

We report the total inference time for 50 diffusion steps for HunyuanVideo 1.5 below:

<div align="center">
<img src="./assets/speed.png" alt="" width="100%">
</div> 


## 📚 Citation

```bibtex
@misc{hunyuanvideo2025，
      title={HunyuanVideo 1.5 Technical Report},
      author={Tencent Hunyuan Foundation Model Team},
      year={2025},
      publisher = {GitHub},
      howpublished = {\url{https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5}},
}
```

## 🙏 Acknowledgements
We would like to thank the contributors to the [Transformers](https://github.com/huggingface/transformers), [Diffusers](https://github.com/huggingface/diffusers) , [HuggingFace](https://huggingface.co/) and [Qwen-VL](https://github.com/QwenLM/Qwen-VL), for their open research and exploration.

## 🌟 Github Star History

<a href="https://star-history.com/#Tencent-Hunyuan/HunyuanVideo-1.5&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Tencent-Hunyuan/HunyuanVideo-1.5&type=Date1&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=Tencent-Hunyuan/HunyuanVideo-1.5&type=Date1" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=Tencent-Hunyuan/HunyuanVideo-1.5&type=Date1" />
 </picture>
</a>
