[Read in English](./README.md)

# HunyuanVideo-1.5

<div align="center">
<img src="./assets/logo.png" alt="HunyuanVideo-1.5 Logo" width="80%">

# 🎬 HunyuanVideo-1.5: 一款领先的轻量级视频生成模型

</div>


<div align="center">
<!-- <img src="./assets/banner.png" alt="HunyuanVideo-1.5 Banner" width="800"> -->

</div>


HunyuanVideo-1.5作为一款轻量级视频生成模型，仅需83亿参数即可提供顶级画质，大幅降低使用门槛。该模型在消费级显卡上运行流畅，让每位开发者和创作者都能轻松使用。本代码库提供生成创意视频所需的实现方案与工具集。


<div align="center">
  <a href="https://hunyuan.tencent.com/video/zh?tabIndex=0" target="_blank"><img src=https://img.shields.io/badge/Official%20Site-333399.svg?logo=homepage height=22px></a>
  <a href=https://huggingface.co/tencent/HunyuanVideo-1.5 target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20Models-d96902.svg height=22px></a>
  <a href=https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5 target="_blank"><img src= https://img.shields.io/badge/Page-bb8a2e.svg?logo=github height=22px></a>
  <a href="https://arxiv.org/pdf/2511.18870" target="_blank"><img src=https://img.shields.io/badge/Report-b5212f.svg?logo=arxiv height=22px></a>
  <a href=https://x.com/TencentHunyuan target="_blank"><img src=https://img.shields.io/badge/Hunyuan-black.svg?logo=x height=22px></a>
  <a href="https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/assets/HunyuanVideo_1_5_Prompt_Handbook_EN.md" target="_blank"><img src=https://img.shields.io/badge/📚-PromptHandBook-blue.svg?logo=book height=22px></a> <br/>
  <a href="./ComfyUI/README.md" target="_blank"><img src=https://img.shields.io/badge/ComfyUI-blue.svg?logo=book height=22px></a>
  <a href="https://github.com/ModelTC/LightX2V" target="_blank"><img src=https://img.shields.io/badge/LightX2V-yellow.svg?logo=book height=22px></a>
  <a href="https://tusi.cn/models/933574988890423836" target="_blank"><img src=https://img.shields.io/badge/吐司-purple.svg?logo=book height=22px></a>
  <a href="https://tensor.art/models/933574988890423836" target="_blank"><img src=https://img.shields.io/badge/TensorArt-cyan.svg?logo=book height=22px></a>
</div>


<p align="center">
    👏 加入我们的 <a href="./assets/wechat.png" target="_blank">微信社区</a> 和 <a href="https://discord.gg/ehjWMqF5wY">Discord</a> |
💻 <a href="https://hunyuan.tencent.com/video/zh?tabIndex=0">官方网站 立即体验模型！</a>&nbsp&nbsp
</p>

## 🔥🔥🔥 最新动态
* 🚀 Dec 23, 2025: 支持 Fp8 gemm 推理！🔥🔥🔥🆕
* 🚀 Dec 05, 2025: **新模型发布**：我们现已发布 [480p I2V 步数蒸馏模型](https://huggingface.co/tencent/HunyuanVideo-1.5/tree/main/transformer/480p_i2v_step_distilled)，建议使用 8 或 12 步生成视频！在 RTX 4090 上，端到端生成耗时减少 75%，单卡 RTX 4090 可在 **75 秒**内生成视频。步数蒸馏模型在保持与原模型相当质量的同时实现了显著的加速。详细的质量对比请参见[步数蒸馏对比文档](./assets/step_distillation_comparison.md)。如需更快的生成速度，您也可以尝试使用4步推理（速度更快，质量略有下降）。**启用步数蒸馏模型，请运行 `generate.py` 并使用 `--enable_step_distill` 参数。** 详细的使用说明请参见[使用方法](#-使用方法)。 🔥🔥🔥🆕
* 📚 Dec 05, 2025: **训练代码和 LoRA 微调脚本已发布**：我们现已开源 HunyuanVideo-1.5 的完整训练代码！训练脚本（`train.py`）提供了完整的训练流程，支持分布式训练、FSDP、context parallel、梯度检查点等功能。HunyuanVideo-1.5 使用 Muon 优化器进行训练，我们在[训练](#-训练)部分已开源。**如果您希望继续训练我们的模型，或使用 LoRA 进行微调，请使用 Muon 优化器。** 详细使用说明请参见[训练](#-训练)部分。 🔥🔥🔥🆕
* 🎉 **Diffusers 支持**：HunyuanVideo-1.5 现已支持 Hugging Face Diffusers！查看我们的 [Diffusers 集合](https://huggingface.co/collections/hunyuanvideo-community/hunyuanvideo-15) 以便轻松集成。 🔥🔥🔥🆕
* 🚀 Nov 27, 2025: 我们现已支持 cache 推理（deepcache, teacache, taylorcache），可极大加速推理！请 pull 最新代码体验。 🔥🔥🔥🆕 
* 🚀 Nov 24, 2025: 我们现已支持 deepcache 推理。
* 👋 Nov 20, 2025: 我们开源了 HunyuanVideo-1.5的代码和推理权重

## 🎥 演示视频
<div align="center">
  <video src="https://github.com/user-attachments/assets/d45ec78e-ea40-47f1-8d4d-f4d9a0682e2d" width="60%"> </video>
</div>

## 🧩 社区贡献

如果您在项目中使用或开发了 HunyuanVideo-1.5，欢迎告知我们。

- **Diffusers** - [HunyuanVideo-1.5 Diffusers](https://huggingface.co/collections/hunyuanvideo-community/hunyuanvideo-15): HunyuanVideo-1.5 的官方 Hugging Face Diffusers 集成。使用 Diffusers 库轻松使用 HunyuanVideo-1.5，无缝集成到您的项目中。详情请参阅[使用 Diffusers](#使用-diffusers) 部分。

- **ComfyUI** - [ComfyUI](https://github.com/comfyanonymous/ComfyUI): 一个强大且模块化的扩散模型图形界面，采用节点式工作流。ComfyUI 支持 HunyuanVideo-1.5，并提供多种工程加速优化以实现快速推理。
我们提供了一个 [ComfyUI 使用指南](./ComfyUI/README.md) 用于 HunyuanVideo-1.5。
- **社区实现的 ComfyUI 插件** - [comfyui_hunyuanvideo_1.5_plugin](https://github.com/yuanyuan-spec/comfyui_hunyuanvideo_1.5_plugin): 社区实现的 HunyuanVideo-1.5 ComfyUI 插件，提供简化版和完整版节点集，支持快速使用或深度工作流定制，内置自动模型下载功能。

- **LightX2V** - [LightX2V](https://github.com/ModelTC/LightX2V): 一个轻量级高效的视频生成框架，集成了 HunyuanVideo-1.5，支持多种工程加速技术以实现快速推理。

- **Wan2GP v9.62** - [Wan2GP](https://github.com/deepbeepmeep/Wan2GP): Wan2GP 是一款对显存要求非常低的应用（在 Hunyuan Video 1.5 下最低仅需 6GB 显存），支持 Lora 加速器实现 8 步生成，并且提供多种视频生成辅助工具。

- **ComfyUI-MagCache** - [ComfyUI-MagCache](https://github.com/Zehong-Ma/ComfyUI-MagCache): MagCache 是一种无需训练的缓存方法，通过估计模型输出在不同时间步之间的波动差异来加速视频生成。在 20 步推理下，可为 HunyuanVideo-1.5 实现 1.7 倍加速。

- **LeMiCa** - [LeMiCa](https://github.com/UnicomAI/LeMiCa): 中国联通数据科学与人工智能研究院提出的一种无需训练、全局最优建模的缓存加速框架。LeMiCa 可为 HunyuanVideo-1.5 提供多档可调加速配置，最高可实现 2.9×(T2V) 与 3.9×(I2V) 的推理加速。详情请参阅[LeMiCa4HunyuanVideo1.5](https://github.com/UnicomAI/LeMiCa/tree/main/LeMiCa4HunyuanVideo1.5)。


## 📑 开源计划
- HunyuanVideo-1.5 (文生视频/图生视频)
  - [x] 推理代码和模型权重
  - [x] 支持 ComfyUI
  - [x] 支持 LightX2V
  - [x] Diffusers 支持
  - [ ] 发布所有模型权重（稀疏注意力、蒸馏模型和超分辨率模型）


## 📋 目录
- [🔥🔥🔥 最新动态](#-最新动态)
- [🎥 演示视频](#-演示视频)
- [🧩 社区贡献](#-社区贡献)
- [📑 开源计划](#-开源计划)
- [📖 模型介绍](#-模型介绍)
- [✨ 核心特性](#-核心特性)
- [📜 系统要求](#-系统要求)
- [🛠️ 依赖安装](#️-依赖安装)
- [🧱 下载预训练模型](#-下载预训练模型)
- [📝 提示词指南](#-提示词指南)
- [🔑 推理](#-推理)
  - [使用源代码推理](#使用源代码推理)
  - [使用 Diffusers](#使用-diffusers)
  - [命令行参数](#命令行参数)
  - [最优推理配置](#最优推理配置)
- [🎓 训练](#-训练)
- [🎬 更多示例](#-更多示例)
- [📊 性能评估](#-性能评估)
- [📚 引用](#-引用)
- [🙏 致谢](#-致谢)
- [🌟 GitHub Star 历史](#-github-star-历史)


## 📖 Introduction
我们推出了 HunyuanVideo-1.5，一个轻量级但功能强大的视频生成模型。该模型仅使用8.3B参数就实现了开源最先进的视觉质量和运动连贯性，并能在消费级 GPU 上进行高效推理。这一成果基于几个关键组件，包括精细的数据整理、采用稀疏注意力SSTA的DiT 架构、通过专用 OCR 编码增强的双语理解能力、渐进式预训练和后训练，以及高效的视频超分辨率网络。利用这些设计，我们开发了一个统一的框架，能够跨多种时长和分辨率生成高质量的文生视频和图生视频。大量实验证明，这个紧凑而高效的模型在开源模型中确立了新的技术标杆。通过发布 HunyuanVideo-1.5 的代码和权重，我们为社区提供了一个高性能的基础，显著降低了视频创作和研究的成本，使先进的视频生成技术对所有人更加触手可及。

## ✨ Key Features
- **轻量级高性能架构**：我们提出了一种高效架构，将 83 亿参数的 Diffusion Transformer（DiT）与 3D 因果 VAE 相结合，在空间维度实现了 16 倍的压缩，在时间轴上实现了 4 倍的压缩。此外，创新的 SSTA机制修剪了冗余的时空 kv 块，显著减少了长视频序列的计算开销，并加速了推理，在 10 秒 720p 视频合成中，相比 FlashAttention-3 实现了端到端 $1.87 \times $ 的加速。


<div align="center">
<img src="./assets/hy_video_1_5_dit.png" alt="HunyuanVideo-1.5 DiT" width="600">
</div> 


- **视频超分辨率增强**：我们开发了一个高效的少步数超分辨率网络，可将输出上采样至 1080p。它在增强锐度的同时校正失真，从而优化细节和整体视觉纹理。

<div align="center">
<img src="./assets/hy_video_1_5_vsr.png" alt="HunyuanVideo-1.5 VSR" width="600">
</div> 

- **端到端训练优化**：本工作采用了多阶段、渐进式的训练策略，覆盖了从预训练到后训练的整个流程。结合 Muon 优化器加速收敛，这种方法整体上优化了运动连贯性、美学质量和对人类偏好的对齐，实现了专业级的内容生成。


## 📜系统要求

### 硬件要求

- **GPU**：支持 CUDA 的 NVIDIA GPU
- **最低 GPU 显存**：14 GB（启用模型卸载时）
  
  > **注意：** 上述内存要求是在启用模型卸载的情况下测量的。如果您的 GPU 有足够的显存，可以禁用卸载以提高推理速度。

### 软件要求

- **操作系统**：Linux
- **Python**：Python 3.10 或更高版本
- **CUDA**：与您的 PyTorch 安装兼容的 CUDA 版本

## 🛠️ 依赖安装

### 步骤 1：克隆仓库

```bash
git clone https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5.git
cd HunyuanVideo-1.5
```

### 步骤 2：安装基础依赖

```bash
pip install -r requirements.txt
pip install -i https://mirrors.tencent.com/pypi/simple/ --upgrade tencentcloud-sdk-python
```

### 步骤 3：安装注意力库

* Flash Attention: 
  安装 Flash Attention 以实现更快的推理速度和更低的 GPU 内存消耗。
  详细安装说明请参考 [Flash Attention](https://github.com/Dao-AILab/flash-attention)。

* Flex-Block-Attention: 
  flex-block-attn 仅在使用稀疏注意力以实现更快推理时需要，可以通过以下命令安装：
  ```bash
  git clone https://github.com/Tencent-Hunyuan/flex-block-attn.git
  cd flex-block-attn
  git submodule update --init --recursive
  python3 setup.py install
  ```

* SageAttention: 
  要启用 SageAttention 以实现更快的推理，您需要通过以下命令安装：
  > **注意**: 启用 SageAttention 将自动禁用 Flex-Block-Attention。
  ```bash
  git clone https://github.com/cooper1637/SageAttention.git
  cd SageAttention 
  export EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=32 # Optional
  python3 setup.py install
  ```

* SGL-Kernel:
  要启用 fp8 量化的 gemm，您需要通过以下命令安装：
  ```bash
  pip install sgl-kernel==0.3.18
  ```

## 🧱 下载预训练模型

> 💡 蒸馏模型和稀疏注意力模型即将发布，敬请期待。请关注 Hugging Face 模型卡片获取最新更新。

在生成视频之前，请先下载预训练模型。详细说明请参考 [checkpoints-download.md](checkpoints-download.md)。

### 模型卡片
|模型名称| 下载链接                     |
|-|---------------------------| 
|HunyuanVideo-1.5-480P-T2V|[480P-T2V](https://huggingface.co/tencent/HunyuanVideo-1.5/tree/main/transformer/480p_t2v) |
|HunyuanVideo-1.5-480P-I2V |[480P-I2V](https://huggingface.co/tencent/HunyuanVideo-1.5/tree/main/transformer/480p_i2v) |
|HunyuanVideo-1.5-480P-T2V-cfg-distill | [480P-T2V-cfg-distill](https://huggingface.co/tencent/HunyuanVideo-1.5/tree/main/transformer/480p_t2v_distilled) |
|HunyuanVideo-1.5-480P-I2V-cfg-distill |[480P-I2V-cfg-distill](https://huggingface.co/tencent/HunyuanVideo-1.5/tree/main/transformer/480p_i2v_distilled) |
|HunyuanVideo-1.5-480P-I2V-step-distill |[480P-I2V-step-distill](https://huggingface.co/tencent/HunyuanVideo-1.5/tree/main/transformer/480p_i2v_step_distilled) |
|HunyuanVideo-1.5-720P-T2V|[720P-T2V](https://huggingface.co/tencent/HunyuanVideo-1.5/tree/main/transformer/720p_t2v) |
|HunyuanVideo-1.5-720P-I2V |[720P-I2V](https://huggingface.co/tencent/HunyuanVideo-1.5/tree/main/transformer/720p_i2v) |
|HunyuanVideo-1.5-720P-T2V-cfg-distill| Comming soon |
|HunyuanVideo-1.5-720P-I2V-cfg-distill |[720P-I2V-cfg-distill](https://huggingface.co/tencent/HunyuanVideo-1.5/tree/main/transformer/720p_i2v_distilled) |
|HunyuanVideo-1.5-720P-T2V-sparse-cfg-distill| Comming soon |
|HunyuanVideo-1.5-720P-I2V-sparse-cfg-distill |[720P-I2V-sparse-cfg-distill](https://huggingface.co/tencent/HunyuanVideo-1.5/tree/main/transformer/720p_i2v_distilled_sparse) |
|HunyuanVideo-1.5-720P-sr-step-distill |[720P-sr](https://huggingface.co/tencent/HunyuanVideo-1.5/tree/main/transformer/720p_sr_distilled) |
|HunyuanVideo-1.5-1080P-sr-step-distill |[1080P-sr](https://huggingface.co/tencent/HunyuanVideo-1.5/tree/main/transformer/1080p_sr_distilled) |

## 📝 提示词指南
### 提示词撰写手册
提示词增强在我们的模型生成高质量视频方面起着至关重要的作用。通过撰写更长、更详细的提示词，生成的视频质量将得到显著改善。我们鼓励您编写全面且描述性的提示词，以获得最佳的视频质量。我们建议社区伙伴参考我们的官方指南，了解如何撰写有效的提示词。


**参考：** **[HunyuanVideo-1.5 提示词手册](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/assets/HunyuanVideo_1_5_Prompt_Handbook_EN.md)**


### 自动提示词增强的系统提示词
对于希望为其他大模型优化提示词的用户，建议参考文件 `hyvideo/utils/rewrite/t2v_prompt.py` 中 `t2v_rewrite_system_prompt` 的定义来指导文生视频的提示词重写。同样，对于图生视频重写，请参考 `hyvideo/utils/rewrite/i2v_prompt.py` 中 `i2v_rewrite_system_prompt` 的定义。


## 🔑 推理

### 使用源代码推理


对于提示词重写，我们推荐使用 Gemini 或通过 vLLM 部署的大模型。当前代码库仅支持兼容 vLLM 接口的模型，如果您希望使用 Gemini，需自行实现相关接口调用。

对于 vLLM 接口的模型，需要注意 T2V 和 I2V 推荐使用不同的模型和环境变量：

- 文生视频（T2V）：推荐使用 [Qwen3-235B-A22B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507)，并配置 `T2V_REWRITE_BASE_URL` 与 `T2V_REWRITE_MODEL_NAME`
- 图生视频（I2V）：推荐使用 [Qwen3-VL-235B-A22B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct)，并配置 `I2V_REWRITE_BASE_URL` 与 `I2V_REWRITE_MODEL_NAME`

> 你也可以将上述模型名替换为任何你已部署、与 vLLM 兼容的模型（包括 HuggingFace 等模型）。
>
> 默认为开启提示词重写（`--rewrite` 默认值为 `true`）。若需显式关闭，可以使用 `--rewrite false` 或 `--rewrite 0`。如果未配置 vLLM 提示词重写相关服务，管道会在本地直接生成，无远程重写。

示例：生成视频（支持 T2V/I2V。T2V 模式下设置 `IMAGE_PATH=none`，I2V 模式下指定图像路径）

> 💡 **提示**：为了更快的推理速度，您可以使用 `--enable_step_distill` 参数启用步数蒸馏模型。步数蒸馏模型（480p I2V）可使用 8 或 12 步（推荐）生成视频，在 RTX 4090 上可提速高达 75%，同时保持相当的质量。
>
> **Tips:** 如果您的 GPU 内存 > 14GB 但您在生成过程中遇到 OOM (Out of Memory) 错误，可以尝试在运行前设置以下环境变量：
> ```bash
> export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
> ```
> 
> **Tips:** 如果您有 CPU 内存有限并且遇到推理时的 OOM 错误，可以尝试禁用重叠组卸载，通过添加以下参数：
> ```bash
> --overlap_group_offloading false
> ```

```bash
export T2V_REWRITE_BASE_URL="<your_vllm_server_base_url>"
export T2V_REWRITE_MODEL_NAME="<your_model_name>"
export I2V_REWRITE_BASE_URL="<your_vllm_server_base_url>"
export I2V_REWRITE_MODEL_NAME="<your_model_name>"

PROMPT='A girl holding a paper with words "Hello, world!"'

IMAGE_PATH=/path/to/image.png # 可选，none 或 <图像路径> 以启用 i2v 模式
SEED=1
ASPECT_RATIO=16:9
RESOLUTION=480p
OUTPUT_PATH=./outputs/output.mp4
MODEL_PATH=./ckpts # 预训练模型路径

# 加速推理配置
N_INFERENCE_GPU=8 # 并行推理 GPU 数量
CFG_DISTILLED=true # 使用 CFG 蒸馏模型进行推理，2倍加速
SAGE_ATTN=true # 使用 SageAttention 进行推理
SPARSE_ATTN=false # 使用稀疏注意力进行推理（仅 720p 模型配备了稀疏注意力）。请确保 flex-block-attn 已安装
OVERLAP_GROUP_OFFLOADING=true # 仅在组卸载启用时有效，会显著增加 CPU 内存占用，但能够提速
ENABLE_CACHE=true # 启用特征缓存进行推理。显著提升推理速度
CACHE_TYPE=deepcache # 支持：deepcache, teacache, taylorcache
ENABLE_STEP_DISTILL=true # 启用 480p I2V 步数蒸馏模型，推荐 8 或 12 步，最高可达 6 倍加速


# 提升质量配置
REWRITE=true # 启用提示词重写。请确保 rewrite vLLM server 已部署和配置。
ENABLE_SR=true # 启用超分辨率

torchrun --nproc_per_node=$N_INFERENCE_GPU generate.py \
  --prompt "$PROMPT" \
  --image_path $IMAGE_PATH \
  --resolution $RESOLUTION \
  --aspect_ratio $ASPECT_RATIO \
  --seed $SEED \
  --rewrite $REWRITE \
  --cfg_distilled $CFG_DISTILLED \
  --enable_step_distill $ENABLE_STEP_DISTILL \
  --sparse_attn $SPARSE_ATTN --use_sageattn $SAGE_ATTN \
  --enable_cache $ENABLE_CACHE --cache_type $CACHE_TYPE \
  --overlap_group_offloading $OVERLAP_GROUP_OFFLOADING \
  --sr $ENABLE_SR --save_pre_sr_video \
  --output_path $OUTPUT_PATH \
  --model_path $MODEL_PATH
```

### 命令行参数

| 参数 | 类型 | 是否必需 | 默认值 | 描述 |
|----------|------|----------|---------|-------------|
| `--prompt` | str | 是 | - | 用于视频生成的文本提示 |
| `--negative_prompt` | str | 否 | `''` | 用于视频生成的负向提示词 |
| `--resolution` | str | 是 | - | 视频分辨率：`480p` 或 `720p` |
| `--model_path` | str | 是 | - | 预训练模型目录的路径 |
| `--aspect_ratio` | str | 否 | `16:9` | 输出视频的宽高比 |
| `--num_inference_steps` | int | 否 | `50` | 推理步数 |
| `--video_length` | int | 否 | `121` | 要生成的帧数 |
| `--seed` | int | 否 | `123` | 随机种子，用于可复现性 |
| `--image_path` | str | 否 | `None` | 参考图像的路径（启用图生视频模式）。使用 `none` 或 `None` 可明确使用文生视频模式 |
| `--output_path` | str | 否 | `None` | 输出文件路径（如果未提供，则保存到 `./outputs/output_{transformer_version}_{timestamp}.mp4`） |
| `--sr` | bool | 否 | `true` | 启用超分辨率（使用 `--sr false` 或 `--sr 0` 来禁用） |
| `--save_pre_sr_video` | bool | 否 | `false` | 保存超分辨率处理前的原始视频（使用 `--save_pre_sr_video` 或 `--save_pre_sr_video true` 来启用，仅在启用超分辨率时有效） |
| `--rewrite` | bool | 否 | `true` | 启用提示词重写（使用 `--rewrite false` 或 `--rewrite 0` 来禁用，禁用可能导致视频生成质量降低） |
| `--cfg_distilled` | bool | 否 | `false` | 启用 CFG 蒸馏模型以加速推理（约 2 倍加速，使用 `--cfg_distilled` 或 `--cfg_distilled true` 来启用） |
| `--enable_step_distill` | bool | 否 | `false` | 启用 480p I2V 步数蒸馏模型（推荐 8 或 12 步，在 RTX 4090 上可提速约 75%，使用 `--enable_step_distill` 或 `--enable_step_distill true` 来启用） |
| `--sparse_attn` | bool | 否 | `false` | 启用稀疏注意力以加速推理（约 1.5-2 倍加速，需要 H 系列 GPU，会自动启用 CFG 蒸馏，使用 `--sparse_attn` 或 `--sparse_attn true` 来启用） |
| `--offloading` | bool | 否 | `true` | 启用 CPU 卸载（使用 `--offloading false` 或 `--offloading 0` 来禁用，如果 GPU 内存允许，禁用后速度会更快） |
| `--group_offloading` | bool | 否 | `None` | 启用组卸载（默认：None，如果启用了 offloading 则自动启用。使用 `--group_offloading` 或 `--group_offloading true/1` 来启用，`--group_offloading false/0` 来禁用） |
| `--overlap_group_offloading` | bool | 否 | `true` | 启用重叠组卸载（默认：true）。会显著增加 CPU 内存占用，但能够提速。使用 `--overlap_group_offloading` 或 `--overlap_group_offloading true/1` 来启用，`--overlap_group_offloading false/0` 来禁用 |
| `--dtype` | str | 否 | `bf16` | Transformer 的数据类型：`bf16`（更快，内存占用更低）或 `fp32`（质量更好，速度更慢，内存占用更高） |
| `--use_sageattn` | bool | 否 | `false` | 启用 SageAttention（使用 `--use_sageattn` 或 `--use_sageattn true/1` 来启用，`--use_sageattn false/0` 来禁用） |
| `--sage_blocks_range` | str | 否 | `0-53` | SageAttention 块范围（例如：`0-5` 或 `0,1,2,3,4,5`） |
| `--enable_torch_compile` | bool | 否 | `false` | 启用 torch compile 以优化 transformer（使用 `--enable_torch_compile` 或 `--enable_torch_compile true/1` 来启用，`--enable_torch_compile false/0` 来禁用） |
| `--enable_cache` | bool | 否 | `false` | 启用 transformer 缓存（使用 `--enable_cache` 或 `--enable_cache true/1` 来启用，`--enable_cache false/0` 来禁用） |
| `--cache_type` | str | 否 | `deepcache` | Transformer 的缓存类型（例如：`deepcache, teacache, taylorcache`） |
| `--no_cache_block_id` | str | 否 | `53` | 从 deepcache 中排除的块（例如：`0-5` 或 `0,1,2,3,4,5`） |
| `--cache_start_step` | int | 否 | `11` | 使用缓存时跳过的起始步数 |
| `--cache_end_step` | int | 否 | `45` | 使用缓存时跳过的结束步数 |
| `--total_steps` | int | 否 | `50` | 总推理步数 |
| `--cache_step_interval` | int | 否 | `4` | 使用缓存时跳过的步数间隔 |

**注意：** 使用 `--nproc_per_node` 指定使用的 GPU 数量。例如，`--nproc_per_node=8` 表示使用 8 个 GPU。

### 最优推理配置

下表提供了每个模型的最优推理配置（CFG 缩放、嵌入 CFG 缩放、流偏移和推理步数），以获得最佳生成质量：

| 模型 | CFG 缩放 | 嵌入 CFG 缩放 | 流偏移 | 推理步数 |
|-------|-----------|-------------------|------------|-----------------|
| 480p T2V | 6 | None | 5 | 50 |
| 480p I2V | 6 | None | 5 | 50 |
| 720p T2V | 6 | None | 9 | 50 |
| 720p I2V | 6 | None | 7 | 50 |
| 480p T2V cfg 蒸馏 | 1 | None | 5 | 50 |
| 480p I2V cfg 蒸馏 | 1 | None | 5 | 50 |
| 480p I2V 步数蒸馏 | 1 | None | 7 | 8 或 12（推荐） |
| 720p T2V cfg 蒸馏 | 1 | None | 9 | 50 |
| 720p I2V cfg 蒸馏 | 1 | None | 7 | 50 |
| 720p T2V cfg 蒸馏稀疏 | 1 | None | 9 | 50 |
| 720p I2V cfg 蒸馏稀疏 | 1 | None | 7 | 50 |
| 480→720 超分 步数蒸馏 | 1 | None | 2 | 6 |
| 720→1080 超分 步数蒸馏 | 1 | None | 2 | 8 |

**请注意我们提供的cfg蒸馏模型，需要50步的推理步数来获得正确的结果.**

### 使用 Diffusers

HunyuanVideo-1.5 现已支持 Hugging Face Diffusers！您可以使用 Diffusers 库轻松使用：

**基础使用：**

```python
import torch

dtype = torch.bfloat16
device = "cuda:0"

from diffusers import HunyuanVideo15Pipeline
from diffusers.utils import export_to_video

pipe = HunyuanVideo15Pipeline.from_pretrained("hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v", torch_dtype=dtype)
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

generator = torch.Generator(device=device).manual_seed(seed)

video = pipe(
    prompt=prompt,
    generator=generator,
    num_frames=121,
    num_inference_steps=50,
).frames[0]

export_to_video(video, "output.mp4", fps=24)
```

**使用注意力后端优化：**

HunyuanVideo-1.5 使用可变长度序列的注意力掩码。为了获得最佳性能，我们建议使用能够高效处理填充的注意力后端。

我们建议安装 kernels（`pip install kernels`）以访问预构建的注意力内核。

```python
import torch

dtype = torch.bfloat16
device = "cuda:0"

from diffusers import HunyuanVideo15Pipeline, attention_backend
from diffusers.utils import export_to_video

pipe = HunyuanVideo15Pipeline.from_pretrained("hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v", torch_dtype=dtype)
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

generator = torch.Generator(device=device).manual_seed(seed)

with attention_backend("_flash_3_hub"): # 如果您不在 H100/H800 上，可以使用 `"flash_hub"`
    video = pipe(
        prompt=prompt,
        generator=generator,
        num_frames=121,
        num_inference_steps=50,
    ).frames[0]
    export_to_video(video, "output.mp4", fps=24)
```

更多详情，请访问 [HunyuanVideo-1.5 Diffusers 集合](https://huggingface.co/collections/hunyuanvideo-community/hunyuanvideo-15)。


## 🎓 训练

HunyuanVideo-1.5 使用 **Muon 优化器**进行训练，该优化器能够加速收敛并提高训练稳定性。Muon 优化器结合了基于动量的更新和 Newton-Schulz 正交化方法，可高效优化大规模视频生成模型。

### 快速开始

训练脚本（`train.py`）为 HunyuanVideo-1.5 提供了完整的训练流程。使用方法如下：

#### 1. 实现您的数据加载器

替换 `train.py` 中的 `create_dummy_dataloader()` 函数，实现您自己的数据加载器。数据集的 `__getitem__` 方法应返回单个样本。

- **必需字段：**
  - `"pixel_values"`: `torch.Tensor` - 视频：`[C, F, H, W]` 或图像：`[C, H, W]`
    - 像素值必须在 `[-1, 1]` 范围内
    - 注意：对于视频数据，时间维度 F 必须是 `4n+1`（例如：1, 5, 9, 13, 17, ...）
  - `"text"`: `str` - 该样本的文本提示词
  - `"data_type"`: `str` - `"video"` 或 `"image"`

- **可选字段（用于性能优化）：**
  - `"latents"`: 预编码的 VAE 潜在表示（跳过 VAE 编码以加速训练）
  - `"byt5_text_ids"` 和 `"byt5_text_mask"`: 预分词的 byT5 输入

详细的格式文档请参见 `train.py` 中的 `create_dummy_dataloader()` 函数。

#### 2. 运行训练

**单 GPU：**
```bash
python train.py --pretrained_model_root <预训练模型路径> [其他参数]
```

**多 GPU：**
```bash
N=8
torchrun --nproc_per_node=$N train.py --pretrained_model_root <预训练模型路径> [其他参数]
```

**示例：**
```bash
torchrun --nproc_per_node=8 train.py \
  --pretrained_model_root ./ckpts \
  --learning_rate 1e-5 \
  --batch_size 1 \
  --max_steps 10000 \
  --output_dir ./outputs \
  --enable_fsdp \
  --enable_gradient_checkpointing \
  --sp_size 8
```

#### 3. 关键训练参数

| 参数 | 描述 | 默认值 |
|-----------|-------------|---------|
| `--pretrained_model_root` | 预训练模型路径（必需） | - |
| `--learning_rate` | 学习率 | 1e-5 |
| `--batch_size` | 批次大小 | 1 |
| `--max_steps` | 最大训练步数 | 10000 |
| `--warmup_steps` | 预热步数 | 500 |
| `--gradient_accumulation_steps` | 梯度累积步数 | 1 |
| `--enable_fsdp` | 启用 FSDP 进行分布式训练 | true |
| `--enable_gradient_checkpointing` | 启用梯度检查点 | true |
| `--sp_size` | 序列并行大小（必须能整除 world_size） | 8 |
| `--i2v_prob` | 视频数据使用 i2v 任务的概率 | 0.3 |
| `--use_muon` | 使用 Muon 优化器 | true |
| `--resume_from_checkpoint` | 从检查点目录恢复训练 | None |
| `--use_lora` | 启用 LoRA 微调 | false |
| `--lora_r` | LoRA rank | 8 |
| `--lora_alpha` | LoRA alpha 缩放参数 | 16 |
| `--lora_dropout` | LoRA dropout 率 | 0.0 |
| `--pretrained_lora_path` | 预训练 LoRA 适配器路径 | None |

#### 4. 监控训练

- 检查点按 `--save_interval` 指定的间隔保存到 `output_dir`
- 验证视频按 `--validation_interval` 指定的间隔生成
- 训练日志按 `--log_interval` 指定的间隔打印到控制台

#### 5. 恢复训练

使用 `--resume_from_checkpoint <检查点目录>` 从保存的检查点恢复训练：
```bash
python train.py \
  --pretrained_model_root <路径> \
  --resume_from_checkpoint ./outputs/checkpoint-1000
```

#### 6. LoRA 微调

启用 LoRA 微调，在训练命令中添加 `--use_lora`。LoRA 适配器将保存在检查点目录的 `lora/` 子目录下：

```bash
torchrun --nproc_per_node=8 train.py \
  --pretrained_model_root ./ckpts \
  --use_lora \
  --lora_r 8 \
  --lora_alpha 16 \
  --learning_rate 1e-4 \
  --output_dir ./outputs
```

加载预训练的 LoRA 适配器，使用 `--pretrained_lora_path`：
```bash
torchrun --nproc_per_node=8 train.py \
  --pretrained_model_root ./ckpts \
  --use_lora \
  --pretrained_lora_path ./outputs/checkpoint-1000/lora/default
```


## 📊 性能评估
### 评分
我们使用全面的评分方法来评估文生视频生成，考虑了五个关键维度：文本-视频一致性、视觉质量、结构稳定性、运动效果以及单帧的美学质量。对于图生视频生成，评估包括图像-视频一致性、指令响应性、视觉质量、结构稳定性和运动效果。

<div align="center">
<img src="./assets/T2V_Rating.png" alt="rating result of t2v" width="800">
</div> 

---

<div align="center">
<img src="./assets/I2V_Rating.png" alt="rating result of i2v" width="800">
</div> 


### GSB
GSB（Good/Same/Bad）评估法被广泛用于基于整体视频感知质量来评估两个模型的相对性能。我们精心构建了300个多样化文本提示词和300个图像样本，以覆盖文本生成视频和图像生成视频任务的平衡应用场景。针对每个提示词或图像输入，各模型均在单次运行中生成同等数量的视频样本以确保可比性。为保持公平性，每个输入仅执行一次推理且不进行任何结果筛选。所有参与对比的模型均采用其默认配置进行评估，并由百余名专业评估员完成评测过程。


<div align="center">
<img src="./assets/T2V_GSB.png" alt="rating result of t2v" width="800">
</div>

---

<div align="center">
<img src="./assets/I2V_GSB.png" alt="gsb result of i2v" width="800">
</div> 

### 推理速度
我们在8块H800 GPU上启用了基础工程级加速技术，报告推理速度，以展示在实际部署场景中可实现的实用性能。
请注意，在本实验中，我们不以牺牲生成质量为代价追求最极端的加速，而是在保持几乎相同的输出质量的同时实现显著的速度提升。

我们在下方报告了HunyuanVideo-1.5在50个扩散步数下的总推理时间：

<div align="center">
<img src="./assets/speed.png" alt="" width="100%">
</div> 

## 🎬 更多示例
|特性|示例1|示例2|
|------|------|------|
|指令跟随能力|<video src="https://github.com/user-attachments/assets/fdc3c27b-69f5-46a1-b707-0b57510fa32f" width="600"> </video> <details><summary>📋 Show input prompt</summary> ```一名哀伤的黑发中国女子凝望天空，复古胶片风格烘托出怀旧戏剧氛围``` </details> <details><summary>📋 Show rewrite prompt</summary> ```俯视角度，一位有着深色，略带凌乱的长卷发的年轻中国女性，佩戴着闪耀的珍珠项链和圆形金色耳环，她凌乱的头发被风吹散，她微微抬头，望向天空，神情十分哀伤，眼中含着泪水。嘴唇涂着红色口红。背景是带有华丽红色花纹的图案。画面呈现复古电影风格，色调低饱和，带着轻微柔焦，烘托情绪氛围，质感仿佛20世纪90年代的经典胶片风格，营造出怀旧且富有戏剧性的感觉。``` </details>|<video src="https://github.com/user-attachments/assets/3fcb42cc-cdd3-4651-86a6-645a858561c4" width="600"> </video> <details><summary>📋 Show input prompt</summary> ```建筑蓝图上的线条化为实体，瞬间生长出一个完整的复古工业风办公空间。``` </details> <details><summary>📋 Show rewrite prompt</summary> ```一座空旷的现代阁楼里，有一张铺展在地板中央的建筑蓝图。忽然间，图纸上的线条泛起微光，仿佛被某种无形的力量唤醒。紧接着，那些发光的线条开始向上延伸，从平面中挣脱，勾勒出立体的轮廓——就像在空中进行一场无声的3D打印。随后，奇迹在加速发生：极简的橡木办公桌、优雅的伊姆斯风格皮质椅、高挑的工业风金属书架，还有几盏爱迪生灯泡，以光纹为骨架迅速“生长”出来。转瞬间，线条被真实的材质填充——木材的温润、皮革的质感、金属的冷静，都在眨眼间完整呈现。最终，所有家具稳固落地，蓝图的光芒悄然褪去。一个完整的办公空间，就这样从二维的图纸中诞生。``` </details>|
|流畅运动生成|<video src="https://github.com/user-attachments/assets/447847f0-490a-45f9-a86d-a67ab1ff4231" width="600"> </video> <details><summary>📋 Show input prompt</summary> ```A DJ is immersed in his musical world. He wears a pair of professional, matte-black headphones, revealing a focused expression. He wears a black bomber jacket, zipped open to reveal a T-shirt underneath. His upper body sways back and forth rhythmically to the throbbing electronic beats, his head moving with precise movement. The mixing console in front of him serves as the primary source of light. In the distance, the cool white glow of several stadium floodlights casts a deep, dark haze across the vast field, casting long shadows across the emerald green grass, creating a stark contrast to the brightly lit area surrounding the DJ booth. His hands danced swiftly and precisely across the equipment. The entire scene was filled with high-tech dynamics and the solitary creative passion. Against the backdrop of the vast and silent night stadium, it created an atmosphere of high focus, energy, and a slightly surreal feeling.``` </details> <details><summary>📋 Show rewrite prompt</summary> ```slowly advancing medium shot, shot from a level angle, focuses on the center of an empty football field, where a DJ is immersed in his musical world. He wears a pair of professional, matte-black headphones, one earcup slightly removed, revealing a focused expression and a brow beaded with sweat from his intense concentration. He wears a black bomber jacket, zipped open to reveal a T-shirt underneath. His upper body sways back and forth rhythmically to the throbbing electronic beats, his head moving with precise movement. The mixing console in front of him serves as the primary source of light. In the distance, the cool white glow of several stadium floodlights casts a deep, dark haze across the vast field, casting long shadows across the emerald green grass, creating a stark contrast to the brightly lit area surrounding the DJ booth. His hands danced swiftly and precisely across the equipment, one hand steadily pushing and pulling a long volume fader, while the fingers of the other nimbly jumped between the illuminated knobs and pads, sometimes decisively cutting a bass line, sometimes triggering an echo effect. The entire scene was filled with high-tech dynamics and the solitary creative passion. Against the backdrop of the vast and silent night stadium, it created an atmosphere of high focus, energy, and a slightly surreal feeling.``` </details>|<video src="https://github.com/user-attachments/assets/49057fe8-a102-4fd7-bd92-e9561abb9f45" width="600"> </video> <details><summary>📋 Show input prompt</summary> ```A figure skater performs a rapid, graceful Biellmann spin, captured from all angles.``` </details> <details><summary>📋 Show rewrite prompt</summary> ```The video captures a figure skater performing a Biellmann spin on ice. The subject is a female skater in a glittering costume. Initially, she spins on one leg. Then, she reaches back and pulls her free leg up. Next, she spins rapidly, becoming a blur of motion, with ice shavings spraying from her skate blade. The background is an ice rink with blurred advertising boards. The camera circles around the subject to capture the spin from all angles. The lighting is spotlit, creating lens flares and sparkles on her costume. The overall video presents a graceful artistic sports style.``` </details>|
|电影级美学|<video src="https://github.com/user-attachments/assets/4098cf72-357d-4b81-97df-6752064ce0c3" width="600"> </video> <details><summary>📋 Show input prompt</summary> ```固定镜头,焦点在图片里的挂钟上，镜头轻微摇晃营造手持摄影感，​wjw,filmphotos,Film Grain,Reversal film photography，Wong Kar-wai movies,cinematic photography, HK film style,neon lighting, in the style of Wong Kar Wai film``` </details> <details><summary>📋 Show rewrite prompt</summary> ```Handheld lens shooting, the camera focuses on the wall clock hanging on the green-toned wall, shaking slightly. The second hand sweeps steadily across the clock face, and the shadow of the clock cast on the wall shifts subtly with the movement of the lens.``` </details>|<video src="https://github.com/user-attachments/assets/2b4575e5-79f1-4011-bed0-e8380198f7c9" width="600"> </video> <details><summary>📋 Show input prompt</summary> ```The leaves of calamus shine in the sunlight, dotted with dewdrops that trickle down to the ground with the breeze.``` </details> <details><summary>📋 Show rewrite prompt</summary> ```A macro shot focuses on long, slender calamus leaves, rendered in a cinematic photography realistic style. The main leaf, a vibrant, deep green, is positioned diagonally across the frame. Its surface is covered in tiny, glistening spherical dewdrops that catch and refract the bright morning sunlight, creating sparkling highlights. Initially, a larger, perfectly round dewdrop clings to the upper section of the leaf, its surface tension holding it in place. Then, as the leaf sways almost imperceptibly, the dewdrop begins to slowly dislodge. Next, it starts to trickle down the central vein of the leaf, its shape elongating slightly as it moves, leaving a subtle, glistening wet trail in its path. Finally, it reaches the pointed tip of the leaf, hangs for a brief moment, and falls out of the bottom of the frame. In the background, other leaves and blades of grass are softly blurred, creating a beautiful bokeh effect with soft, out-of-focus circles of light. The environment is bathed in the warm, golden glow of early morning sunlight, which streams in from behind the leaves, backlighting them and causing their wet edges to shine brilliantly. The overall impression is one of serene, natural beauty, captured in a highly realistic and detailed manner. This is a macro shot. The camera tilts down very slowly, following the path of the main dewdrop as it travels down the leaf. The lighting is soft and natural, with strong backlighting to create a radiant, glowing effect on the dewdrops and leaf edges, characteristic of professional nature photography. The atmosphere is peaceful and serene. The overall video presents a cinematic photography realistic style.``` </details>|
|文字渲染|<video src="https://github.com/user-attachments/assets/7c964fc5-c27e-4bd0-bf3f-eb8fca2caef6" width="600"> </video> <details><summary>📋 Show input prompt</summary> ```赛博朋克风格的夜晚街角，一个巨大的招牌上， “Hunyuan Video 1.5”的霓虹灯管轮廓已经安装好。镜头推进，霓虹灯从“H”开始，伴随着‘滋滋’的电流声，每个字母依次亮起粉紫色的光芒，直到全部点亮，照亮了潮湿的街道。赛博朋克，城市美学``` </details> <details><summary>📋 Show rewrite prompt</summary> ```On a wet street corner in a cyberpunk city at night, a large neon sign reading "Hunyuan Video 1.5" lights up sequentially, illuminating the dark, rainy environment with a pinkish-purple glow. he scene is a dark, rain-slicked street corner in a futuristic, cinematic cyberpunk city. Mounted on the metallic, weathered facade of a building is a massive, unlit neon sign. The sign's glass tube framework clearly spells out the words "Hunyuan Video 1.5". Initially, the street is dimly lit, with ambient light from distant skyscrapers creating shimmering reflections on the wet asphalt below. Then, the camera zooms in slowly toward the sign. As it moves, a low electrical sizzling sound begins. In the background, the dense urban landscape of the cyberpunk metropolis is visible through a light atmospheric haze, with towering structures adorned with their own flickering advertisements. A complex web of cables and pipes crisscrosses between the buildings. The shot is at a low angle, looking up at the sign to emphasize its grand scale. The lighting is high-contrast and dramatic, dominated by the neon glow which creates sharp, specular reflections and deep shadows. The atmosphere is moody and tech-noir. The overall video presents a cinematic photography realistic style.,``` </details>|<video src="https://github.com/user-attachments/assets/73e8b741-baec-4a40-9d36-a1435172ab64" width="600"> </video> <details><summary>📋 Show input prompt</summary> ```一张铺开的中国宣纸上，浓墨滴入水中，晕染出壮丽的山水画轮廓。山峰、云雾、孤舟在墨色中自然形成。随后，这些水墨元素巧妙地流动、重组，在画面的留白处汇聚成"Hunyuan Video 1.5"的书法字体。优雅，诗意，文化底蕴``` </details> <details><summary>📋 Show rewrite prompt</summary> ```A drop of black ink blooms on wet Chinese Xuan paper, forming a landscape painting before the ink elements fluidly reassemble into the calligraphic text "Hunyuan Video 1.5". On a flat, laid-out sheet of off-white Chinese Xuan paper with a subtle, fibrous texture, the scene unfolds. Initially, a single, concentrated drop of deep black ink falls into a clear, wet area at the center of the paper. Then, the ink instantly begins to bloom outwards in intricate, flowing tendrils of varying shades from jet-black to smoky grey. As it spreads, the ink wash naturally and rapidly forms the silhouette of a majestic mountain range with sharp, defined peaks. Next, softer, diluted grey tones billow around the mountains, creating layers of atmospheric mist and clouds, while a simple, dark stroke materializes as a lone boat on a tranquil, watery expanse at the base. As the landscape is formed, the ink elements—the lines of the mountains, wisps of cloud, and the shape of the boat—begin to deconstruct, dissolving into flowing streams of liquid ink. Finally, these streams move gracefully across the paper's empty white space, converging and elegantly reorganizing to form the text "Hunyuan Video 1.5" in a fluid, semi-cursive calligraphic style. The background is the minimalist expanse of the Xuan paper itself, its texture providing a subtle depth. The entire process is lit by soft, even, diffused light from above, which enhances the rich tonal variations of the ink and the delicate texture of the paper without creating harsh shadows. Bird's-eye view. The camera is positioned directly above the subject, capturing the entire process. The camera remains static. The aesthetic is a high-quality, dynamic Chinese ink wash animation style, perfectly simulating the real-world physics of ink spreading on wet paper. The entire sheet of paper and the final text are kept fully within the frame. Poetic, elegant, artistic. The overall video presents a dynamic Chinese ink wash animation style.``` </details>|
|物理合理性|<video src="https://github.com/user-attachments/assets/f1d74e48-cc03-415d-b75f-f7186a4fb41d" width="600"> </video> <details><summary>📋 Show input prompt</summary> ```In a sleek museum gallery, a woman pauses before a gilded oil painting. The painted man inside slowly comes alive, lifting a bottle and pouring real wine straight from the canvas into her glass. Surrounded by stylish art critics moving naturally through the hall, she accepts the pour with calm elegance, as if the impossible were routine. ``` </details> <details><summary>📋 Show rewrite prompt</summary> ```In a sleek museum gallery, a woman receives a glass of wine poured directly from an animated oil painting. A sophisticated woman with dark hair tied back elegantly stands in the mid-ground. She is wearing a simple, black silk sleeveless dress and holds a clear, crystal wine glass in her right hand. She is positioned before a large, baroque-style oil painting in an ornate, gilded frame. Inside the painting, an aristocratic man with a mustache, dressed in a dark velvet doublet with a white lace collar, is depicted. His form is defined by visible, impasto oil brushstrokes. Initially, the woman watches the painting with calm poise. Then, the painted man's arm slowly animates, his painted texture retained as he lifts a dark bottle. Next, a photorealistic stream of red wine emerges directly from the flat canvas surface, arcing through the air and splashing gently into the real crystal glass she holds. She remains perfectly still, accepting the impossible pour with a subtle, knowing smile. The setting is a modern art gallery with high white walls and polished dark concrete floors that reflect the ambient light. Focused track lighting from the high ceiling casts a warm, dramatic spotlight on the woman and the painting, creating soft shadows. In the background, two other gallery patrons, a man and a woman in stylish, modern attire, stroll slowly from right to left, their figures slightly blurred by a shallow depth of field, moving naturally through the hall. The shot is at an eye-level angle with the woman. The camera remains static, capturing the surreal event in a steady medium shot. The lighting is high-contrast and dramatic, reminiscent of a cinematic photography realistic style, using soft side lighting to accentuate the woman's features and the texture of the painting. The mood is surreal, elegant, and mysterious. The overall video presents a cinematic photography realistic style.``` </details>|<video src="https://github.com/user-attachments/assets/07bcce06-ff4f-4688-8c60-c02f600635ea" width="600"> </video> <details><summary>📋 Show input prompt</summary> ```An intact soda can is slowly crushed by a hand.``` </details> <details><summary>📋 Show rewrite prompt</summary> ```In a medium close-up, a hand slowly crushes an intact red and white soda can on a wooden table. A male hand with visible, realistic skin texture is wrapped firmly around the middle of an intact, pristine red and white aluminum soda can. The can, covered in glistening condensation droplets, rests on a dark, polished wooden surface. The cinematic realism captures every minute detail of the scene. Initially, the hand's grip is steady, with the can's cylindrical shape perfectly preserved. Then, the fingers begin to tighten slowly, the knuckles whitening slightly from the exertion. Next, the smooth aluminum surface starts to buckle under the controlled pressure, a sharp crease forming vertically down its side as the metallic sheen distorts. As the hand continues its deliberate squeeze, the can collapses inward progressively, the vibrant red paint wrinkling as the metal structure crumples. Finally, the can is left significantly crushed, its form now an irregular, crumpled shape held tightly in the fist. The scene takes place on a dark, polished wooden tabletop that catches soft, diffuse reflections. The grain of the wood is faintly discernible, adding a layer of texture to the foreground. The background is completely out of focus, rendered as a soft, dark, and non-descript blur, which isolates the main action and enhances the photorealistic quality of the shot. The shot is a medium close-up, presented in a cinematic photography realistic style. The camera remains static at a slightly high angle, looking down to provide a clear and unobstructed view of the can's deformation. Soft side lighting creates high contrast, sculpting the muscles and tendons of the hand while casting specular highlights on the metallic can and the water droplets. The atmosphere is focused and intense. The overall video presents a cinematic photography realistic style.``` </details>|
|摄像机运动|<video src="https://github.com/user-attachments/assets/6deacbfe-4cca-48d7-a2be-cb638a3e01cb" width="600"> </video> <details><summary>📋 Show input prompt</summary> ```圣诞节的家中，小女孩靠着妈妈听妈妈读书，背景是下着雪的窗外，镜头缓慢下移，一只可爱的长毛小白猫戴着圣诞帽趴在温暖的地摊上``` </details> <details><summary>📋 Show rewrite prompt</summary> ```In a cozy home on Christmas, a young girl leans against her mother as they read a book, and the camera moves down to reveal a fluffy white cat in a Santa hat resting on a warm rug. In a warmly lit living room on a snowy Christmas evening, a young mother and her little daughter are sitting together on a comfortable sofa. The mother, with a gentle expression and wearing a cream-colored knitted sweater, holds an open storybook with colorful illustrations. Her daughter, a small girl with brown hair in pigtails and a red pajama set, leans her head affectionately on her mother's shoulder, her eyes fixed on the book. On the floor below them, a fluffy, long-haired white cat is curled up on a plush, beige wool rug. The cat wears a tiny red and white Santa hat perched between its ears. Initially, the shot focuses on the mother and daughter, capturing their quiet, shared moment. The mother’s finger gently rests on the page of the book. Then, the camera slowly moves downward, gliding past the book and their laps. Finally, the camera settles at a low angle, bringing the adorable white cat into sharp focus as the primary subject. The cat's chest gently rises and falls with each breath, its eyes peacefully closed. Through a large window in the background, large, soft snowflakes can be seen falling silently against the dark blue twilight sky, creating a peaceful and serene backdrop. Faint, out-of-focus golden Christmas lights twinkle in the corner of the room, adding to the warm, festive atmosphere. The scene is imbued with a sense of comfort and holiday warmth, creating a beautiful cinematic photography realistic image. The camera slowly moves downward. The shot uses soft, warm interior lighting that casts gentle shadows, creating a high-contrast, cinematic look. A shallow depth of field keeps the focus on the subjects while beautifully blurring the background elements. The mood is heartwarming, peaceful, and festive. The overall video presents a cinematic photography realistic style.``` </details>|<video src="https://github.com/user-attachments/assets/8e72ed0f-f8ac-445b-97e5-eb4b16fbc121" width="600"> </video> <details><summary>📋 Show input prompt</summary> ```The hiker begins walking forward along the trail, causing the water bottle to swing rhythmically with each step. The camera gradually pulls back and rises to reveal a vast desert landscape stretching out ahead.``` </details> <details><summary>📋 Show rewrite prompt</summary> ```The hiker begins walking forward along the trail, causing the water bottle to swing rhythmically with each step. The camera gradually pulls back and rises to reveal a vast desert landscape stretching out ahead, while the sun position shifts from afternoon to dusk, casting increasingly longer shadows across the terrain as the figure becomes smaller in the frame.``` </details>|
|多风格支持|<video src="https://github.com/user-attachments/assets/65b2c5a5-e6ba-43be-9462-a98b03b675f1" width="600"> </video> <details><summary>📋 Show input prompt</summary> ```Have the cake man begin to take chunks out of himself and eat it.``` </details> <details><summary>📋 Show rewrite prompt</summary> ```The cake man sits on the chair, with his hands resting on his knees. Then, he slowly raises his right hand and breaks off a piece of cake from his left shoulder. Next, he brings the piece of cake to his mouth and begins to chew. At the same time, his eyes widen slightly, and his mouth parts gently. After that, he raises his right hand again, breaks off another piece of cake from his right arm, and repeats the action of bringing it to his mouth to chew.``` </details>|<video src="https://github.com/user-attachments/assets/de5f7480-b79c-4fc1-b345-c5880a3b5f9e" width="600"> </video> <details><summary>📋 Show input prompt</summary> ```A little girl, carrying a colorful handbag, skips through the garden.  The video uses claymation style.``` </details> <details><summary>📋 Show rewrite prompt</summary> ```A little girl with a colorful handbag skips through a whimsical claymation garden. In a vibrant garden constructed entirely from clay, a young girl, meticulously crafted in a claymation style, skips joyfully. She has chunky, sculpted yellow clay hair tied in pigtails that bounce with a slight stiffness, simple black button eyes, and a wide, permanently etched smile. She wears a simple pink clay dress with a white collar. In her left hand, she carries a small handbag molded from bright red and blue clay, which swings in a slightly jerky arc as she moves. Initially, the girl lifts her right leg high, her body momentarily suspended in a classic stop-motion pose. Then, she hops forward, landing lightly as her left leg swings through for the next skip. Her arms move in an exaggerated, back-and-forth rhythm, characteristic of stop-motion animation. Her movements are intentionally not perfectly fluid, highlighting the frame-by-frame nature of the claymation technique. The garden around her is a whimsical, textured world. In the foreground and mid-ground, oversized flowers with swirled purple and orange petals stand on thick green stems. The ground is a textured mat of green clay, showing subtle fingerprints and tool marks that add to the handmade charm. In the background, a pale blue clay backdrop features a simplified, smiling sun molded from yellow clay. The shot is at an eye-level angle with the main subject. The camera follows the subject, moving smoothly to the right to keep her in the frame. The lighting is bright and even, casting soft shadows that emphasize the rounded, three-dimensional forms of the clay models. The overall video presents a charming and detailed claymation style.``` </details>|
|高图视一致性|<img src="https://github.com/user-attachments/assets/3bc8e55d-c211-454e-8067-128c0e215eb6"> <video src="https://github.com/user-attachments/assets/3e6b7ee9-ec66-4e46-a446-801b1c1a1c81" width="600"> </video> <details><summary>📋 Show input prompt</summary> ```女孩放下书，站起身，转身向屋内走去。镜头拉远。``` </details> <details><summary>📋 Show rewrite prompt</summary> ```女孩合上手中的书，将书放在身侧的窗台上。随后，她缓缓站起身，转身向屋内走去，身影逐渐没入门后的阴影中。镜头缓缓拉远，露出更多被绿植覆盖的屋檐和墙体。``` </details>|<img src="https://github.com/user-attachments/assets/7657ce60-90b5-4fdc-b713-0eaa55829b09"> <video src="https://github.com/user-attachments/assets/9ca24021-2353-40d5-8a4d-0f8e67d51826" width="600"> </video> <details><summary>📋 Show input prompt</summary> ```女人手上的鸟亲了女人一口``` </details> <details><summary>📋 Show rewrite prompt</summary> ```女人手臂上的白色鹦鹉缓缓转过头，将喙轻轻触碰女人的脸颊，随后收回头部。女人嘴角微微上扬，目光温柔地注视着鹦鹉。背景中的绿植保持静止。``` </details>|


## 📚 引用
```bibtex
@misc{hunyuanvideo2025,
      title={HunyuanVideo 1.5 Technical Report}, 
      author={Tencent Hunyuan Foundation Model Team},
      year={2025},
      eprint={2511.18870},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.18870}, 
}
```

## 🙏 致谢
我们要感谢 [Transformers](https://github.com/huggingface/transformers), [Diffusers](https://github.com/huggingface/diffusers) , [HuggingFace](https://huggingface.co/) 以及 [Qwen-VL](https://github.com/QwenLM/Qwen-VL)的贡献者，感谢他们的公开研究和探索。

## 🌟 GitHub Star 历史

<a href="https://star-history.com/#Tencent-Hunyuan/HunyuanVideo-1.5&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Tencent-Hunyuan/HunyuanVideo-1.5&type=Date1&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=Tencent-Hunyuan/HunyuanVideo-1.5&type=Date1" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=Tencent-Hunyuan/HunyuanVideo-1.5&type=Date1" />
 </picture>
</a>
