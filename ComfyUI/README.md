# ComfyUI Usage Guide for HunyuanVideo-1.5

This guide helps you use ComfyUI to run HunyuanVideo-1.5 model.

## 📋 Table of Contents

- [Installation](#installation)
- [Download Models](#download-models)
- [Workflow Templates](#workflow-templates)
- [Links](#links)

## 🚀 Installation

Install the latest version of ComfyUI from the official repository:

```bash
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
```

For detailed installation instructions, please refer to the [ComfyUI official repository](https://github.com/comfyanonymous/ComfyUI).


## 📥 Download Models

### HunyuanVideo-1.5 Checkpoint

Download the ComfyUI checkpoint from Hugging Face:

**Model**: https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged

## 🎨 Workflow Templates

Find workflow templates at:

- **720p Image-to-Video**: https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_hunyuan_video_1.5_720p_i2v.json
- **720p Text-to-Video**: https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_hunyuan_video_1.5_720p_t2v.json


* **Important**: Workflows are still being updated. Please ensure you are using the latest version of the workflow templates to guarantee the best generation results.
* **Important**: The distilled models are CFG-distilled models rather than step-distilled models. Please run with 50 inference steps for best generation quality. The recommended configurations are provided in the following table.


### Recommended Inference Configurations

**Important!** The following table is for the best generation quality. If you want to generate faster, you can use the following configurations:


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


### Prompt Writing Tips

- Use detailed, descriptive prompts for better results
- Refer to the [HunyuanVideo-1.5 Prompt Handbook](https://doc.weixin.qq.com/doc/w3_AXcAcwZSAGgCNACVygLxeQjyn4FYS?scode=AJEAIQdfAAoSfXnTj0AAkA-gaeACk) for guidance
- Include details about camera movement, style, lighting, etc.

## 🔗 Links

- **ComfyUI Repository**: https://github.com/comfyanonymous/ComfyUI
- **HunyuanVideo-1.5 Repository**: https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5
- **Model Checkpoint**: https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged
- **Workflow Templates**: https://github.com/Comfy-Org/workflow_templates/tree/main/templates
- **Prompt Handbook**: https://doc.weixin.qq.com/doc/w3_AXcAcwZSAGgCNACVygLxeQjyn4FYS?scode=AJEAIQdfAAoSfXnTj0AAkA-gaeACk

---

For issues or questions, please visit the [HunyuanVideo-1.5 repository](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5) to submit issues or check documentation.
