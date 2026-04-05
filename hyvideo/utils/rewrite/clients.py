# -*- coding: utf-8 -*-
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

import json
import time
import ast
import os
import io
import base64

import openai
from loguru import logger


class NonStreamResponse(object):
    def __init__(self):
        self.response = ""

    def _deserialize(self, obj):
        self.response = json.dumps(obj)

# DeepSeekClient
class DeepSeekClient(object):
    def __init__(self, key_id, key_secret):
        from tencentcloud.common.common_client import CommonClient
        from tencentcloud.common import credential
        from tencentcloud.common.profile.client_profile import ClientProfile
        from tencentcloud.common.profile.http_profile import HttpProfile
        cred = credential.Credential(key_id, key_secret)
        httpProfile = HttpProfile()
        httpProfile.endpoint = "lkeap.tencentcloudapi.com"
        httpProfile.reqTimeout = 40000  # The streaming interface may take a longer time.
        clientProfile = ClientProfile()
        clientProfile.httpProfile = httpProfile
        self.common_client = CommonClient("lkeap", "2024-05-22", cred, "ap-guangzhou", profile=clientProfile)

    def run_single_recaption(self, system_prompt, input_prompt):
        post_dict = {
            "Model": "deepseek-v3.1",
            "Messages": [
                {
                    "Role": "system",
                    "Content": system_prompt
                },
                {
                    "Role": "user",
                    "Content": input_prompt
                }
            ],
            "Stream": False,
            "Thinking": {"Type": "enabled"},
        }
        while True:
            try:
                resp = self.common_client._call_and_deserialize("ChatCompletions", post_dict, NonStreamResponse)
                break
            except Exception as e:
                logger.error(e)
                time.sleep(1)
        resp = self.common_client._call_and_deserialize("ChatCompletions", post_dict, NonStreamResponse)
        response = resp.response
        response = ast.literal_eval(response)
        content = response["Choices"][0]["Message"]["Content"]
        reasoning_content = response["Choices"][0]["Message"]["ReasoningContent"]
        print('Initial prompt: ', input_prompt)
        print('Recaption prompt: ', content)

        return content, reasoning_content

class MiniMaxClient(object):
    """
    Client for MiniMax Cloud API (OpenAI-compatible).
    Recommended models: MiniMax-M2.7, MiniMax-M2.5, MiniMax-M2.5-highspeed (204K context).
    """

    def __init__(self, api_key=None, model_name=None):
        self.api_key = api_key or os.getenv("MINIMAX_API_KEY", "")
        self.model_name = model_name or "MiniMax-M2.7"
        self.base_url = "https://api.minimax.io/v1"

    def minimax_api_call(self, system_prompt: str, user_input: str, temperature: float, max_tokens: int):
        """
        Use MiniMax Chat API to perform text rewriting, parse <think>...</think> sections
        for reasoning content, and return (thinking, result).
        """
        # MiniMax temperature range: [0.0, 1.0]
        temperature = max(0.0, min(temperature, 1.0))

        client = openai.OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=600)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]

        last_err = None

        for i in range(10):
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False,
                )
                content = response.choices[0].message.content
                thinking = ""
                result = content or ""
                if content and "</think>" in content:
                    head, tail = content.split("</think>", 1)
                    thinking = head.replace("<think>", "").strip()
                    result = tail.strip()
                return thinking, result
            except Exception as e:
                last_err = e
                if i < 9:
                    time.sleep(2 ** i)
                    continue
                raise last_err

    def run_single_recaption(self, system_prompt, input_prompt, temperature=0.1, max_tokens=4096):
        thinking, result = self.minimax_api_call(system_prompt, input_prompt, temperature, max_tokens)
        return result


class QwenClient(object):
    def __init__(self, base_url=None, model_name=None):
        # Recommended model: Qwen3-235B-A22B-Thinking-2507
        self.base_url = base_url
        self.model_name = model_name

    def qwen_api_call(self, system_prompt: str, user_input: str, temperature: float, max_tokens: int):
        """
        Use Qwen Chat API to perform text rewriting, parse <think>...</think> sections for reasoning content, and return (thinking, result).
        """
        client = openai.OpenAI(base_url=self.base_url, api_key="None", timeout=600)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]

        last_err = None

        for i in range(10):
            try:
                stream = False
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=stream,
                )
                content = response.choices[0].message.content
                thinking = ""
                result = content or ""
                if content and "</think>" in content:
                    head, tail = content.split("</think>", 1)
                    thinking = head.replace("<think>", "").strip()
                    result = tail.strip()
                return thinking, result
            except Exception as e:
                last_err = e
                if i < 9:
                    time.sleep(2 ** i)
                    continue
                raise last_err


    def run_single_recaption(self, system_prompt, input_prompt, temperature=0.1, max_tokens=4096):
        thinking, result = self.qwen_api_call(system_prompt, input_prompt, temperature, max_tokens)
        return result


class QwenVLClient(object):

    def __init__(self, base_url=None, model_name=None):
        self.base_url = base_url
        self.model_name = model_name
        self.api_key = "None"
        self.max_image_size = int(os.getenv("I2V_REWRITE_MAX_IMAGE_SIZE", "1024")) # 控制送入模型的最大图像分辨率

    def _encode_image_to_base64(self, image_path: str, max_dimension: int) -> str:
        """
        参考 hyvideo/utils/rewrite/qwen_vllm.py 的实现：
        加载本地图片，将其按比例缩放到 max_dimension，然后编码为 Base64 data URL。
        """
        try:
            from PIL import Image
        except ImportError as e:
            logger.error("Pillow (PIL) is required for QwenVLClient image encoding but is not installed.")
            raise e

        try:
            image = image_path
            if not isinstance(image, Image.Image):
                image = Image.open(image)

            with image as img:
                if img.width > max_dimension or img.height > max_dimension:
                    img.thumbnail((max_dimension, max_dimension))

                buffer = io.BytesIO()

                # 去除透明通道，统一转为 JPEG
                if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                    img = img.convert("RGB")

                img.save(buffer, format="JPEG")
                mime_type = "image/jpeg"

                encoded_string = base64.b64encode(buffer.getvalue()).decode("utf-8")
                return f"data:{mime_type};base64,{encoded_string}"
        except Exception as e:
            logger.error(f"Failed to encode image {image_path} to base64: {e}")
            raise

    def qwen_api_call(
        self,
        system_prompt: str,
        user_input: str,
        temperature: float,
        max_tokens: int,
        img_path: str = None,
    ):
        """
        Use Qwen3-VL to perform text rewriting.
        """
        client = openai.OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=3600,
        )

        assert "{}" in system_prompt, "system_prompt must contain {{}}"
        prompt_text = system_prompt.format(user_input)

        
        assert img_path is not None, "img_path is required"
        base64_image = self._encode_image_to_base64(img_path, max_dimension=self.max_image_size)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": base64_image,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt_text,
                    },
                ],
            }
        ]

        last_err = None

        for i in range(10):
            try:

                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False,
                )
                content = response.choices[0].message.content
                thinking = ""
                result = content or ""
                # 和 QwenClient 一样，兼容 <think> 思维模式输出
                if content and "</think>" in content:
                    head, tail = content.split("</think>", 1)
                    thinking = head.replace("<think>", "").strip()
                    result = tail.strip()
                return thinking, result
            except Exception as e:
                last_err = e
                logger.error(f"QwenVLClient request failed (attempt {i + 1}/10): {e}")
                if i < 9:
                    time.sleep(2)
                else:
                    raise last_err

    def run_single_recaption(
        self,
        system_prompt,
        input_prompt,
        temperature=0.1,
        max_tokens=4096,
        img_path: str = None,
    ):
        thinking, result = self.qwen_api_call(
            system_prompt,
            input_prompt,
            temperature,
            max_tokens,
            img_path=img_path,
        )
        return result