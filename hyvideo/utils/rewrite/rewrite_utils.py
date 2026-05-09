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

import os
from hyvideo.utils.rewrite.clients import QwenClient, QwenVLClient, MiniMaxClient
from hyvideo.utils.rewrite.t2v_prompt import t2v_rewrite_system_prompt
from hyvideo.utils.rewrite.i2v_prompt import i2v_rewrite_system_prompt


def _create_t2v_client():
    """Create the appropriate T2V rewrite client based on REWRITE_PROVIDER env var."""
    provider = os.getenv("REWRITE_PROVIDER", "").lower()
    if provider == "minimax":
        return MiniMaxClient(
            api_key=os.getenv("MINIMAX_API_KEY"),
            model_name=os.getenv("T2V_REWRITE_MODEL_NAME", "MiniMax-M2.7"),
        )
    # Default: QwenClient (vLLM-compatible)
    base_url = os.getenv("T2V_REWRITE_BASE_URL")
    model_name = os.getenv("T2V_REWRITE_MODEL_NAME")
    if not base_url or not model_name:
        raise EnvironmentError(
            "T2V_REWRITE_BASE_URL and T2V_REWRITE_MODEL_NAME must be set in the environment variables "
            "when prompt rewriting is enabled. Please configure the rewrite service correctly. "
            "Alternatively, set REWRITE_PROVIDER=minimax and MINIMAX_API_KEY to use MiniMax Cloud API."
        )
    return QwenClient(base_url, model_name)


def t2v_rewrite(user_prompt, rewrite_client=None):
    if rewrite_client is None:
        rewrite_client = _create_t2v_client()
    try:
        rewritten_prompt = rewrite_client.run_single_recaption(t2v_rewrite_system_prompt, user_prompt)
    except Exception as e:
        raise ValueError(
            f"Failed to rewrite prompt using {type(rewrite_client).__name__}: {e}"
        )
    return rewritten_prompt

def i2v_rewrite(user_input, img_path, rewrite_client=None):
    """
    Use a rewrite client to generate a rewritten prompt for image-to-video.
    """
    i2v_base_url = os.getenv("I2V_REWRITE_BASE_URL")
    i2v_model_name = os.getenv("I2V_REWRITE_MODEL_NAME")
    if not i2v_base_url or not i2v_model_name:
        raise EnvironmentError(
            "I2V_REWRITE_BASE_URL and I2V_REWRITE_MODEL_NAME must be set in the environment variables "
            "for image-to-video prompt rewriting. Please set them before running, e.g.:\n"
            'export I2V_REWRITE_BASE_URL="YOUR_I2V_REWRITE_BASE_URL"\n'
            'export I2V_REWRITE_MODEL_NAME="YOUR_I2V_REWRITE_MODEL_NAME"\n'
        )
    if rewrite_client is None:
        rewrite_client = QwenVLClient(i2v_base_url, i2v_model_name)
    try:
        rewritten_prompt = rewrite_client.run_single_recaption(i2v_rewrite_system_prompt, user_input, img_path=img_path)
    except Exception as e:
        raise ValueError(
            f"Failed to rewrite prompt using {type(rewrite_client).__name__}: {e}"
        )
    return rewritten_prompt

def run_prompt_rewrite(user_prompt, img_path, task_type):
    if task_type == "i2v":
        return i2v_rewrite(user_prompt, img_path)
    elif task_type == "t2v":
        return t2v_rewrite(user_prompt)
    else:
        raise ValueError(f"Unsupported task_type: {task_type}. Must be 'i2v' or 't2v'")
