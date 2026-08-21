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
# third party's use or distribution of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.

import unittest

import torch

from hyvideo.pipelines.pipeline_utils import randn_tensor


class RandnTensorTest(unittest.TestCase):

    def test_reproducible_independent_of_global_rng(self):
        def sample_stages():
            generator = torch.Generator("cpu").manual_seed(42)
            return tuple(randn_tensor((2, 3), generator=generator) for _ in range(3))

        first = sample_stages()

        torch.manual_seed(1234)
        torch.randn(100)

        second = sample_stages()
        for first_stage, second_stage in zip(first, second):
            torch.testing.assert_close(first_stage, second_stage, rtol=0, atol=0)

    def test_supports_per_sample_generators(self):
        generators = [
            torch.Generator("cpu").manual_seed(1),
            torch.Generator("cpu").manual_seed(2),
        ]
        actual = randn_tensor((2, 4), generator=generators)

        expected = torch.cat([
            torch.randn((1, 4), generator=torch.Generator("cpu").manual_seed(1)),
            torch.randn((1, 4), generator=torch.Generator("cpu").manual_seed(2)),
        ])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_rejects_generator_batch_mismatch(self):
        generators = [torch.Generator("cpu").manual_seed(1)]

        with self.assertRaisesRegex(ValueError, "effective batch size of 2"):
            randn_tensor((2, 4), generator=generators)
