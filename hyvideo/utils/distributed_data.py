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

from typing import Optional

from torch.utils.data import DataLoader, Dataset, DistributedSampler


def create_data_parallel_sampler(
    dataset: Dataset,
    *,
    dp_rank: int,
    dp_size: int,
    seed: int,
) -> Optional[DistributedSampler]:
    """Shard a map-style dataset across data-parallel, rather than sequence-parallel, ranks."""
    if dp_size < 1:
        raise ValueError(f"dp_size must be positive, got {dp_size}")
    if not 0 <= dp_rank < dp_size:
        raise ValueError(f"dp_rank must be in [0, {dp_size}), got {dp_rank}")
    if dp_size == 1:
        return None

    return DistributedSampler(
        dataset,
        num_replicas=dp_size,
        rank=dp_rank,
        shuffle=True,
        seed=seed,
    )


def set_dataloader_epoch(dataloader: DataLoader, epoch: int) -> None:
    """Reseed distributed shuffling before each pass over a dataloader."""
    sampler = getattr(dataloader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)
