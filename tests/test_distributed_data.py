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

import unittest

from hyvideo.utils.distributed_data import (
    create_data_parallel_sampler,
    set_dataloader_epoch,
)


class DistributedDataTest(unittest.TestCase):
    def test_sampler_shards_across_data_parallel_replicas(self):
        dataset = list(range(12))
        rank_0 = create_data_parallel_sampler(dataset, dp_rank=0, dp_size=2, seed=42)
        rank_1 = create_data_parallel_sampler(dataset, dp_rank=1, dp_size=2, seed=42)

        rank_0_indices = list(rank_0)
        rank_1_indices = list(rank_1)

        self.assertTrue(set(rank_0_indices).isdisjoint(rank_1_indices))
        self.assertEqual(set(rank_0_indices + rank_1_indices), set(range(12)))

    def test_sequence_parallel_ranks_share_a_data_parallel_shard(self):
        dataset = list(range(12))
        first_sp_rank = create_data_parallel_sampler(
            dataset, dp_rank=1, dp_size=2, seed=42
        )
        second_sp_rank = create_data_parallel_sampler(
            dataset, dp_rank=1, dp_size=2, seed=42
        )

        self.assertEqual(list(first_sp_rank), list(second_sp_rank))

    def test_single_replica_keeps_the_regular_shuffle_path(self):
        self.assertIsNone(
            create_data_parallel_sampler([0, 1], dp_rank=0, dp_size=1, seed=42)
        )

    def test_dataloader_epoch_is_forwarded_to_sampler(self):
        class Sampler:
            def __init__(self):
                self.epoch = None

            def set_epoch(self, epoch):
                self.epoch = epoch

        class Loader:
            sampler = Sampler()

        loader = Loader()
        set_dataloader_epoch(loader, 7)

        self.assertEqual(loader.sampler.epoch, 7)

    def test_epoch_changes_distributed_shuffle_order(self):
        dataset = list(range(12))
        rank_0 = create_data_parallel_sampler(dataset, dp_rank=0, dp_size=2, seed=42)
        rank_1 = create_data_parallel_sampler(dataset, dp_rank=1, dp_size=2, seed=42)
        epoch_0_indices = [list(rank_0), list(rank_1)]

        rank_0.set_epoch(1)
        rank_1.set_epoch(1)
        epoch_1_indices = [list(rank_0), list(rank_1)]

        self.assertNotEqual(epoch_0_indices, epoch_1_indices)
        for shards in (epoch_0_indices, epoch_1_indices):
            self.assertTrue(set(shards[0]).isdisjoint(shards[1]))
            self.assertEqual(set(shards[0] + shards[1]), set(dataset))

    def test_invalid_data_parallel_coordinates_fail_early(self):
        with self.assertRaisesRegex(ValueError, "dp_size must be positive"):
            create_data_parallel_sampler([0], dp_rank=0, dp_size=0, seed=42)
        with self.assertRaisesRegex(ValueError, "dp_rank must be"):
            create_data_parallel_sampler([0], dp_rank=2, dp_size=2, seed=42)


if __name__ == "__main__":
    unittest.main()
