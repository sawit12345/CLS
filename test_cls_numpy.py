import unittest

import numpy as np

from cls_numpy import (
    ComplementaryLearningSystem,
    Hippocampus,
    IndexedSparseNeocortex,
    corrupt_sdr,
)


class CLSTest(unittest.TestCase):
    def test_sdr_address_space_is_combinatorial(self):
        cortex = IndexedSparseNeocortex(sdr_size=4096, active_bits=64, bucket_count=8192)
        self.assertGreater(cortex.address_space_log10(), 120.0)
        self.assertEqual(cortex.capacity_for_bucket_load(2.0), 16384)

    def test_hippocampus_one_shot_partial_cue(self):
        rng = np.random.default_rng(1)
        hippocampus = Hippocampus(sdr_size=512, active_bits=32, recall_threshold=0.65)
        sdr = np.sort(rng.choice(512, size=32, replace=False))
        hippocampus.store(sdr, "episode")

        cue = corrupt_sdr(sdr, 512, drop_fraction=0.2, seed=2)
        result = hippocampus.recall(cue)
        self.assertEqual(result.payload, "episode")
        self.assertGreaterEqual(result.overlap, 0.75)

    def test_neocortex_retrieves_many_noisy_sdrs(self):
        rng = np.random.default_rng(3)
        cortex = IndexedSparseNeocortex(
            sdr_size=2048,
            active_bits=40,
            tables=32,
            bucket_count=32768,
            recall_threshold=0.28,
            seed=4,
        )
        sdrs = [np.sort(rng.choice(2048, size=40, replace=False)) for _ in range(1500)]
        for i, sdr in enumerate(sdrs):
            cortex.store(sdr, i)

        correct = 0
        for i in rng.choice(len(sdrs), size=150, replace=False):
            cue = corrupt_sdr(sdrs[i], 2048, drop_fraction=0.15, seed=int(i))
            result = cortex.query(cue)
            correct += int(result.payload == int(i))

        self.assertGreaterEqual(correct / 150, 0.98)
        self.assertLess(cortex.expected_bucket_load(), 0.05)

    def test_full_cls_observe_consolidate_infer(self):
        rng = np.random.default_rng(5)
        cls = ComplementaryLearningSystem(input_dim=16, sdr_size=1024, active_bits=32, seed=5)
        x = rng.normal(size=16)
        cls.observe(x, "apple")
        self.assertEqual(cls.infer(x).payload, "apple")
        self.assertEqual(cls.consolidate(), 1)
        self.assertEqual(cls.infer(x, prefer_hippocampus=False).payload, "apple")


if __name__ == "__main__":
    unittest.main()
