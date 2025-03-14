#!/usr/bin/env python3

"""
Test module for the sparse linear layer
"""

import unittest

import torch

from voxelium.torch_extensions.inplace_topk import InplaceTopK


ATOL = 1e-6
PERTURB_EPS = 1e-6

class TestTopK(unittest.TestCase):
    def _run(self, device):
        size = 100  # Number of batches (or "rows")
        k = 50     # Number of top elements to track
        num_candidates = 2000  # Number of candidates per batch

        # Initialize module
        topk_module = InplaceTopK(size=size, k=k, dtype=torch.float32).to(device)

        # Generate random candidate values and indices
        candidate_values = torch.randn(size, num_candidates, dtype=torch.float32, device=device) * 100  # Random values between 0-100
        candidate_indices = torch.randint(0, 100, (num_candidates,), dtype=torch.int64, device=device)  # Random indices

        # Run InplaceTopK
        topk_module(candidate_values, candidate_indices)

        # Use torch.topk to compute expected results
        expected_top_values, order = torch.topk(candidate_values, k=k, dim=1, largest=True)
        candidate_indices_ = candidate_indices[None].expand((size, num_candidates))
        expected_top_indices = candidate_indices_.gather(1, order)

        top_values, top_indices = topk_module.get_sorted()

        expected_mean = candidate_values.mean(-1)
        torch.testing.assert_close(topk_module.get_mean(), expected_mean, atol=1e-5, rtol=1e-5)

        expected_std = candidate_values.std(-1)
        torch.testing.assert_close(topk_module.get_std(), expected_std, atol=1, rtol=1e-2)

        # Compare results
        torch.testing.assert_close(top_values, expected_top_values, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(top_indices, expected_top_indices)

    def test_cpu(self):
        self._run("cpu")

    def test_cuda(self):
        self._run("cuda")


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    test = TestTopK()

    test.test_cpu()
    test.test_cuda()

    print("All good!")
