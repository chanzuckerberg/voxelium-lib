#!/usr/bin/env python3

"""
Test module for the sparse linear layer
"""

import unittest

import torch

from voxelium.base.explicit_grid_utils import size_to_maxr
from voxelium.base.explicit_grid_utils import make_explicit_grid2d
from voxelium.torch_extensions.sparse3d import *
from voxelium.relion import eulerToMatrix


ATOL = 1e-6
PERTURB_EPS = 1e-6

class TestSparseLinear(unittest.TestCase):
    def test_gradcheck_projection_cpu(self):
        self.assertTrue(self._gradcheck_projection("cpu"))

    def test_gradcheck_projection_cuda(self):
        self.assertTrue(self._gradcheck_projection("cuda:0"))

    @staticmethod
    def _gradcheck_projection(device):
        input_size = 5
        img_size = 8
        bsize = 2

        max_r = size_to_maxr(img_size)
        coord, mask = make_explicit_grid2d(size=img_size)
        coord = coord.to(device).double()

        p = ReconstructionLayer(img_size, input_size, dtype=torch.double, do_bias=True).to(device)
        input = torch.randn(bsize, p.input_size, dtype=torch.float64, requires_grad=True).to(p.weight.device)
        angles = torch.randn(bsize, 3, dtype=torch.float64, requires_grad=True).to(p.weight.device)
        rot_matrices = eulerToMatrix(angles)

        return torch.autograd.gradcheck(
            TrilinearProjection.apply,
            (
                input,  # input
                p.weight.double(),  # weight
                p.bias.double(),  # bias
                p.grid3d_index,  # grid3d_index
                rot_matrices.detach(),  # rot_matrices
                coord,  # grid2d_coord
                max_r,  # max_r
                0,  # backprop_eps
                True  # testing
            ),
            eps=PERTURB_EPS,
            atol=ATOL
        )

    def test_forward_projection_cpu(self):
        self.assertTrue(self._forward_projection("cpu"))

    def test_forward_projection_cuda(self):
        self.assertTrue(self._forward_projection("cuda"))

    @staticmethod
    def _forward_projection(device):
        input_size = 6
        img_size = 16

        max_r = size_to_maxr(img_size)
        coord, mask = make_explicit_grid2d(size=img_size)
        coord = coord.to(device)

        p = ReconstructionLayer(img_size, input_size)
        p.to(device)

        ref = TestSparseLinear._make_random_ref(img_size, input_size)
        p.weight.data *= 0
        p.bias.data *= 0
        for i in range(input_size):
            p.set_base(ref[i].to(device), i)

        input = torch.eye(input_size).to(device)
        rot_matrices = torch.eye(3).unsqueeze(0).expand(input_size, 3, 3).contiguous().to(device)
        projection_ = p(input, max_r=max_r, grid2d_coord=coord, rot_matrices=rot_matrices)
        projection = torch.zeros(input_size, (img_size + 1) * (img_size//2 + 1), 2).to(device)
        projection[:, mask, :] = projection_
        projection = projection.view(input_size, img_size + 1, img_size//2 + 1, 2)
        projection = projection.cpu().detach().numpy()
        projection = projection[:, :-1]
        ref_projection = ref[:, img_size//2, :, :]
        ref_projection = torch.view_as_real(ref_projection)
        ref_projection = ref_projection.cpu().detach().numpy()

        return np.all(np.abs(projection - ref_projection) < ATOL)

    def test_forward_volume_extraction_cpu(self):
        self.assertTrue(self._forward_volume_extraction("cpu"))

    def test_forward_volume_extraction_cuda(self):
        self.assertTrue(self._forward_volume_extraction("cuda"))

    @staticmethod
    def _forward_volume_extraction(device):
        input_size = 5
        img_size = 16

        p = ReconstructionLayer(img_size, input_size)
        p.to(device)

        ref = TestSparseLinear._make_random_ref(img_size, input_size)
        ref = ref.to(device)

        p.weight.data *= 0
        p.bias.data *= 0
        p.set_base(ref[0], 0)

        input = torch.zeros([1, input_size]).to(device)
        input[0, 0] = 1
        vol = p(input)[0, :-1, :-1].cpu().detach().numpy()

        ref = torch.view_as_real(ref)
        ref = ref[0, ..., :].cpu().detach().numpy()

        return np.all(np.abs(vol - ref) < ATOL)

    def test_gradcheck_volume_extraction_cpu(self):
        self.assertTrue(self._gradcheck_volume_extraction("cpu"))

    def test_gradcheck_volume_extraction_cuda(self):
        self.assertTrue(self._gradcheck_volume_extraction("cuda:0"))

    @staticmethod
    def _gradcheck_volume_extraction(device):
        input_size = 3
        img_size = 8
        bsize = 2
        p = ReconstructionLayer(img_size, input_size, dtype=torch.double).to(device)
        input = torch.randn(bsize, p.input_size, dtype=torch.float64, requires_grad=True).to(p.weight.device)

        return torch.autograd.gradcheck(
            VolumeExtraction.apply,
            (
                input,  # input
                p.weight.double(),  # weight
                p.bias.double(),  # bias
                p.grid3d_index,  # grid3d_index
                size_to_maxr(img_size)  # maxr
            ),
            eps=PERTURB_EPS,
            atol=ATOL
        )

    @staticmethod
    def _make_random_ref(img_size, input_size):
        max_r = size_to_maxr(img_size)
        ls = torch.linspace(-img_size // 2, img_size // 2, img_size + 1)
        lsx = torch.linspace(0, img_size // 2, img_size // 2 + 1)
        z, y, x = np.meshgrid(ls, ls, lsx, indexing='ij')
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        mask = r < max_r
        mask = mask[:-1, :-1]
        n = np.sum(mask)
        mask = torch.Tensor(mask).bool()
        ref = torch.zeros([input_size, img_size, img_size, img_size // 2 + 1, 2])
        for i in range(input_size):
            ref[i, mask] = torch.empty(n, 2).normal_()
        ref = torch.view_as_complex(ref)
        return ref


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    test = TestSparseLinear()

    test.test_gradcheck_projection_cpu()
    test.test_gradcheck_projection_cuda()
    test.test_forward_projection_cpu()
    test.test_forward_projection_cuda()

    test.test_forward_volume_extraction_cpu()
    test.test_forward_volume_extraction_cuda()
    test.test_gradcheck_volume_extraction_cpu()
    test.test_gradcheck_volume_extraction_cuda()

    print("All good!")

    # device = "cuda:0"
    # device = "cpu"
    #
    # def _make_random_ref(img_size, input_size):
    #     ls = torch.linspace(-img_size // 2, img_size // 2, img_size + 1)
    #     lsx = torch.linspace(0, img_size // 2, img_size // 2 + 1)
    #     z, y, x = np.meshgrid(ls, ls, lsx, indexing='ij')
    #     r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    #     mask = (r <= img_size // 2) & (z == 0)
    #     mask = torch.Tensor(mask).bool()
    #     ref = torch.zeros([img_size + 1, img_size + 1, img_size // 2 + 1, 2])
    #     ref[mask] = 0.5
    #     return ref.unsqueeze(-2).expand(img_size + 1, img_size + 1, img_size // 2 + 1, input_size, 2)
    #
    #
    # input_size = 5
    # img_size = 16
    #
    # p = SparseVolumeReconstructionLinear(img_size, input_size)
    # p.to(device)
    # max_r = img_size // 2
    # coord, mask = make_explicit_grid2d(size=img_size)
    # coord = coord.to(device)
    #
    # ref = _make_random_ref(img_size, input_size)
    # p.weight.data *= 0
    # p.bias.data *= 0
    # p.set_reference(ref.to(device))
    #
    # input = torch.ones([1, input_size]).to(device) * 1. / input_size
    # angles = np.zeros([1, 3])
    # angles[0, 1] = 1
    # angles = torch.Tensor(angles).to(device)
    # angles.requires_grad = True
    # angles.retain_grad()
    # angles.register_hook(lambda grad: print(grad))
    # optimizer = torch.optim.SGD([angles], lr=0.01, momentum=0.9)
    # rot_matrices = eulerToMatrix(angles)
    # # rot_matrices.retain_grad()
    # projection_ = p(input, max_r=max_r, grid2d_coord=coord, rot_matrices=rot_matrices)
    # projection = torch.zeros((img_size + 1) * (img_size // 2 + 1), 2).to(device)
    # projection[mask, :] = projection_[0]
    # projection = projection.view(img_size + 1, img_size // 2 + 1, 2)
    # ref_projection = ref[img_size // 2, :, :, 0, :]
    #
    # loss = torch.mean(torch.square(projection.cpu() - ref_projection))
    # loss.backward()
    #
    # optimizer.step()
    # print(angles.grad)
