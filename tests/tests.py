from voxelium.torch_extensions.sparse3d import test as sparse_test
from voxelium.torch_extensions.inplace_topk import test as topk_test

def test_sparse3d():
    sparse_test.main()

def test_topk():
    topk_test.main()
