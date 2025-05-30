from voxelium.torch_extensions.sparse3d import test as sparse_test
from voxelium.torch_extensions.inplace_topk import test as topk_test

if __name__ == "__main__":
    sparse_test.main()
    topk_test.main()
    print("All tests passed successfully.")