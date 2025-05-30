import unittest
import voxelium.torch_extensions.sparse3d.test as sparse_test
import voxelium.torch_extensions.inplace_topk.test as topk_test

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Load tests from both modules
    suite.addTests(loader.loadTestsFromModule(sparse_test))
    suite.addTests(loader.loadTestsFromModule(topk_test))

    runner = unittest.TextTestRunner()
    result = runner.run(suite)

    if result.wasSuccessful():
        print("All tests passed successfully.")
    else:
        print("Some tests failed.")
        exit(1)