import numpy as np
from autodiff import Tensor

def test_tensor_product_square_matrices():
    A = Tensor(np.array([[1, 2], [3, 4]]))
    B = Tensor(np.array([[5, 6], [7, 8]]))
    C = A @ B
    C.backward()
    np.testing.assert_allclose(A.grad, [[11, 15], [11, 15]])
    np.testing.assert_allclose(B.grad, [[4, 4], [6, 6]])

def test_differnt_shape_tensors():
    A = Tensor([[1, 2, 3],
        [4, 5, 6]])
    B = Tensor([[1, 0, 2, 1],
        [3, 1, 0, 2],
        [0, 2, 5, 1]])
    C = A @ B
    C.backward()
    np.testing.assert_allclose(A.grad, [[4, 6, 8], [4, 6, 8]])
    np.testing.assert_allclose(B.grad, [[5, 5, 5, 5], [7, 7, 7, 7], [9, 9, 9, 9]])