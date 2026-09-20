import numpy as np
from autodiff import Tensor
from helpers import numericGrad

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

def test_matmul_gradcheck():
    rng = np.random.default_rng(0)
    A_data = rng.normal(size=(5, 3))
    B_data = rng.normal(size=(3, 2))

    A, B = Tensor(A_data), Tensor(B_data)
    C = A @ B
    C.backward()

    def L_A(a):
        return (Tensor(a) @ Tensor(B_data)).data.sum()

    def L_B(b):
        return (Tensor(A_data) @ Tensor(b)).data.sum()

    np.testing.assert_allclose(A.grad, numericGrad(L_A, A_data), atol=1e-5)
    np.testing.assert_allclose(B.grad, numericGrad(L_B, B_data), atol=1e-5)