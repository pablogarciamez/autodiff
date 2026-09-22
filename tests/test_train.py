from autodiff import Tensor
import numpy as np

def test_two_layer_xor_converges():
    np.random.seed(0)

    lr = 0.05

    D_in = 2
    D_out = 1
    H = 4
    W1 = Tensor(np.random.randn(D_in, H) * 0.1)
    b1 = Tensor(np.zeros(H))
    W2 = Tensor(np.random.randn(H, D_out) * 0.1)
    b2 = Tensor(np.zeros(D_out))
    X = Tensor([[0,0],[0,1],[1,0],[1,1]])
    target = Tensor([[0],[1],[1],[0]])

    for i in range(400):
        y = (X @ W1 + b1).relu() @ W2 + b2
        loss = ((y - target) ** 2).sum()
        loss.backward()
        W1.data -= lr * W1.grad
        b1.data -= lr * b1.grad
        W2.data -= lr * W2.grad
        b2.data -= lr * b2.grad
        W1.grad = np.zeros_like(W1.grad)
        b1.grad = np.zeros_like(b1.grad)
        W2.grad = np.zeros_like(W2.grad)
        b2.grad = np.zeros_like(b2.grad)
        
    assert loss.data < 1e-3