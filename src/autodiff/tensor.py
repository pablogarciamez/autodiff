import numpy as np
from .value import buildTopo

class Tensor:
    def __init__(self, data, prev = (), op = ""):
        self.data = np.array(data, dtype=float)
        self.grad = np.zeros_like(self.data)
        self.prev = prev
        self.op = op

    def __matmul__(self, other):
        return Tensor(self.data @ other.data, (self, other), "@")

    def backwardStep(self):
        if self.op == "@":
            A, B = self.prev
            G = self.grad
            A.grad += G @ B.data.T
            B.grad += A.data.T @ G

    def backward(self):
        topo = []
        buildTopo(self, topo, [])
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node.backwardStep()