import numpy as np
from .value import buildTopo

class Tensor:
    def __init__(self, data, prev = (), op = "", axis = None):
        self.data = np.array(data, dtype=float)
        self.grad = np.zeros_like(self.data)
        self.prev = prev
        self.op = op
        self.axis = axis

    def __matmul__(self, other):
        return Tensor(self.data @ other.data, (self, other), "@")

    def sum(self, axis = None):
        return Tensor(self.data.sum(axis), (self,), "sum", axis)

    def relu(self):
        return Tensor(np.maximum(self.data, 0), (self,), "relu")

    def __add__(self, other):
        return Tensor(self.data + other.data, (self, other), "+")

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            out = Tensor(self.data * other, (self,), "*")
            out.mul = other
            return out
        else:
            return Tensor(self.data * other.data, (self, other), "*")

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * (-1)

    def __sub__(self, other):
        return self + (-other)

    def __pow__(self, exp):
        out = Tensor(self.data ** exp, (self,), "**")
        out.exp = exp
        return out

    def backwardStep(self):
        if self.op == "@":
            A, B = self.prev
            G = self.grad
            A.grad += G @ B.data.T
            B.grad += A.data.T @ G
        if self.op == "sum":
            A, = self.prev
            axis = self.axis
            A.grad += self.grad if axis is None else np.expand_dims(self.grad, axis)
        if self.op == "relu":
            A, = self.prev
            A.grad += (A.data > 0) * self.grad
        if self.op == "+":
            A, B = self.prev
            A.grad += self.grad
            B.grad += self.grad.sum(axis = tuple(range(len(self.grad.shape) - len(B.grad.shape))))
        if self.op == "*":
            if len(self.prev) == 2:
                A, B = self.prev
                A.grad += B.data * self.grad
                B.grad += A.data * self.grad
            else:
                A, = self.prev
                A.grad += self.mul * self.grad
        if self.op == "**":
            A, = self.prev
            A.grad += self.exp * A.data ** (self.exp - 1) * self.grad

    def backward(self):
        topo = []
        buildTopo(self, topo, [])
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node.backwardStep()