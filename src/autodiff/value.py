
class Value():
    @staticmethod
    def toValue(val):
        if isinstance(val, (int, float)):
            return Value(val)
        if isinstance(val, Value):
            return val
        raise TypeError(f"Cannot convert {type(val)} to Value")

    def __init__(self, data, prev = (), op = ""):
        self.data = data
        self.grad = 0.0
        self.prev = prev
        self.op = op

    def __add__(self, other):
        other = Value.toValue(other)
        return Value(self.data + other.data, (self, other), "+")

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * (-1)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return -(self - other)

    def __mul__(self, other):
        other = Value.toValue(other)
        return Value(self.data * other.data, (self, other), "*")

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = Value.toValue(other)
        return self * other ** (-1)

    def __rtruediv__(self, other):
        other = Value.toValue(other)
        return other / self

    def __pow__(self, n):
        out = Value(self.data ** n, (self,), "**")
        out.n = n
        return out

    def relu(self):
        return Value(max(0.0, self.data), (self,), "relu")

    def backwardStep(self):
        if self.op == "+":
            for child in self.prev:
                child.grad += self.grad
        elif self.op == "*":
            self.prev[0].grad += self.prev[1].data * self.grad
            self.prev[1].grad += self.prev[0].data * self.grad
        elif self.op == "**":
            self.prev[0].grad += self.n * self.prev[0].data ** (self.n - 1) * self.grad
        elif self.op == "relu":
            self.prev[0].grad += self.grad if self.prev[0].data > 0 else 0

    def backward(self):
        topo = []
        buildTopo(self, topo, [])
        self.grad = 1
        for node in reversed(topo):
            node.backwardStep()

def buildTopo(node, topo, visited):
    if node not in visited:
        visited.append(node)
        for child in node.prev:
            buildTopo(child, topo, visited)
        topo.append(node)

x = Value(2.0)
y = Value(-3.0)
f = (x * y).relu()
f.backward()
print(x.grad, y.grad)