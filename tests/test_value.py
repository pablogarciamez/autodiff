import pytest
from autodiff import Value, buildTopo


# ---------- forwardStep ----------

def test_init():
    v = Value(2.0)
    assert v.data == 2.0
    assert v.grad == 0.0
    assert v.prev == ()
    assert v.op == ""


def test_mul_forward():
    assert (Value(2.0) * Value(3.0)).data == 6.0


def test_add_forward():
    assert (Value(2.0) + Value(3.0)).data == 5.0


def test_f_forward():
    x, y = Value(2.0), Value(3.0)
    f = x * y + x
    assert f.data == 8.0


def test_prev_de_f():
    x, y = Value(2.0), Value(3.0)
    z = x * y
    f = z + x
    assert f.prev[0] is z and f.prev[0].data == 6.0
    assert f.prev[1] is x and f.prev[1].data == 2.0


# ---------- topologicalOrder ----------

def test_build_topo():
    x, y = Value(2.0), Value(3.0)
    f = x * y + x
    topo = []
    buildTopo(f, topo, [])
    assert [n.data for n in topo] == [2.0, 3.0, 6.0, 8.0]
    assert topo.count(x) == 1


# ---------- grads: sum & product ----------

def test_grad_f():
    x, y = Value(2.0), Value(3.0)
    f = x * y + x
    f.backward()
    assert x.grad == 4.0   # y + 1
    assert y.grad == 2.0   # x


def test_grad_x_times_x():
    x = Value(2.0)
    f = x * x
    f.backward()
    assert x.grad == 4.0


# ---------- literals ----------

def test_literal_on_right():
    x = Value(2.0)
    f = x * 3
    f.backward()
    assert f.data == 6.0
    assert x.grad == 3.0


def test_literal_on_left():
    x = Value(2.0)
    f = 3 * x
    f.backward()
    assert f.data == 6.0
    assert x.grad == 3.0


def test_type_not_supported():
    with pytest.raises(TypeError):
        Value(2.0) * "a"


# ---------- subs & negs ----------

def test_neg():
    x = Value(2.0)
    f = -x
    f.backward()
    assert f.data == -2.0
    assert x.grad == -1.0


def test_sub_grad():
    x, y = Value(2.0), Value(3.0)
    f = x * y - x
    f.backward()
    assert f.data == 4.0
    assert x.grad == 2.0   # y - 1
    assert y.grad == 2.0   # x


def test_rsub():
    x = Value(2.0)
    f = 5 - x
    f.backward()
    assert f.data == 3.0
    assert x.grad == -1.0


# ---------- power ----------

def test_pow_cube():
    x = Value(2.0)
    f = x ** 3
    f.backward()
    assert f.data == 8.0
    assert x.grad == 12.0  # 3x^2


def test_pow_with_product():
    a = Value(2.0)
    fa = a ** 2
    fa.backward()

    b = Value(2.0)
    fb = b * b
    fb.backward()

    assert a.grad == 4.0
    assert b.grad == 4.0
    assert a.grad == b.grad


# ---------- division ----------

def test_rtruediv():
    x = Value(2.0)
    f = 6 / x
    f.backward()
    assert f.data == 3.0
    assert x.grad == -1.5  # -6 / x^2


def test_truediv():
    x = Value(2.0)
    f = x / 2
    f.backward()
    assert f.data == 1.0
    assert x.grad == 0.5


# ---------- ReLU ----------

def test_relu_positive():
    x, y = Value(2.0), Value(3.0)
    f = (x * y).relu()
    f.backward()
    assert f.data == 6.0
    assert x.grad == 3.0
    assert y.grad == 2.0

def test_relu_zero():
    x, y = Value(2.0), Value(0.0)
    f = (x * y).relu()
    f.backward()
    assert f.data == 0.0
    assert x.grad == 0.0
    assert y.grad == 0.0

def test_relu_negative():
    x, y = Value(2.0), Value(-3.0)
    f = (x * y).relu()
    f.backward()
    assert f.data == 0.0
    assert x.grad == 0.0
    assert y.grad == 0.0