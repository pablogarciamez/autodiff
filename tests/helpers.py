import numpy as np

def numericGrad(f, x, eps=1e-6):
    x = np.array(x, dtype=float)
    g = np.zeros_like(x)
    print(x)
    for idx in np.ndindex(x.shape):
        old = x[idx]
        x[idx] = old + eps
        up = f(x)
        x[idx] = old - eps
        down = f(x)
        x[idx] = old
        g[idx] = (up - down) / (2 * eps)
    return g