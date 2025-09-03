# go_odif/cere.py
import numpy as np

def relu(x):
    return np.maximum(0, x)

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi)*(x + 0.044715*np.power(x,3))))

_ACTS = {"relu": relu, "gelu": gelu}

class CERE:
    """Cheap random MLP with frozen weights. Transform only provided X (caller chooses subsets)."""
    def __init__(self, input_dim: int, d_out: int = 128, depth: int = 1, activation: str = "relu", seed: int = 0):
        self.input_dim = input_dim
        self.d_out = d_out
        self.depth = depth
        self.activation = _ACTS.get(activation, relu)
        self.rng = np.random.RandomState(seed)
        self.W = []
        self.b = []
        d_prev = input_dim
        for _ in range(depth):
            W = self.rng.normal(0, 1/np.sqrt(max(1, d_prev)), size=(d_prev, d_out))
            b = self.rng.normal(0, 0.01, size=(d_out,))
            self.W.append(W)
            self.b.append(b)
            d_prev = d_out

    def transform(self, X):
        H = X
        for l in range(self.depth):
            H = self.activation(H @ self.W[l] + self.b[l])
        return H
