import numpy as np

class MatMul:
    def __init__(self, W):
        self.W = W
        self.x = None
        self.dW = None

    def forword(self, x):
        out = np.dot(x, self.W)
        self.x = x
        return out

    def backword(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(dout, self.W.T)
        return dx