import numpy as np
from activations import relu, softmax


class NeuralNetwork:

    def __init__(self):
        self.W1 = np.random.randn(784, 128) * 0.01
        self.b1 = np.zeros((1, 128))

        self.W2 = np.random.randn(128, 10) * 0.01
        self.b2 = np.zeros((1, 10))

    def forward(self, X):

        self.Z1 = np.add(np.dot(self.W1.T, X), self.b1)
        self.A1 = relu(self.Z1)

        self.Z2 = np.add(np.dot(self.A1.T, self.W2), self.b2)
        self.A2 = softmax(self.Z2)

        return self.A2
