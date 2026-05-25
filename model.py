import numpy as np
from activations import relu, relu_derivative, softmax


class NeuralNetwork:

    def __init__(self):
        self.W1 = np.random.randn(784, 128) * 0.01
        self.b1 = np.zeros((1, 128))

        self.W2 = np.random.randn(128, 10) * 0.01
        self.b2 = np.zeros((1, 10))

    def forward(self, X):

        self.Z1 = np.add(np.dot(X, self.W1), self.b1)  # Linear calculation
        self.A1 = relu(
            self.Z1
        )  # Non linear activation function => non linear value calculation

        self.Z2 = np.add(np.dot(self.A1, self.W2), self.b2)  # Linear calculation
        self.A2 = softmax(self.Z2)  # Non linear calculation

        return self.A2

    def backward(self, X, y):

        m = X.shape[0]

        #  output layer gradient
        dZ2 = self.A2.copy()
        dZ2[np.arange(m), y] -= 1
        dZ2 /= m

        dW2 = np.dot(self.A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        #  hidden layer(s) gradient(s)
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * relu_derivative(self.Z1)

        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        return dW1, db1, dW2, db2

    def update_parameters(self, dW1, db1, dW2, db2, learning_rate):

        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
