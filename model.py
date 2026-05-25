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

        return self.A2  # Model predictions

    def backward(self, X, y):

        self.m = X.shape[0]

        #  output layer gradient
        self.dZ2 = self.A2.copy()
        self.dZ2[np.arange(self.m), y] -= 1
        self.dZ2 /= self.m

        self.dW2 = np.dot(self.A1.T, self.dZ2)
        self.db2 = np.sum(self.dZ2, axis=0, keepdims=True)

        #  hidden layer(s) gradient(s)
        self.dA1 = np.dot(self.dZ2, self.W2.T)
        self.dZ1 = self.dA1 * relu_derivative(self.Z1)

        self.dW1 = np.dot(X.T, self.dZ1)
        self.db1 = np.sum(self.dZ1, axis=0, keepdims=True)

        return self.dW1, self.db1, self.dW2, self.db2

    def update_parameters(self, dW1, db1, dW2, db2, learning_rate):

        self.W1 -= learning_rate * self.dW1
        self.b1 -= learning_rate * self.db1

        self.W2 -= learning_rate * self.dW2
        self.b2 -= learning_rate * self.db2

    def debug_shapes(self): # Helper function for understanding tensor shapes

        print("=== FORWARD PASS ===")
        print("Z1 :", self.Z1.shape)
        print("A1 :", self.A1.shape)
        print("Z2 :", self.Z2.shape)
        print("A2 :", self.A2.shape)

        print("\n=== BACKWARD PASS ===")
        print("m:", self.m)
        print()
        print("dZ2:", self.dZ2.shape)
        print("dW2:", self.dW2.shape)
        print("db2:", self.db2.shape)
        print()

        print("dA1:", self.dA1.shape)
        print("dZ1:", self.dZ1.shape)
        print()

        print("dW1:", self.dW1.shape)
        print("db1:", self.db1.shape)