from model import NeuralNetwork
from utils import load_images, load_labels, accuracy
from loss import cross_entropy_loss

X_train = load_images("data\train-images.idx3-ubyte")
y_train = load_labels("data\train-labels.idx1-ubyte")

model = NeuralNetwork()

epochs = 20
learning_rate = 0.1

for epoch in range(epochs):
    y_pred = model.forward(X_train)

    loss = cross_entropy_loss(y_pred, y_train)
    accuracy_score = accuracy(y_pred, y_train)

    dW1, db1, dW2, db2 = model.backward(X_train, y_train)

    model.update_parameters(dW1, db1, dW2, db2, learning_rate)

    print(f"Epoch: {epoch+1} | Loss: {loss} | Accuracy: {accuracy_score}")