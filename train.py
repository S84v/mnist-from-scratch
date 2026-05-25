from model import NeuralNetwork
from utils import load_images, load_labels, accuracy
from loss import cross_entropy_loss
import os

# Color formatting for terminal output
os.system("")

YELLOW = "\033[93m"
GREEN = "\033[92m"
WHITE = "\033[97m"
RESET = "\033[0m"

X_train = load_images(r"data\train-images.idx3-ubyte")
y_train = load_labels(r"data\train-labels.idx1-ubyte")

model = NeuralNetwork()

epochs = int(input("Epochs: "))
learning_rate = float(input("Learning rate: "))
print()

for epoch in range(epochs):
    y_pred = model.forward(X_train)

    loss = cross_entropy_loss(y_pred, y_train)
    accuracy_score = accuracy(y_pred, y_train)

    dW1, db1, dW2, db2 = model.backward(X_train, y_train)

    model.update_parameters(dW1, db1, dW2, db2, learning_rate)

    print(
        f"{WHITE}Epoch: {epoch+1}{RESET} |{YELLOW} Loss: {loss:.4f}{RESET} |{GREEN} Accuracy: {accuracy_score:.4f}{RESET}"
    )
